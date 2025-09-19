# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import numpy as np

import ttnn
import os
from pathlib import Path
from models.experimental.panoptic_deeplab.common import load_torch_model_state
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.panoptic_deeplab.common import (
    load_torch_model_state,
    preprocess_image,
    save_preprocessed_inputs,
    _populate_all_decoders,
)
from tests.ttnn.utils_for_testing import check_with_pcc


class PanopticDeepLabPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        model_location_generator=None,
        resolution=(512, 1024),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.math_fidelity = math_fidelity
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.real_input_path = input_path

        self.torch_model = TorchPanopticDeepLab()
        self.torch_model = load_torch_model_state(self.torch_model, "panoptic_deeplab", self.model_location_generator)

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            self.torch_input_tensor, _, _, _ = preprocess_image(
                self.real_input_path, self.resolution[1], self.resolution[0], self.device, self.inputs_mesh_mapper
            )
            base_name = Path(self.real_input_path).stem
            torch_input_path = save_preprocessed_inputs(
                self.torch_input_tensor, "models/experimental/panoptic_deeplab/resources/test_inputs", base_name
            )
            self.torch_input_tensor = self.load_real_input(torch_input_path)

            # Verify shape matches expected dimensions
            expected_shape = (batch_size * self.num_devices, 3, self.resolution[0], self.resolution[1])
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
        else:
            self.torch_input_tensor = torch.randn(
                (self.batch_size, 3, self.resolution[0], self.resolution[1]), dtype=torch.float32
            )

        # Preprocess model parameters
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Populate conv_args for decoders via one small warm-up pass
        _populate_all_decoders(self.torch_model, self.parameters)

        self.ttnn_model = TTPanopticDeepLab(
            parameters=self.parameters,
            model_config={
                "MATH_FIDELITY": self.math_fidelity,
                "WEIGHTS_DTYPE": self.weight_dtype,
                "ACTIVATIONS_DTYPE": self.act_dtype,
            },
        )

        (
            self.torch_output_sem_seg,
            self.torch_output_ins_seg_offset,
            self.torch_output_ins_seg_center,
        ) = self.torch_model(self.torch_input_tensor)
        self.torch_input_tensor = self.torch_input_tensor.permute(0, 2, 3, 1)

    def setup_dram_interleaved_input(self, torch_input_tensor=None, mesh_mapper=None):
        # Inputs to Panoptic deeplab need to be in ttnn.DRAM_MEMORY_CONFIG for supporting DRAM sliced Conv2d
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    @staticmethod
    def _tt_to_torch_nchw(tt_tensor, device, mesh_composer, expected_shape):
        """Convert TTNN NHWC tensor back to Torch NCHW and reshape to expected batch/shape."""
        t = ttnn.to_torch(tt_tensor, device=device, mesh_composer=mesh_composer)
        t = torch.reshape(t, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]))
        return torch.permute(t, (0, 3, 1, 2))

    def load_real_input(self, input_path: str) -> torch.Tensor:
        """Load real input from saved file"""

        if input_path.endswith(".pt"):
            # Load PyTorch tensor
            data = torch.load(input_path, map_location="cpu")
            if isinstance(data, dict):
                tensor = data["tensor"]
                logger.info(f"Loaded input metadata: {data.keys()}")
                if "stats" in data:
                    logger.info(f"Original input stats: {data['stats']}")
            else:
                tensor = data
        elif input_path.endswith(".npy"):
            # Load numpy array
            np_array = np.load(input_path)
            tensor = torch.from_numpy(np_array)
        else:
            raise ValueError(f"Unsupported input file format: {input_path}")

        # Ensure tensor is float32
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        return tensor

    def run(self):
        self.tt_output_sem_seg, self.tt_output_ins_seg_offset, self.tt_output_ins_seg_center = self.ttnn_model(
            self.input_tensor, self.device
        )

    def validate(self):
        """Validate three heads (semantic, offsets, centers) in a uniform loop."""
        checks = [
            ("Semantic Segmentation Head", self.tt_output_sem_seg, self.torch_output_sem_seg),
            ("Instance Segmentation Offset Head", self.tt_output_ins_seg_offset, self.torch_output_ins_seg_offset),
            ("Instance Segmentation Center Head", self.tt_output_ins_seg_center, self.torch_output_ins_seg_center),
        ]

        self._PCC_THRESH = 0.97
        self.pcc_passed = self.pcc_message = []

        for name, tt_out, torch_ref in checks:
            out = self._tt_to_torch_nchw(tt_out, self.device, self.outputs_mesh_composer, torch_ref.shape)
            passed, msg = check_with_pcc(torch_ref, out, pcc=self._PCC_THRESH)
            self.pcc_passed.append(passed)
            self.pcc_message.append(msg)

        assert all(self.pcc_passed), logger.error(f"{name} PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - {name}: batch_size={self.batch_size}, "
            f"act_dtype={self.act_dtype}, "
            f"weight_dtype={self.weight_dtype}, "
            f"math_fidelity={self.math_fidelity}, "
            f"PCC Semantic={self.pcc_message[0]}, "
            f"PCC Instance Offset={self.pcc_message[1]}, "
            f"PCC Instance Center={self.pcc_message[2]}, "
            f"shape={tt_out.shape}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.tt_output_sem_seg)
        ttnn.deallocate(self.tt_output_ins_seg_offset)
        ttnn.deallocate(self.tt_output_ins_seg_center)
