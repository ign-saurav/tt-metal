# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
import numpy as np

from pathlib import Path
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.common import (
    load_torch_model_state,
    preprocess_image,
    save_preprocessed_inputs,
    _populate_all_decoders,
)


class PanopticDeepLabTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.real_input_path = "./models/experimental/panoptic_deeplab/resources/input.png"

        # Initialize torch model
        torch_model = TorchPanopticDeepLab()
        torch_model = load_torch_model_state(torch_model, "panoptic_deeplab")

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            self.torch_input_tensor, self.ttnn_input_tensor, self.original_image, self.original_size = preprocess_image(
                self.real_input_path, self.width, self.height, self.device, self.inputs_mesh_mapper
            )
            base_name = Path(self.real_input_path).stem
            torch_input_path = save_preprocessed_inputs(
                self.torch_input_tensor, "models/experimental/panoptic_deeplab/resources/test_inputs", base_name
            )
            logger.info(f"Preprocessed inputs saved for testing: {torch_input_path}")
            logger.info(f"Loading real input from: {self.real_input_path}")
            self.torch_input_tensor = self.load_real_input(torch_input_path)

            # Verify shape matches expected dimensions
            expected_shape = (batch_size * self.num_devices, in_channels, height, width)
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
        else:
            logger.info("Using random input tensor (no real input provided)")
            input_shape = (batch_size * self.num_devices, in_channels, height, width)
            self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Populate conv_args for decoders via one small warm-up pass
        _populate_all_decoders(torch_model, parameters)

        # Run torch model with bfloat16
        logger.info("Running PyTorch model...")
        self.torch_output_tensor, self.torch_output_tensor_2, self.torch_output_tensor_3 = torch_model(
            self.torch_input_tensor
        )

        # Convert input to TTNN format (NHWC)
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # Initialize TTNN model
        logger.info("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        # First run configures JIT, second run is optimized
        for phase in ("JIT configuration", "optimized"):
            logger.info(f"Running TTNN model pass ({phase})...")
            self.input_tensor = ttnn.to_device(tt_host_tensor, device)
            self.run()
            self.validate()

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

    @classmethod
    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),  # inputs
                None,  # weights
                ttnn.ConcatMeshToTensor(device, dim=0),  # outputs
            )
        return None, None, None

    @staticmethod
    def _tt_to_torch_nchw(tt_tensor, device, mesh_composer, expected_shape):
        """Convert TTNN NHWC tensor back to Torch NCHW and reshape to expected batch/shape."""
        t = ttnn.to_torch(tt_tensor, device=device, mesh_composer=mesh_composer)
        t = torch.reshape(t, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]))
        return torch.permute(t, (0, 3, 1, 2))

    def run(self):
        self.output_tensor, self.output_tensor_2, self.output_tensor_3 = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor, self.output_tensor_2, self.output_tensor_3

    def validate(self):
        """Validate three heads (semantic, offsets, centers) in a uniform loop."""
        checks = [
            ("Semantic Segmentation Head", self.output_tensor, self.torch_output_tensor),
            ("Instance Segmentation Offset Head", self.output_tensor_2, self.torch_output_tensor_2),
            ("Instance Segmentation Center Head", self.output_tensor_3, self.torch_output_tensor_3),
        ]

        self._PCC_THRESH = 0.97

        for name, tt_out, torch_ref in checks:
            out = self._tt_to_torch_nchw(tt_out, self.device, self.output_mesh_composer, torch_ref.shape)
            passed, msg = check_with_pcc(torch_ref, out, pcc=self._PCC_THRESH)
            assert passed, logger.error(f"{name} PCC check failed: {msg}")

            logger.info(
                f"Panoptic DeepLab - {name}: batch_size={self.batch_size}, "
                f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
                f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
                f"math_fidelity={model_config['MATH_FIDELITY']}, "
                f"PCC={msg}, shape={tt_out.shape}"
            )

        return True, f"All heads passed PCC ≥ {self._PCC_THRESH}"


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (1, 3, 512, 1024),
    ],
)
def test_panoptic_deeplab(
    device,
    batch_size,
    in_channels,
    height,
    width,
):
    PanopticDeepLabTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    )
