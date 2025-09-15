# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor


class PanopticDeepLabTestInfra:
    _seeded = False
    _PCC_THRESH = 0.97

    def __init__(self, device, batch_size, in_channels, height, width, model_config):
        super().__init__()
        self._maybe_seed()

        # Core state
        self.device = device
        self.model_config = model_config
        self.num_devices = device.get_num_devices()
        self.batch_size, self.in_channels, self.height, self.width = (
            batch_size,
            in_channels,
            height,
            width,
        )
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch reference model + inputs
        torch_model = TorchPanopticDeepLab().eval()
        input_shape = (batch_size * self.num_devices, in_channels, height, width)
        self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float)

        # Preprocess TTNN parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Populate conv_args for decoders via one small warm-up pass
        self._populate_all_decoders(torch_model, parameters)

        # Run Torch once (fp32) → then bf16 for parity with TTNN
        logger.info("Running PyTorch model...")
        self.torch_output_tensor, self.torch_output_tensor_2, self.torch_output_tensor_3 = torch_model(
            self.torch_input_tensor
        )

        # Convert input to TTNN NHWC host tensor
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # TTNN model
        logger.info("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLab(parameters=parameters, model_config=model_config)

        # First run configures JIT, second run is optimized
        for phase in ("JIT configuration", "optimized"):
            logger.info(f"Running TTNN model pass ({phase})...")
            self.input_tensor = ttnn.to_device(tt_host_tensor, device)
            self.run()
            self.validate()

    # --------------------------- Setup & helpers ---------------------------

    @classmethod
    def _maybe_seed(cls):
        if not cls._seeded:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            cls._seeded = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),  # inputs
                None,  # weights
                ttnn.ConcatMeshToTensor(device, dim=0),  # outputs
            )
        return None, None, None

    @staticmethod
    def _infer_and_set(module, params_holder, attr_name, run_fn):
        """Infer conv args for a TTNN module and set them if present in parameters."""
        if hasattr(params_holder, attr_name):
            args = infer_ttnn_module_args(model=module, run_model=run_fn, device=None)
            getattr(params_holder, attr_name).conv_args = args

    def _populate_decoder(self, torch_dec, params_dec):
        """Warm up a single decoder (semantic or instance) to populate conv_args."""
        if not (torch_dec and params_dec):
            return

        # Synthetic tensors that match typical Panoptic-DeepLab strides
        input_tensor = torch.randn(1, 2048, 32, 64)
        res3_tensor = torch.randn(1, 512, 64, 128)
        res2_tensor = torch.randn(1, 256, 128, 256)

        # ASPP
        self._infer_and_set(torch_dec.aspp, params_dec, "aspp", lambda m: m(input_tensor))
        aspp_out = torch_dec.aspp(input_tensor)

        # res3
        self._infer_and_set(torch_dec.res3, params_dec, "res3", lambda m: m(aspp_out, res3_tensor))
        res3_out = torch_dec.res3(aspp_out, res3_tensor)

        # res2
        self._infer_and_set(torch_dec.res2, params_dec, "res2", lambda m: m(res3_out, res2_tensor))
        res2_out = torch_dec.res2(res3_out, res2_tensor)

        # heads (one or two, if present)
        if hasattr(torch_dec, "head_1"):
            self._infer_and_set(torch_dec.head_1, params_dec, "head_1", lambda m: m(res2_out))
        if hasattr(torch_dec, "head_2"):
            self._infer_and_set(torch_dec.head_2, params_dec, "head_2", lambda m: m(res2_out))

    def _populate_all_decoders(self, torch_model, parameters):
        if hasattr(parameters, "semantic_decoder"):
            self._populate_decoder(torch_model.semantic_decoder, parameters.semantic_decoder)
        if hasattr(parameters, "instance_decoder"):
            self._populate_decoder(torch_model.instance_decoder, parameters.instance_decoder)

    @staticmethod
    def _tt_to_torch_nchw(tt_tensor, device, mesh_composer, expected_shape):
        """Convert TTNN NHWC tensor back to Torch NCHW and reshape to expected batch/shape."""
        t = ttnn.to_torch(tt_tensor, device=device, mesh_composer=mesh_composer)
        t = torch.reshape(t, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]))
        return torch.permute(t, (0, 3, 1, 2))

    # --------------------------- Core runs/validation ---------------------------

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

        for name, tt_out, torch_ref in checks:
            out = self._tt_to_torch_nchw(tt_out, self.device, self.output_mesh_composer, torch_ref.shape)
            passed, msg = check_with_pcc(torch_ref, out, pcc=self._PCC_THRESH)
            assert passed, logger.error(f"{name} PCC check failed: {msg}")

            logger.info(
                f"Panoptic DeepLab - {name}: batch_size={self.batch_size}, "
                f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
                f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
                f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
                f"PCC={msg}, shape={tt_out.shape}"
            )

        return True, f"All heads passed PCC ≥ {self._PCC_THRESH}"


# --------------------------- Test config ---------------------------

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size, in_channels, height, width", [(1, 3, 512, 1024)])
def test_panoptic_deeplab(device, batch_size, in_channels, height, width):
    PanopticDeepLabTestInfra(device, batch_size, in_channels, height, width, model_config)
