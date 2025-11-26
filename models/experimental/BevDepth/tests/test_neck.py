# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.BevDepth.tests.ref_bev_depth_neck import BEVDepthHead
from models.experimental.BevDepth.tt.bev_depth_neck import TtBEVDepthHead, neck_optimisations
from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor


class NeckTestInfra:
    def __init__(
        self,
        device,
        model_config,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

        # Core config
        self.device = device
        self.num_devices = device.get_num_devices()
        self.model_config = model_config

        # Mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch model
        torch_model = BEVDepthHead()
        # Note: Load weights if checkpoint is available
        torch_model.load_weights("../reference/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
        torch_model.eval()

        # Synthetic inputs (4 inputs with different channel sizes)
        self.torch_input_tensors = self._create_input_tensors()

        # Torch output
        self.torch_output = torch_model(*self.torch_input_tensors)

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_model,
            run_model=lambda m: m(*self.torch_input_tensors),
            device=None,
        )
        # print(parameters)

        # Initialize TTNN model
        self.ttnn_model = TtBEVDepthHead(parameters, model_config, layer_optimisations=neck_optimisations)

        # Run model in phases and validate
        logger.info(f"Running TTNN Neck model")

        # Rebuild TTNN inputs (since buffers may be freed across passes)
        self.input_tensors = []
        for torch_input in self.torch_input_tensors:
            tt_host_tensor = ttnn.from_torch(
                torch_input.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat8_b,
                device=self.device,
                mesh_mapper=self.inputs_mesh_mapper,
            )
            tt_input = ttnn.to_device(tt_host_tensor, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
            self.input_tensors.append(tt_input)

        # Optional: reinstantiate TTNN model
        self.ttnn_model = TtBEVDepthHead(parameters, model_config, layer_optimisations=neck_optimisations)

        self.run()
        self.validate()

    def _create_input_tensors(self):
        # Create 4 inputs with different channel sizes
        shapes = [
            (2, 160, 128, 128),  # x0
            (2, 160, 64, 64),  # x1
            (2, 320, 32, 32),  # x2
            (2, 640, 16, 16),  # x3
        ]
        logger.info(f"Generating synthetic input tensors with shapes {shapes}")
        return [torch.randn(shape, dtype=torch.float32) for shape in shapes]

    @classmethod
    def get_mesh_mappers(cls, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        logger.info("Running TTNN Neck model...")
        self.ttnn_output = self.ttnn_model(*self.input_tensors, device=self.device)
        return self.ttnn_output

    def _tt_to_torch_nchw(self, tt_tensor, expected_shape):
        torch_tensor = ttnn.to_torch(tt_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        torch_tensor = torch.reshape(
            torch_tensor,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        return torch.permute(torch_tensor, (0, 3, 1, 2))

    def validate(self):
        logger.info("Validating TTNN output against PyTorch...")

        # Convert TTNN tensor to torch format
        tt_tensor_torch = self._tt_to_torch_nchw(self.ttnn_output, self.torch_output.shape)

        # Deallocate to save memory
        ttnn.deallocate(self.ttnn_output)

        pcc_threshold = 0.99
        passed, msg = check_with_pcc(self.torch_output, tt_tensor_torch, pcc=pcc_threshold)

        if not passed:
            logger.error(f"Neck PCC check failed: {msg}")
        else:
            logger.info(f"Neck PCC check passed: {msg}")

        assert passed, f"Neck PCC check failed: {msg}"

        logger.info(
            f"BEVDepth Neck Tested: "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        )

        return True, msg


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_neck(device):
    NeckTestInfra(
        device=device,
        model_config=model_config,
    )
