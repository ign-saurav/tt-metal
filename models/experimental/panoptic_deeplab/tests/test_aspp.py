# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
from models.experimental.panoptic_deeplab.tt.aspp import TTASPP
from models.experimental.panoptic_deeplab.tt.common import load_torch_model_state
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor


class ASPPTestInfra:
    def __init__(self, device, batch_size, input_channels, height, width, model_config, name):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

        # Initialize core config
        self.device = device
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.model_config = model_config
        self.name = name
        self.num_devices = device.get_num_devices()

        # Mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        logger.info(f"Initializing ASPP test for module: {name}")

        # Torch model
        torch_model = ASPPModel()
        torch_model = load_torch_model_state(torch_model, name)

        # Create synthetic input
        self.torch_input_tensor = self._create_input_tensor()

        # Run torch model
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Preprocess model
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Initialize TTNN model
        self.ttnn_model = TTASPP(parameters, model_config)

        # Prepare TTNN input
        logger.info("Converting input to TTNN tensor...")

        # Run model and validate
        for phase in ("JIT configuration", "optimized"):
            logger.info(f"Running TTNN model pass ({phase})...")

            # Re-convert input tensor (TTNN may deallocate buffers)
            tt_host_tensor = ttnn.from_torch(
                self.torch_input_tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat8_b,
                device=self.device,
                mesh_mapper=self.inputs_mesh_mapper,
            )
            self.input_tensor = ttnn.to_device(tt_host_tensor, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Optional: Re-instantiate model if it's not stateless
            self.ttnn_model = TTASPP(parameters, self.model_config)

            self.run()
            self.validate()

    def _create_input_tensor(self):
        shape = (self.batch_size * self.num_devices, self.input_channels, self.height, self.width)
        logger.info(f"Generating synthetic input tensor of shape {shape}")
        return torch.randn(shape, dtype=torch.float32)

    @classmethod
    def get_mesh_mappers(cls, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),  # inputs
                None,  # weights
                ttnn.ConcatMeshToTensor(device, dim=0),  # outputs
            )
        return None, None, None

    def run(self):
        logger.info("Running TTNN ASPP model...")
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor

    def _tt_to_torch_nchw(self, tt_tensor, expected_shape):
        torch_tensor = ttnn.to_torch(tt_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        torch_tensor = torch.reshape(
            torch_tensor,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        return torch.permute(torch_tensor, (0, 3, 1, 2))

    def validate(self):
        logger.info("Validating TTNN output against PyTorch...")
        tt_output_tensor_torch = self._tt_to_torch_nchw(self.output_tensor, self.torch_output_tensor.shape)

        # Deallocate to save memory
        ttnn.deallocate(self.output_tensor)

        pcc_threshold = 0.99
        passed, msg = check_with_pcc(self.torch_output_tensor, tt_output_tensor_torch, pcc=pcc_threshold)
        assert passed, logger.error(f"ASPP PCC check failed: {msg}")

        logger.info(
            f"ASPP layer `{self.name}` passed: "
            f"batch_size={self.batch_size}, "
            f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
            f"PCC={msg}"
        )

        return True, msg


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, height, width",
    [
        (1, 2048, 32, 64),
    ],
)
@pytest.mark.parametrize("name", ["semantic_decoder.aspp", "instance_decoder.aspp"])
def test_aspp(device, batch_size, input_channels, height, width, name):
    ASPPTestInfra(
        device=device,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
        model_config=model_config,
        name=name,
    )
