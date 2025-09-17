# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.tt.head import TTHead, head_layer_optimisations
from models.experimental.panoptic_deeplab.common import load_torch_model_state
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor


class HeadTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        intermediate_channels,
        out_channels,
        height,
        width,
        model_config,
        name,
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
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.model_config = model_config
        self.name = name

        # Mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        logger.info(f"Initializing Head test for module: {name}")

        # Torch model
        torch_model = HeadModel(self.in_channels, self.intermediate_channels, self.out_channels)
        torch_model = load_torch_model_state(torch_model, name)

        # Synthetic input
        self.torch_input_tensor = self._create_input_tensor()

        # Torch output
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_model,
            run_model=lambda m: m(self.torch_input_tensor),
            device=None,
        )

        # Initialize TTNN model
        self.ttnn_model = TTHead(parameters, model_config, layer_optimisations=head_layer_optimisations[self.name])

        # Run model in phases and validate
        logger.info(f"Running TTNN Head model")

        # Rebuild TTNN input (since buffers may be freed across passes)
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat8_b,
            device=self.device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_tensor = ttnn.to_device(tt_host_tensor, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Optional: reinstantiate TTNN model
        self.ttnn_model = TTHead(parameters, model_config, layer_optimisations=head_layer_optimisations[self.name])

        self.run()
        self.validate()

    def _create_input_tensor(self):
        shape = (self.batch_size * self.num_devices, self.in_channels, self.height, self.width)
        logger.info(f"Generating synthetic input tensor of shape {shape}")
        return torch.randn(shape, dtype=torch.float32)

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
        logger.info("Running TTNN Head model...")
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
        assert passed, logger.error(f"Head PCC check failed: {msg}")

        logger.info(
            f"Head `{self.name}` passed: "
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
    "batch_size, in_channels, intermediate_channels, out_channels, height, width, name",
    [
        (1, 256, 256, 19, 128, 256, "semantic_decoder.head_1"),
        (1, 128, 32, 2, 128, 256, "instance_decoder.head_1"),
        (1, 128, 32, 1, 128, 256, "instance_decoder.head_2"),
    ],
)
def test_head(device, batch_size, in_channels, intermediate_channels, out_channels, height, width, name):
    HeadTestInfra(
        device=device,
        batch_size=batch_size,
        in_channels=in_channels,
        intermediate_channels=intermediate_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        model_config=model_config,
        name=name,
    )
