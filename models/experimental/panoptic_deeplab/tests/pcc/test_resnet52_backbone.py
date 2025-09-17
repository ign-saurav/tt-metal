# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.common import load_torch_model_state


class BackboneTestInfra:
    def __init__(self, device, batch_size, in_channels, height, width, model_config, name):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Only seed once
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size * self.num_devices
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch model + weights
        torch_model = TorchBackbone()
        torch_model = load_torch_model_state(torch_model, name)

        # Torch input + golden output
        self.torch_input_tensor = torch.randn((self.batch_size, in_channels, height, width), dtype=torch.float)
        self.torch_output = torch_model(self.torch_input_tensor)

        # Preprocess parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Convert input to TTNN host tensor
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)

        # TTNN model
        self.ttnn_model = TTBackbone(parameters=parameters, model_config=model_config)

        # Move input to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        # Run + validate
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output = self.output_tensor if output_tensor is None else output_tensor

        valid_pcc = {"res_2": 0.99, "res_3": 0.99, "res_5": 0.99}
        self.pcc_passed_all = []
        self.pcc_message_all = []

        for key in tt_output:
            tt_output_tensor_torch = ttnn.to_torch(
                tt_output[key],
                dtype=self.torch_output[key].dtype,
                device=self.device,
                mesh_composer=self.output_mesh_composer,
            )

            # Free device memory
            ttnn.deallocate(tt_output[key])

            expected_shape = self.torch_output[key].shape
            tt_output_tensor_torch = torch.reshape(
                tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
            )
            tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

            pcc_passed, pcc_message = check_with_pcc(self.torch_output[key], tt_output_tensor_torch, pcc=valid_pcc[key])
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        logger.info(
            f"ResNet52 Backbone - batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={self.pcc_message_all}"
        )

        return self.pcc_passed_all, self.pcc_message_all


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width, name",
    [
        (1, 3, 512, 1024, "backbone"),
    ],
)
def test_backbone(device, batch_size, in_channels, height, width, name):
    BackboneTestInfra(device, batch_size, in_channels, height, width, model_config, name)
