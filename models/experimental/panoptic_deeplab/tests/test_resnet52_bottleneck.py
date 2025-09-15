# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from torchvision.models.resnet import Bottleneck
from models.experimental.panoptic_deeplab.tt.bottleneck import TTBottleneck, bottleneck_layer_optimisations
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.common import load_torch_model_state


class BottleneckTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        dilation,
        downsample,
        name,
        model_config,
    ):
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
        self.batch_size = batch_size
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.name = name

        downsample_conv = None
        if downsample:
            downsample_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, padding=0, bias=False
                ),
                torch.nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        # torch model
        torch_model = Bottleneck(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample_conv
        )
        torch_model = load_torch_model_state(torch_model, f"backbone.{name}")

        input_shape = (batch_size * self.num_devices, inplanes, height, width)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # golden
        self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        ## ttnn
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        self.ttnn_model = TTBottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            model_config=model_config,
            dilation=dilation,
            name=name,
            layer_optimisations=bottleneck_layer_optimisations[name[:6]],
        )

        # First run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        # Optimized run
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
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
        self.output_tensor, _ = self.ttnn_model(
            self.input_tensor,
            self.device,
            self.input_tensor.shape,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor, device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Deallocate output tesnors
        ttnn.deallocate(tt_output_tensor)

        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        batch_size = tt_output_tensor_torch.shape[0]

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc
        )

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"ResNet50 Bottleneck Block {self.name} - batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, inplanes, planes, height, width, stride, dilation, downsample, name",
    (
        # Layer 1
        (1, 128, 64, 128, 256, 1, 1, True, "layer1.0"),
        (1, 256, 64, 128, 256, 1, 1, False, "layer1.1"),
        # Layer 2
        (1, 256, 128, 128, 256, 2, 1, True, "layer2.0"),
        (1, 512, 128, 64, 128, 1, 1, False, "layer2.1"),
        # Layer 3
        (1, 512, 256, 64, 128, 2, 1, True, "layer3.0"),
        (1, 1024, 256, 32, 64, 1, 1, False, "layer3.1"),
        # Layer 4
        (1, 1024, 512, 32, 64, 1, 2, True, "layer4.0"),
        (1, 2048, 512, 32, 64, 1, 4, False, "layer4.1"),
        (1, 2048, 512, 32, 64, 1, 8, False, "layer4.2"),
    ),
)
def test_bottleneck(device, batch_size, inplanes, planes, height, width, stride, dilation, downsample, name):
    BottleneckTestInfra(
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        dilation,
        downsample,
        name,
        model_config,
    )
