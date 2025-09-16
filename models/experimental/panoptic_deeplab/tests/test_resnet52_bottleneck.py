# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import check_with_pcc
from torchvision.models.resnet import Bottleneck
from models.experimental.panoptic_deeplab.tt.bottleneck import TTBottleneck, get_bottleneck_optimisation
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.tt.common import load_torch_model_state


class BottleneckTestInfra:
    def __init__(
        self, device, batch_size, inplanes, planes, height, width, stride, dilation, downsample, name, model_config
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Seed once for determinism
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Optional downsample layer
        downsample_conv = None
        if downsample:
            downsample_conv = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        # Torch model
        torch_model = Bottleneck(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample_conv
        )
        torch_model = load_torch_model_state(torch_model, name)

        # Torch input + golden output
        input_shape = (batch_size * self.num_devices, inplanes, height, width)
        self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Preprocess model params
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
        self.ttnn_model = TTBottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            dilation=dilation,
            name=name,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation(name),
        )

        # Move input to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        # Run + validate
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),  # inputs
                None,  # weights
                ttnn.ConcatMeshToTensor(device, dim=0),  # outputs
            )
        return None, None, None

    def run(self):
        self.output_tensor, _ = self.ttnn_model(self.input_tensor, self.device, self.input_tensor.shape)
        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor, device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Free device memory
        ttnn.deallocate(tt_output_tensor)

        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc
        )

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Bottleneck `{self.name}` passed: "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={self.pcc_message}"
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
    [
        # res2
        (1, 128, 64, 128, 256, 1, 1, True, "backbone.res2.0"),
        (1, 256, 64, 128, 256, 1, 1, False, "backbone.res2.1"),
        # res3
        (1, 256, 128, 128, 256, 2, 1, True, "backbone.res3.0"),
        (1, 512, 128, 64, 128, 1, 1, False, "backbone.res3.1"),
        # res4
        (1, 512, 256, 64, 128, 2, 1, True, "backbone.res4.0"),
        (1, 1024, 256, 32, 64, 1, 1, False, "backbone.res4.1"),
        # res5
        (1, 1024, 512, 32, 64, 1, 2, True, "backbone.res5.0"),
        (1, 2048, 512, 32, 64, 1, 4, False, "backbone.res5.1"),
        (1, 2048, 512, 32, 64, 1, 8, False, "backbone.res5.2"),
    ],
)
def test_bottleneck(device, batch_size, inplanes, planes, height, width, stride, dilation, downsample, name):
    BottleneckTestInfra(
        device, batch_size, inplanes, planes, height, width, stride, dilation, downsample, name, model_config
    )
