# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
from models.experimental.panoptic_deeplab.tt.stem import resnet52Stem, neck_optimisations
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.common import load_torch_model_state


class Resnet52StemTestInfra:
    def __init__(self, device, batch_size, inplanes, planes, height, width, stride, model_config, name):
        super().__init__()
        self._init_seeds()
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.name = name

        # Build reference torch model
        torch_model = DeepLabStem(in_channels=inplanes, out_channels=planes, stride=stride)
        torch_model = load_torch_model_state(torch_model, name)

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Prepare golden inputs/outputs
        input_shape = (self.batch_size, inplanes, height, width)
        self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Convert input to TTNN format
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        # Build TTNN model
        self.ttnn_model = resnet52Stem(
            parameters=parameters,
            stride=stride,
            model_config=model_config,
            layer_optimisations=neck_optimisations,
        )

        # Run + validate
        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor

    def validate(self, model_config, output_tensor=None):
        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensor
        ttnn.deallocate(tt_output_tensor)

        # Reshape + permute back to NCHW
        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        # PCC validation
        pcc_passed, pcc_message = check_with_pcc(self.torch_output_tensor, tt_output_tensor_torch, pcc=0.99)
        assert pcc_passed, logger.error(f"PCC check failed: {pcc_message}")

        logger.info(
            f"ResNet52 Stem Block [{self.name}] - "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={pcc_message}"
        )
        return pcc_passed, pcc_message


# Default model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, inplanes, planes, height, width, stride, name",
    [
        (1, 3, 128, 512, 1024, 1, "backbone.stem"),
    ],
)
def test_stem(device, batch_size, inplanes, planes, height, width, stride, name):
    Resnet52StemTestInfra(
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        model_config,
        name,
    )
