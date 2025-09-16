# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.res_block import (
    TTRes,
    res_layer_optimisations,
)
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.res_block import (
    ResModel,
)


class ResTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        in_channels,
        upsample_channels,
        intermediate_channels,
        out_channels,
        height_res,
        width_res,
        height,
        width,
        name,
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
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.height_res = height_res
        self.width_res = width_res
        self.height = height
        self.width = width
        self.name = name
        self.upsample_channels = upsample_channels
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        self.torch_input_tensor = torch.randn(
            (self.batch_size, self.upsample_channels, self.height, self.width), dtype=torch.float32
        )

        self.torch_res_input_tensor = torch.randn(
            (self.batch_size, self.in_channels, self.height_res, self.width_res), dtype=torch.float32
        )

        # torch model
        torch_model = ResModel(self.in_channels, self.intermediate_channels, self.out_channels).eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = {}
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_model,
            run_model=lambda model: model(self.torch_input_tensor, self.torch_res_input_tensor),
            device=None,
        )

        # Generate input tensors for different model blocks
        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
        self.torch_res_input_tensor = self.torch_res_input_tensor.to(torch.bfloat16)
        self.torch_output_tensor = torch_model(self.torch_input_tensor, self.torch_res_input_tensor)

        # Convert torch tensors to TTNN host tensors (NHWC, bfloat8_b)
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat8_b,
                device=device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)
        tt_host_res_tensor = to_ttnn_host(self.torch_res_input_tensor)

        # Move TTNN host tensors to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.res_input_tensor = ttnn.to_device(tt_host_res_tensor, device)

        # ttnn model
        self.ttnn_model = TTRes(parameters, model_config, layer_optimisations=res_layer_optimisations[self.name])

        # run and validate
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
        self.output_tensor = self.ttnn_model(
            self.input_tensor, self.res_input_tensor, self.upsample_channels, self.device
        )
        return self.output_tensor

    def validate(self, output_tensor=None, output_tensor1=None):
        """Validate outputs"""
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        batch_size = self.batch_size

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab {self.name} - batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, upsample_channels, intermediate_channels, out_channels, height_res, width_res, height, width, name",
    [
        (1, 512, 256, 320, 256, 64, 128, 32, 64, "semantics_res3"),  # semantics res3 block
        (1, 256, 256, 288, 256, 128, 256, 64, 128, "semantics_res2"),  # semantics res2 block
        (1, 512, 256, 320, 128, 64, 128, 32, 64, "instance_res3"),  # instance res3 block
        (1, 256, 128, 160, 128, 128, 256, 64, 128, "instance_res2"),  # instance res2 block
    ],
)
def test_res(
    device,
    batch_size,
    in_channels,
    upsample_channels,
    intermediate_channels,
    out_channels,
    height_res,
    width_res,
    height,
    width,
    name,
):
    ResTestInfra(
        device,
        batch_size,
        model_config,
        in_channels,
        upsample_channels,
        intermediate_channels,
        out_channels,
        height_res,
        width_res,
        height,
        width,
        name,
    )
