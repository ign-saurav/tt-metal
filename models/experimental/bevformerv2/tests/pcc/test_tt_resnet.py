# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from loguru import logger

from models.experimental.bevformerv2.reference.resnet import resnet50_mmdet, ResNet
from models.experimental.bevformerv2.tt.tt_resnet import TtResNet50_MMD_C345
from models.experimental.bevformerv2.common import load_resnet50_backbone_weights
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
    preprocess_model_parameters,
)


def _pack_conv_tensors(weight, bias):
    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
        "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
    }


def _custom_resnet_preprocessor(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        resnet_params = {}

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        resnet_params["conv1"] = _pack_conv_tensors(weight, bias)

        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                block_params = {}
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    bn = getattr(block, f"bn{conv_name[-1]}")
                    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
                    block_params[conv_name] = _pack_conv_tensors(weight, bias)

                if hasattr(block, "downsample") and block.downsample is not None:
                    conv = block.downsample[0]
                    bn = block.downsample[1]
                    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
                    block_params["downsample"] = _pack_conv_tensors(weight, bias)

                resnet_params[f"layer{layer_idx}_{block_idx}"] = block_params

        parameters["res_model"] = resnet_params
    return parameters


def _prepare_resnet_parameters(model: ResNet, example_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=_custom_resnet_preprocessor,
        device=device,
    )

    conv_args = infer_ttnn_module_args(model=model, run_model=lambda mod: mod(example_input), device=None)
    for key in conv_args.keys():
        if hasattr(model, key):
            conv_args[key].module = getattr(model, key)

    parameters.conv_args = conv_args
    return parameters


class ResNetTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Seed once for determinism
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed_all = []
        self.pcc_message_all = []
        self.device = device
        self.batch_size = batch_size

        # Reference model
        reference_model = resnet50_mmdet(out_indices=(1, 2, 3))
        load_resnet50_backbone_weights(reference_model)
        reference_model.eval()

        # Torch input + golden output
        torch_input = torch.randn(batch_size, in_channels, height, width)
        self.torch_outputs = reference_model(torch_input)

        # Prepare parameters
        parameters = _prepare_resnet_parameters(reference_model, torch_input, device)

        # Convert input to TTNN format
        nhwc = torch_input.permute(0, 2, 3, 1).contiguous()
        nhwc = nhwc.reshape(1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[3])
        self.input_tensor = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, device=device)

        # TTNN model
        self.ttnn_model = TtResNet50_MMD_C345(parameters.conv_args, parameters.res_model, device)

        # Run + validate
        self.run()
        self.validate()

    def run(self):
        self.tt_outputs = self.ttnn_model(self.input_tensor, batch_size=self.batch_size)
        return self.tt_outputs

    def validate(self, tt_outputs=None):
        tt_outputs = self.tt_outputs if tt_outputs is None else tt_outputs

        valid_pcc = 0.99
        for level_idx, (torch_level, tt_level) in enumerate(zip(self.torch_outputs, tt_outputs), start=3):
            converted = ttnn.to_torch(tt_level)
            converted = converted.reshape(
                torch_level.shape[0], torch_level.shape[2], torch_level.shape[3], torch_level.shape[1]
            )
            converted = converted.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # Free device memory
            ttnn.deallocate(tt_level)

            pcc_passed, pcc_message = check_with_pcc(torch_level, converted, pcc=valid_pcc)
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)

            assert pcc_passed, logger.error(f"PCC check failed for C{level_idx}: {pcc_message}")
            logger.info(f"PCC(C{level_idx}) = {pcc_message}")

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        logger.info(f"ResNet50_MMD_C345 passed: " f"batch_size={self.batch_size}, " f"PCC={self.pcc_message_all}")

        return self.pcc_passed_all, self.pcc_message_all


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (2, 3, 256, 256),
    ],
)
def test_bevformerv2_resnet_matches_reference(device, reset_seeds, batch_size, in_channels, height, width):
    ResNetTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
    )
