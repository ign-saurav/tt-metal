# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.dependency import FPN
from models.experimental.MapTR.tt.fpn import TtFPN
from models.tt_cnn.tt.builder import Conv2dConfiguration
from tests.ttnn.utils_for_testing import assert_with_pcc


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

FPN_LAYER = "img_neck."


def load_maptr_fpn_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}. Please download the weights first.")

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint)

    fpn_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(FPN_LAYER):
            relative_key = key[len(FPN_LAYER) :]
            if "lateral_convs.0." in relative_key:
                relative_key = relative_key.replace("lateral_convs.0.", "lateral_convs.")
            if "fpn_convs.0." in relative_key:
                relative_key = relative_key.replace("fpn_convs.0.", "fpn_convs.")
            fpn_weights[relative_key] = value

    logger.info(f"Loaded {len(fpn_weights)} weight tensors for FPN")
    return fpn_weights


def load_torch_model_maptr(torch_model: FPN, weights_path: str = MAPTR_WEIGHTS_PATH):
    fpn_weights = load_maptr_fpn_weights(weights_path)
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in fpn_weights:
            new_state_dict[model_key] = fpn_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


def create_conv_config_from_conv(
    conv: torch.nn.Conv2d,
    input_height: int,
    input_width: int,
    batch_size: int,
    weight_ttnn: ttnn.Tensor,
    bias_ttnn: ttnn.Tensor = None,
    activation: ttnn.UnaryWithParam = None,
    deallocate_activation: bool = False,
) -> Conv2dConfiguration:
    kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
    padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=conv.groups,
        dilation=dilation,
        weight=weight_ttnn,
        bias=bias_ttnn,
        activation=activation,
        deallocate_activation=deallocate_activation,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_fpn(device, reset_seeds):
    in_channels = [2048]
    out_channels = 256
    num_outs = 1

    torch_model = FPN(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs)
    torch_model = load_torch_model_maptr(torch_model)

    batch_size = 1
    height = 12
    width = 20
    input_tensor = torch.randn(batch_size, in_channels[0], height, width)
    inputs = [input_tensor]

    torch_output = torch_model(inputs)
    logger.info(f"PyTorch output shape: {torch_output[0].shape}")

    _, _, input_height, input_width = input_tensor.shape
    lateral_input_h = input_height
    lateral_input_w = input_width
    fpn_input_h = input_height
    fpn_input_w = input_width

    lateral_weight_torch = torch_model.lateral_convs[0].conv.weight.data
    lateral_weight_ttnn = ttnn.from_torch(lateral_weight_torch, dtype=ttnn.float32)
    lateral_bias_ttnn = None
    if torch_model.lateral_convs[0].conv.bias is not None:
        lateral_bias_torch = torch_model.lateral_convs[0].conv.bias.data
        lateral_bias_ttnn = ttnn.from_torch(lateral_bias_torch.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    fpn_weight_torch = torch_model.fpn_convs[0].conv.weight.data
    fpn_weight_ttnn = ttnn.from_torch(fpn_weight_torch, dtype=ttnn.float32)
    fpn_bias_ttnn = None
    if torch_model.fpn_convs[0].conv.bias is not None:
        fpn_bias_torch = torch_model.fpn_convs[0].conv.bias.data
        fpn_bias_ttnn = ttnn.from_torch(fpn_bias_torch.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    lateral_conv_config = create_conv_config_from_conv(
        conv=torch_model.lateral_convs[0].conv,
        input_height=lateral_input_h,
        input_width=lateral_input_w,
        batch_size=batch_size,
        weight_ttnn=lateral_weight_ttnn,
        bias_ttnn=lateral_bias_ttnn,
        deallocate_activation=True,
    )

    fpn_conv_config = create_conv_config_from_conv(
        conv=torch_model.fpn_convs[0].conv,
        input_height=fpn_input_h,
        input_width=fpn_input_w,
        batch_size=batch_size,
        weight_ttnn=fpn_weight_ttnn,
        bias_ttnn=fpn_bias_ttnn,
        deallocate_activation=False,
    )

    tt_model = TtFPN(
        lateral_conv_config=lateral_conv_config,
        fpn_conv_config=fpn_conv_config,
        device=device,
    )

    input_tt = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    inputs_tt = [input_tt]

    tt_output = tt_model(inputs_tt)

    tt_output_list = []
    for out in tt_output:
        out_torch = ttnn.to_torch(out)
        out_torch = out_torch.permute(0, 3, 1, 2)
        tt_output_list.append(out_torch)

    for i, (tt_out, torch_out) in enumerate(zip(tt_output_list, torch_output)):
        pcc_passed, pcc_message = assert_with_pcc(tt_out, torch_out, 0.99)
        logger.info(f"FPN output {i} {pcc_message}")
