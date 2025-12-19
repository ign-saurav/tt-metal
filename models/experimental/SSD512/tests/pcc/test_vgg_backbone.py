# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# from models.experimental.SSD512.common import SSD512_L1_SMALL_SIZE
# from models.experimental.SSD512.reference.ssd import vgg
# from models.experimental.SSD512.tt.layers.tt_vgg_backbone import build_vgg_backbone, apply_vgg_backbone
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import TtVGGBackbone
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import create_config_layers

SSD512_L1_SMALL_SIZE = 98304
# SSD512_L1_SMALL_SIZE = 2457


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


@pytest.mark.parametrize("pcc", ((0.99),))
# @pytest.mark.parametrize("size", ((16,)))
@pytest.mark.parametrize("size", ((512,)))
# @pytest.mark.parametrize("size", ((256,)))
@pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_vgg_backbone(device, pcc, size, reset_seeds):
    base = {
        "300": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        "256": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
    }
    batch_size = 1
    input_channels = 3
    # input_channels = 64
    torch_input = torch.randn(batch_size, input_channels, size, size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )

    cfg = base[str(size)]
    torch_layers = vgg(cfg, i=3, batch_norm=False)
    torch_model = nn.Sequential(*torch_layers)
    torch_model.eval()

    model_config = {
        "weights_dtype": ttnn.bfloat8_b,
        "output_dtype": ttnn.bfloat8_b,
        "activation_dtype": ttnn.bfloat8_b,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
    }

    # Optimized config for MaxPool2d layers (only includes applicable parameters)
    pool_config = {
        # "dtype": ttnn.bfloat8_b,  # Map activation_dtype to dtype for MaxPool2d
        "dtype": ttnn.bfloat16,  # Map activation_dtype to dtype for MaxPool2d
        "slice_strategy": L1FullSliceStrategyConfiguration(),
    }

    # #############################################33
    # # torch_model = nn.ModuleList(torch_layers)
    # conv_config_layers = []
    # with torch.no_grad():
    #     x = torch_input
    #     for i, layer in enumerate(torch_model):
    #         # if i<2:
    #         #     continue
    #         print("layer_info", layer.__class__.__name__, x.shape)
    #         # Create Conv2dConfiguration from the current layer, given torch input height, width, batch_size
    #         if isinstance(layer, nn.Conv2d):
    #             conv_config_layers.append(
    #                 Conv2dConfiguration.from_torch(
    #                     layer,
    #                     input_height=x.shape[-2],
    #                     input_width=x.shape[-1],
    #                     batch_size=x.shape[0],
    #                     **model_config,
    #                 )
    #             )
    #         elif isinstance(layer, nn.MaxPool2d):
    #             conv_config_layers.append(
    #                 MaxPool2dConfiguration.from_torch(
    #                     layer,
    #                     input_height=x.shape[-2],
    #                     input_width=x.shape[-1],
    #                     channels=x.shape[-3],
    #                     batch_size=x.shape[0],
    #                     **pool_config,
    #                 )
    #             )

    #         # x = torch.nn.functional.relu(layer(x), inplace=True)
    #         x = layer(x)
    #         # if i>=21:
    #         #     break
    #     torch_output = x
    # ########################################################

    conv_config_layers = create_config_layers(torch_model, torch_input=torch_input)
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_vgg_backbone = TtVGGBackbone(
        conv_config_layer=conv_config_layers,
        batch_size=batch_size,
        device=device,
    )

    tt_output_ttnn = tt_vgg_backbone(device, ttnn_input_tensor)
    tt_output = ttnn.to_torch(tt_output_ttnn)

    expected_shape = torch_output.shape
    if tt_output.shape != (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]):
        B, C, H, W = expected_shape
        tt_output = tt_output.reshape(B, H, W, C)

    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)
    tt_output = tt_output.float()

    if tt_output.shape != torch_output.shape:
        logger.error(f"Shape mismatch! PyTorch: {torch_output.shape}, TTNN: {tt_output.shape}")
        min_shape = [min(s1, s2) for s1, s2 in zip(torch_output.shape, tt_output.shape)]
        torch_output = torch_output[tuple(slice(0, s) for s in min_shape)]
        tt_output = tt_output[tuple(slice(0, s) for s in min_shape)]

    _, pcc_message = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"Vgg Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)
