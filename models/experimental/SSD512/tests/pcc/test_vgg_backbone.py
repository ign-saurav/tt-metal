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
    Conv2dConfiguration,
    MaxPool2dConfiguration,
)
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

# SSD512_L1_SMALL_SIZE = 98304
SSD512_L1_SMALL_SIZE = 2457


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
# @pytest.mark.parametrize("size", ((256,)))
@pytest.mark.parametrize("size", ((512,)))
# @pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_vgg_backbone(device, pcc, size, reset_seeds):
    base = {
        "300": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
    }
    batch_size = 1
    input_channels = 3
    torch_input = torch.randn(batch_size, input_channels, size, size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    cfg = base[str(size)]
    torch_layers = vgg(cfg, i=3, batch_norm=False)
    torch_model = nn.Sequential(*torch_layers)
    torch_model.eval()
    # Export the torch_model to ONNX
    # import os
    # from ttnn.torch_tracer import trace, visualize
    # visualize(torch_model, file_name="model_graph.svg")
    # onnx_filename = "vgg_backbone.onnx"
    # torch.onnx.export(
    #     torch_model,
    #     torch_input,
    #     onnx_filename)
    #     export_params=True,
    #     opset_version=12,
    #     do_constant_folding=True,
    #     # input_names=['input'],
    #     # output_names=['output'],
    #     # dynamic_axes={
    #     #     'input': {0: 'batch_size'},
    #     #     'output': {0: 'batch_size'}
    #     # },
    # )
    # print(f"Exported VGG backbone torch model to {onnx_filename}")
    # model_config = {
    #     "math_fidelity": ttnn.MathFidelity.LoFi,
    #     "weights_dtype": ttnn.bfloat8_b,
    #     "activation_dtype": ttnn.bfloat8_b,
    #     "output_dtype": ttnn.bfloat8_b,
    #     "deallocate_activation":True,
    #     "sharding_strategy":WidthSliceStrategyConfiguration
    # }

    model_config = {
        "weights_dtype": ttnn.bfloat8_b,
        "output_dtype": ttnn.bfloat8_b,
        "activation_dtype": ttnn.bfloat8_b,
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "deallocate_activation": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": False,
        "reallocate_halo_output": True,
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "config_tensors_in_dram": True,
    }

    #############################################33
    # torch_model = nn.ModuleList(torch_layers)
    conv_config_layers = []
    with torch.no_grad():
        x = torch_input
        for i, layer in enumerate(torch_model):
            print(layer.__class__.__name__, x.shape)
            # Create Conv2dConfiguration from the current layer, given torch input height, width, batch_size
            if isinstance(layer, nn.Conv2d):
                conv_config_layers.append(
                    Conv2dConfiguration.from_torch(
                        layer,
                        input_height=x.shape[-2],
                        input_width=x.shape[-1],
                        batch_size=x.shape[0],
                        **model_config,
                    )
                )
            elif isinstance(layer, nn.MaxPool2d):
                conv_config_layers.append(
                    MaxPool2dConfiguration.from_torch(
                        layer,
                        input_height=x.shape[-2],
                        input_width=x.shape[-1],
                        channels=x.shape[-3],
                        batch_size=x.shape[0],
                    )
                )

            # x = torch.nn.functional.relu(layer(x), inplace=True)
            x = layer(x)
        torch_output = x
    ########################################################

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_vgg_backbone = TtVGGBackbone(
        conv_config_layer=conv_config_layers,
        batch_size=batch_size,
        device=device,
    )

    tt_output_ttnn = tt_vgg_backbone(device, ttnn_input_tensor)
    tt_output = ttnn.to_torch(tt_output_ttnn)

    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)
    tt_output = tt_output.float()

    _, pcc_message = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"VGG Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)


# def create_conv_block_config(in_channels, out_channels, input_size, batch_size, device):
#     """Create a typical CNN block: Conv -> Conv -> MaxPool"""

#     input_height, input_width = input_size
#     sharding_strategy = HeightShardedStrategyConfiguration(
#         act_block_h_override=64,
#         reshard_if_not_optimal=False
#     )
