# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

from models.experimental.SSD512.common import SSD512_L1_SMALL_SIZE
from models.experimental.SSD512.reference.ssd import vgg, base
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import build_vgg_backbone, apply_vgg_backbone
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("pcc", ((0.99),))
@pytest.mark.parametrize("size", ((512,)))
@pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_vgg_backbone(device, pcc, size, reset_seeds):
    cfg = base[str(size)]
    torch_layers = vgg(cfg, i=3, batch_norm=False)
    torch_model = nn.Sequential(*torch_layers)
    torch_model.eval()

    batch_size = 1
    input_channels = 3
    torch_input = torch.randn(batch_size, input_channels, size, size)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    layers_config = build_vgg_backbone(size=size, input_channels=input_channels, device=device)
    torch_conv_idx = 0
    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
            while torch_conv_idx < len(torch_model):
                torch_layer = torch_model[torch_conv_idx]
                if isinstance(torch_layer, nn.Conv2d):
                    break
                torch_conv_idx += 1

            torch_conv = torch_model[torch_conv_idx]
            torch_conv_idx += 1

            weight = torch_conv.weight.data.clone()
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

            weight_ttnn = ttnn.from_torch(weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            bias_ttnn = None
            if bias is not None:
                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                )

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            layers_with_weights.append(layer_with_weights)
        else:
            layers_with_weights.append(layer.copy())

    tt_output_ttnn = apply_vgg_backbone(
        torch_input, layers_with_weights, device=device, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_output = ttnn.to_torch(tt_output_ttnn)

    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)
    tt_output = tt_output.float()

    _, pcc_message = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"VGG Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)
