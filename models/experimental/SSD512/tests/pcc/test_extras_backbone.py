# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# from models.experimental.SSD512.reference.ssd import add_extras, extras
# from models.experimental.SSD512.reference.ssd import add_extras
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.SSD512.tt.layers.tt_extras_backbone_ver2 import (
    TtExtrasBackbone,
)
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if len(cfg) == 13:
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]  # Fix padding to match Caffe version (pad=1).
    return layers


@pytest.mark.parametrize("pcc", ((0.99),))
@pytest.mark.parametrize("size", (512,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_extras_backbone(device, pcc, size, reset_seeds):
    extras = {
        "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
        "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
    }
    cfg = extras[str(size)]
    torch_layers = add_extras(cfg, i=1024, batch_norm=False)
    torch_model = nn.ModuleList(torch_layers)
    torch_model.eval()

    batch_size = 1
    input_channels = 1024
    input_height = 64 if size == 512 else 38
    input_width = input_height

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    conv_config_layers = []
    with torch.no_grad():
        x = torch_input
        for i, layer in enumerate(torch_model):
            print(layer.__class__.__name__, x.shape)
            # Create Conv2dConfiguration from the current layer, given torch input height, width, batch_size
            conv_config_layers.append(
                Conv2dConfiguration.from_torch(
                    layer,
                    input_height=x.shape[-2],
                    input_width=x.shape[-1],
                    batch_size=x.shape[0],
                )
            )

            x = torch.nn.functional.relu(layer(x), inplace=True)
        torch_output = x

    tt_extras = TtExtrasBackbone(
        conv_config_layer=conv_config_layers,
        batch_size=batch_size,
        device=device,
    )

    tt_output_ttnn = tt_extras(device, ttnn_input_tensor)
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
    logger.info(f"Extras Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)
