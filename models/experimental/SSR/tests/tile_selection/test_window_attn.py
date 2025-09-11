# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import WindowAttention
from timm.models.layers import to_2tuple

from models.experimental.SSR.tt.tile_selection import TTWindowAttention
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_window_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "qkv"):  # WindowAttention model
            parameters["qkv"] = {}
            parameters["proj"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16)
            parameters["qkv"]["bias"] = preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16)

            # Preprocess relative position bias
            relative_position_bias = torch_model.relative_position_bias_table[
                torch_model.relative_position_index.view(-1)
            ].view(
                torch_model.window_size[0] * torch_model.window_size[1],
                torch_model.window_size[0] * torch_model.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            parameters["relative_position_bias"] = ttnn.from_torch(
                relative_position_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "input_shape, window_size, num_heads, input_resolution",
    (
        ((1083, 49, 96), (7, 7), 3, (128, 128)),
        ((1083, 49, 96), (7, 7), 3, None),
        ((300, 49, 192), (7, 7), 3, (64, 64)),
        ((300, 49, 192), (7, 7), 3, None),
        ((75, 49, 384), (7, 7), 3, (32, 32)),
        ((75, 49, 384), (7, 7), 3, None),
        ((27, 49, 768), (7, 7), 3, (16, 16)),
        ((27, 49, 768), (7, 7), 3, None),
        ((12, 49, 1536), (7, 7), 3, (8, 8)),
        ((12, 49, 1536), (7, 7), 3, None),
    ),
)
def test_window_attn(device, input_shape, window_size, num_heads, input_resolution):
    x = torch.randn(input_shape)

    qkv_bias = True
    qk_scale = None
    attn_drop = 0.0
    proj_drop = 0.0
    dim = input_shape[-1]

    mask_shape_map = {
        (128, 128): (361, 49, 49),
        (64, 64): (100, 49, 49),
        (32, 32): (25, 49, 49),
        (16, 16): (9, 49, 49),
        (8, 8): (4, 49, 49),
    }

    mask = None
    if input_resolution is not None:
        mask_shape = mask_shape_map[input_resolution]
        mask = torch.zeros(mask_shape)

    ref_layer = WindowAttention(
        dim,
        window_size=to_2tuple(window_size),
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
    )
    ref_output = ref_layer(x, mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
    )
    tt_layer = TTWindowAttention(
        parameters=parameters,
        device=device,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
    )
    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_input = ttnn.to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)
    tt_mask = None
    if mask is not None:
        tt_mask = ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = tt_layer(tt_input, tt_mask)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    if does_pass:
        logger.info("WindowAttn Passed!")
    else:
        logger.error("WindowAttn Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
