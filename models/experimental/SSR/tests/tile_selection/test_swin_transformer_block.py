# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import SwinTransformerBlock
from models.experimental.SSR.tt.tile_selection import TTSwinTransformerBlock
from models.experimental.SSR.tests.common.test_mlp import create_mlp_preprocessor
from models.experimental.SSR.tests.tile_selection.test_window_attn import create_window_attention_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_swin_transformer_block_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if hasattr(torch_model, "attn"):  # SwinTransformerBlock model
            # Preprocess attention parameters
            parameters["attn"] = preprocess_model_parameters(
                initialize_model=lambda: torch_model.attn,
                custom_preprocessor=create_window_attention_preprocessor(device),
                device=device,
            )

            # Preprocess layer normalization parameters
            parameters["norm1"] = {}
            parameters["norm1"]["weight"] = ttnn.from_torch(
                torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm1"]["bias"] = ttnn.from_torch(
                torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            parameters["norm2"] = {}
            parameters["norm2"]["weight"] = ttnn.from_torch(
                torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm2"]["bias"] = ttnn.from_torch(
                torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Preprocess MLP parameters
            parameters["mlp"] = preprocess_model_parameters(
                initialize_model=lambda: torch_model.mlp,
                custom_preprocessor=create_mlp_preprocessor(device),
                device=device,
            )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio",
    (
        # Tile selection testcases
        (3, 128, 128, 96, 3, 7, 0, 4.0),
        (3, 128, 128, 96, 3, 7, 3, 4.0),
        (3, 64, 64, 96, 3, 7, 0, 4.0),
        (3, 64, 64, 96, 3, 7, 3, 4.0),
        (3, 32, 32, 96, 3, 7, 0, 4.0),
        (3, 32, 32, 96, 3, 7, 3, 4.0),
        (3, 16, 16, 96, 3, 7, 0, 4.0),
        (3, 16, 16, 96, 3, 7, 3, 4.0),
        (3, 8, 8, 96, 3, 7, 0, 4.0),
        (3, 8, 8, 96, 3, 7, 3, 4.0),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
def test_swin_transformer_block(device, batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio):
    # Create input tensor
    input_shape = (batch_size, height * width, dim)
    x = torch.randn(input_shape)

    # Create reference model
    ref_layer = SwinTransformerBlock(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )

    # Get reference output
    ref_output = ref_layer(x)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_swin_transformer_block_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    tt_layer = TTSwinTransformerBlock(
        parameters=parameters,
        device=device,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
    )

    # Convert input to TTNN tensor
    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_input = ttnn.to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)

    # Run forward pass
    tt_output = tt_layer(tt_input)

    # Convert back to torch
    tt_torch_output = tt2torch_tensor(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(
        ref_output, tt_torch_output, 0.98
    )  # Slightly lower threshold due to accumulated precision differences

    if does_pass:
        logger.info("SwinTransformerBlock Passed!")
    else:
        logger.warning("SwinTransformerBlock Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
