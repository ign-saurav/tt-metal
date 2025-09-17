# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.SSR.tt.tile_selection import TTBasicLayer, TTPatchMerging
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.SSR.tests.tile_selection.test_swin_transformer_block import (
    create_swin_transformer_block_preprocessor,
)
from models.experimental.SSR.tests.tile_selection.test_patch_merging import create_patch_merging_preprocessor
from models.experimental.SSR.reference.SSR.model.net_blocks import PatchMerging, BasicLayer


def create_basic_layer_preprocessor(device, dim, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {"blocks": {}}

        # Process each transformer block
        for i, block in enumerate(torch_model.blocks):
            params["blocks"][i] = preprocess_model_parameters(
                initialize_model=lambda: block,
                custom_preprocessor=create_swin_transformer_block_preprocessor(device, weight_dtype),
                device=device,
            )

        # Process downsampling layer if present
        if torch_model.downsample is not None:
            params["downsample"] = preprocess_model_parameters(
                initialize_model=lambda: torch_model.downsample,
                custom_preprocessor=create_patch_merging_preprocessor(device, dim, weight_dtype=weight_dtype),
                device=device,
            )

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, input_resolution, dim, depth, num_heads, window_size, has_downsample",
    [
        (3, (128, 128), 96, 2, 3, 7, True),
        (3, (64, 64), 192, 2, 3, 7, True),
        (3, (32, 32), 384, 2, 3, 7, True),
        (3, (16, 16), 768, 2, 3, 7, True),
        (3, (8, 8), 1536, 2, 3, 7, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_basic_layer(
    device, batch_size, input_resolution, dim, depth, num_heads, window_size, has_downsample, input_dtype, weight_dtype
):
    torch.manual_seed(0)

    H, W = input_resolution

    # Create reference model
    downsample = PatchMerging if has_downsample else None
    ref_layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=4.0,
        downsample=downsample,
    )
    ref_layer.eval()

    # Create input tensor [B, H*W, C]
    input_tensor = torch.randn(batch_size, H * W, dim)

    # Reference forward pass
    ref_output = ref_layer(input_tensor)

    # Create ttnn model
    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_basic_layer_preprocessor(device, dim, weight_dtype),
        device=device,
    )

    tt_layer = TTBasicLayer(
        device=device,
        parameters=params,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=4.0,
        downsample=TTPatchMerging if has_downsample else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=input_dtype,
    )

    # Convert input to ttnn
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)

    # ttnn forward pass
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.98)
    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("BasicLayer Passed!")
    else:
        logger.warning("BasicLayer Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
