# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.SSR.tt import TTBasicLayer
from models.experimental.SSR.tt.patch_merging import TTPatchMerging
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor, comp_pcc

from models.experimental.SSR.tests.test_swin_transformer_block import create_swin_transformer_block_preprocessor
from models.experimental.SSR.tests.test_patch_merging import create_patch_merging_preprocessor
from models.experimental.SSR.reference.SSR.model.net_blocks import PatchMerging, BasicLayer

import collections


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def create_basic_layer_preprocessor(device, dim):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {"blocks": {}}

        # Process each transformer block
        for i, block in enumerate(torch_model.blocks):
            params["blocks"][i] = preprocess_model_parameters(
                initialize_model=lambda: block,
                custom_preprocessor=create_swin_transformer_block_preprocessor(device),
                device=device,
            )

        # Process downsampling layer if present
        if torch_model.downsample is not None:
            params["downsample"] = preprocess_model_parameters(
                initialize_model=lambda: torch_model.downsample,
                custom_preprocessor=create_patch_merging_preprocessor(device, dim),
                device=device,
            )

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, input_resolution, dim, depth, num_heads, window_size, has_downsample",
    [
        (3, (128, 128), 96, 2, 3, 7, True),  # Custom configuration
        (3, (64, 64), 192, 2, 3, 7, True),  # Custom configuration
        (3, (32, 32), 384, 2, 3, 7, True),  # Custom configuration
        (3, (16, 16), 768, 2, 3, 7, True),  # Custom configuration
        (3, (8, 8), 1536, 2, 3, 7, True),  # Custom configuration
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
def test_basic_layer(device, batch_size, input_resolution, dim, depth, num_heads, window_size, has_downsample):
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
        custom_preprocessor=create_basic_layer_preprocessor(device, dim),
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
    )

    # Convert input to ttnn
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # ttnn forward pass
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.98)

    logger.info(f"Input resolution: {input_resolution}, Dim: {dim}, Depth: {depth}")
    logger.info(f"Num heads: {num_heads}, Window size: {window_size}, Has downsample: {has_downsample}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("BasicLayer Passed!")
    else:
        logger.warning("BasicLayer Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Verify output shapes match
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"

    # Verify expected output shape based on downsampling
    if has_downsample:
        expected_shape = (batch_size, (H // 2) * (W // 2), 2 * dim)
    else:
        expected_shape = (batch_size, H * W, dim)
    assert ref_output.shape == expected_shape, f"Unexpected output shape: {ref_output.shape}, expected {expected_shape}"
