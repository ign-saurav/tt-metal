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

from models.experimental.SSR.reference.SSR.model.net_blocks import PatchMerging, BasicLayer

import collections


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# def create_basic_layer_preprocessor(device):
#     def custom_preprocessor(torch_model, name, ttnn_module_args):
#         params = {"blocks": {}}

#         # Process each transformer block
#         for i, block in enumerate(torch_model.blocks):
#             params["blocks"][i] = {
#                 "norm1": {
#                     "weight": ttnn.from_torch(block.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                     "bias": ttnn.from_torch(block.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                 },
#                 "norm2": {
#                     "weight": ttnn.from_torch(block.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                     "bias": ttnn.from_torch(block.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                 },
#                 "mlp": {
#                     "fc1": {
#                         "weight": ttnn.from_torch(block.mlp.fc1.weight.transpose(0, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                         "bias": ttnn.from_torch(block.mlp.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                     },
#                     "fc2": {
#                         "weight": ttnn.from_torch(block.mlp.fc2.weight.transpose(0, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                         "bias": ttnn.from_torch(block.mlp.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                     },
#                 },
#                 "attn": {}  # Simplified attention parameters
#             }

#         # Process downsampling layer if present
#         if torch_model.downsample is not None:
#             params["downsample"] = {
#                 "reduction": {
#                     "weight": ttnn.from_torch(
#                         torch_model.downsample.reduction.weight.transpose(0, 1),
#                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
#                     )
#                 },
#                 "norm": {
#                     "weight": ttnn.from_torch(torch_model.downsample.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                     "bias": ttnn.from_torch(torch_model.downsample.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
#                 }
#             }

#         return params

#     return custom_preprocessor
#


def create_basic_layer_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {"blocks": {}}
        window_size = to_2tuple(torch_model.window_size)

        # Process each transformer block
        for i, block in enumerate(torch_model.blocks):
            relative_position_bias = block.attn.relative_position_bias_table[
                block.attn.relative_position_index.view(-1)
            ].view(
                window_size[0] * window_size[1],
                window_size[0] * window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            params["blocks"][i] = {
                "norm1": {
                    "weight": ttnn.from_torch(
                        block.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        block.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                },
                "norm2": {
                    "weight": ttnn.from_torch(
                        block.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        block.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                },
                "mlp": {
                    "fc1": {
                        "weight": ttnn.from_torch(
                            block.mlp.fc1.weight.transpose(0, 1),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        ),
                        "bias": ttnn.from_torch(
                            block.mlp.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                        ),
                    },
                    "fc2": {
                        "weight": ttnn.from_torch(
                            block.mlp.fc2.weight.transpose(0, 1),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        ),
                        "bias": ttnn.from_torch(
                            block.mlp.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                        ),
                    },
                },
                "attn": {
                    # QKV projection weights
                    "qkv": {
                        "weight": ttnn.from_torch(
                            block.attn.qkv.weight.transpose(0, 1),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                    },
                    # Output projection weights
                    "proj": {
                        "weight": ttnn.from_torch(
                            block.attn.proj.weight.transpose(0, 1),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                    },
                    "relative_position_bias": ttnn.from_torch(
                        relative_position_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    ),
                },
            }

            # Add QKV bias if present
            if hasattr(block.attn.qkv, "bias") and block.attn.qkv.bias is not None:
                params["blocks"][i]["attn"]["qkv"]["bias"] = ttnn.from_torch(
                    block.attn.qkv.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            # Add projection bias if present
            if hasattr(block.attn.proj, "bias") and block.attn.proj.bias is not None:
                params["blocks"][i]["attn"]["proj"]["bias"] = ttnn.from_torch(
                    block.attn.proj.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        # Process downsampling layer if present
        if torch_model.downsample is not None:
            params["downsample"] = {
                "reduction": {
                    "weight": ttnn.from_torch(
                        torch_model.downsample.reduction.weight.transpose(0, 1),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                },
                "norm": {
                    "weight": ttnn.from_torch(
                        torch_model.downsample.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        torch_model.downsample.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                },
            }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, input_resolution, dim, depth, num_heads, window_size, has_downsample",
    [
        (1, (56, 56), 96, 2, 3, 7, False),  # Swin-Tiny stage 1 without downsample
        (1, (56, 56), 96, 2, 3, 7, True),  # Swin-Tiny stage 1 with downsample
        (2, (28, 28), 192, 2, 6, 7, True),  # Swin-Tiny stage 2
        (1, (14, 14), 384, 6, 12, 7, True),  # Swin-Tiny stage 3
        (1, (7, 7), 768, 2, 24, 7, False),  # Swin-Tiny stage 4 (no downsample)
        (1, (32, 32), 128, 4, 4, 8, True),  # Custom configuration
    ],
)
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
    # setattr(ref_layer, "window_size", [window_size, window_size])

    # Create ttnn model
    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_basic_layer_preprocessor(device),
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_basic_layer_memory_config(device):
    """Test different memory configurations"""
    torch.manual_seed(0)

    input_resolution = (56, 56)
    dim = 96
    depth = 2
    num_heads = 3
    window_size = 7
    batch_size = 1
    H, W = input_resolution

    ref_layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        downsample=None,
    )
    ref_layer.eval()
    setattr(ref_layer, "window_size", [window_size, window_size])

    input_tensor = torch.randn(batch_size, H * W, dim)
    ref_output = ref_layer(input_tensor)

    # Test with L1 memory config
    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_basic_layer_preprocessor(device),
        device=device,
    )

    tt_layer_l1 = TTBasicLayer(
        device=device,
        parameters=params,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output_l1 = tt_layer_l1(tt_input)
    tt_torch_output_l1 = tt2torch_tensor(tt_output_l1)

    does_pass_l1, pcc_message_l1 = comp_pcc(ref_output, tt_torch_output_l1, 0.98)
    logger.info(f"L1 Memory Config: {pcc_message_l1}")
    assert does_pass_l1, f"L1 memory config failed: {pcc_message_l1}"


def test_basic_layer_multiple_iterations(device):
    """Test multiple forward passes to check for memory leaks"""
    torch.manual_seed(0)

    input_resolution = (28, 28)
    dim = 192
    depth = 2
    num_heads = 6
    window_size = 7
    batch_size = 1
    H, W = input_resolution

    ref_layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        downsample=PatchMerging,
    )
    ref_layer.eval()
    setattr(ref_layer, "window_size", [window_size, window_size])

    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_basic_layer_preprocessor(device),
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
        downsample=TTPatchMerging,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiple forward passes to check for memory leaks
    for i in range(3):
        input_tensor = torch.randn(batch_size, H * W, dim)
        ref_output = ref_layer(input_tensor)

        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_output = tt_layer(tt_input)
        tt_torch_output = tt2torch_tensor(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.98)
        logger.info(f"Iteration {i+1}: {pcc_message}")
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"

        # Clean up tensors to prevent memory leaks
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

        logger.info(f"Iteration {i+1} completed successfully")
