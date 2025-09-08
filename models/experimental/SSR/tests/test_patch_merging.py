# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.SSR.reference.SSR.model.net_blocks import PatchMerging
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor, comp_pcc
from models.experimental.SSR.tt.patch_merging import TTPatchMerging


def create_patch_merging_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Linear reduction layer
        params["reduction"] = {
            "weight": ttnn.from_torch(
                torch_model.reduction.weight.transpose(0, 1),  # Transpose for ttnn.linear
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        }

        # Layer normalization
        params["norm"] = {
            "weight": ttnn.from_torch(
                torch_model.norm.weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                torch_model.norm.bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, input_resolution, dim",
    [
        (1, (56, 56), 96),  # Swin-Tiny stage 1 -> 2
        (2, (28, 28), 192),  # Swin-Tiny stage 2 -> 3
        (1, (14, 14), 384),  # Swin-Tiny stage 3 -> 4
        (1, (112, 112), 128),  # Swin-Small stage 1 -> 2
        (2, (56, 56), 256),  # Swin-Small stage 2 -> 3
        (1, (32, 32), 256),  # Custom resolution
    ],
)
def test_patch_merging(device, batch_size, input_resolution, dim):
    torch.manual_seed(0)

    H, W = input_resolution

    # Create reference model
    ref_layer = PatchMerging(input_resolution=input_resolution, dim=dim)
    ref_layer.eval()

    # Create input tensor [B, H*W, C]
    input_tensor = torch.randn(batch_size, H * W, dim)

    # Reference forward pass
    ref_output = ref_layer(input_tensor)

    # Create ttnn model
    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_patch_merging_preprocessor(device),
        device=device,
    )

    tt_layer = TTPatchMerging(
        device=device,
        parameters=params,
        input_resolution=input_resolution,
        dim=dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert input to ttnn
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # ttnn forward pass
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"Input resolution: {input_resolution}, Dim: {dim}, Batch: {batch_size}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("PatchMerging Passed!")
    else:
        logger.warning("PatchMerging Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Verify output shapes match
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"

    # Verify expected output shape
    expected_shape = (batch_size, (H // 2) * (W // 2), 2 * dim)
    assert ref_output.shape == expected_shape, f"Unexpected output shape: {ref_output.shape}, expected {expected_shape}"
    ttnn.close_device(device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patch_merging_memory_config(device):
    """Test different memory configurations"""
    torch.manual_seed(0)

    input_resolution = (56, 56)
    dim = 96
    batch_size = 1
    H, W = input_resolution

    ref_layer = PatchMerging(input_resolution=input_resolution, dim=dim)
    ref_layer.eval()

    input_tensor = torch.randn(batch_size, H * W, dim)
    ref_output = ref_layer(input_tensor)

    # Test with L1 memory config
    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_patch_merging_preprocessor(device),
        device=device,
    )

    tt_layer_l1 = TTPatchMerging(
        device=device,
        parameters=params,
        input_resolution=input_resolution,
        dim=dim,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output_l1 = tt_layer_l1(tt_input)
    tt_torch_output_l1 = tt2torch_tensor(tt_output_l1)

    does_pass_l1, pcc_message_l1 = comp_pcc(ref_output, tt_torch_output_l1, 0.99)
    logger.info(f"L1 Memory Config: {pcc_message_l1}")
    assert does_pass_l1, f"L1 memory config failed: {pcc_message_l1}"
    ttnn.close_device(device)


def test_patch_merging_error_handling(device):
    """Test error handling for invalid input dimensions"""
    input_resolution = (56, 56)
    dim = 96

    ref_layer = PatchMerging(input_resolution=input_resolution, dim=dim)
    ref_layer.eval()

    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_patch_merging_preprocessor(device),
        device=device,
    )

    tt_layer = TTPatchMerging(
        device=device,
        parameters=params,
        input_resolution=input_resolution,
        dim=dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Test with wrong sequence length
    wrong_input = torch.randn(1, 100, dim)  # Wrong sequence length
    tt_wrong_input = ttnn.from_torch(wrong_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    with pytest.raises(AssertionError, match="input feature has wrong size"):
        tt_layer(tt_wrong_input)
    ttnn.close_device(device)


def test_patch_merging_multiple_iterations(device):
    """Test multiple forward passes to check for memory leaks"""
    torch.manual_seed(0)

    input_resolution = (28, 28)
    dim = 192
    batch_size = 1
    H, W = input_resolution

    ref_layer = PatchMerging(input_resolution=input_resolution, dim=dim)
    ref_layer.eval()

    params = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_patch_merging_preprocessor(device),
        device=device,
    )

    tt_layer = TTPatchMerging(
        device=device,
        parameters=params,
        input_resolution=input_resolution,
        dim=dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiple forward passes to check for memory leaks
    for i in range(3):
        input_tensor = torch.randn(batch_size, H * W, dim)
        ref_output = ref_layer(input_tensor)

        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_output = tt_layer(tt_input)
        tt_torch_output = tt2torch_tensor(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)
        logger.info(f"Iteration {i+1}: {pcc_message}")
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"

        # Clean up
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

    ttnn.close_device(device)
