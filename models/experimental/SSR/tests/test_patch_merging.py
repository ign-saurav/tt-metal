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
    (
        (3, (128, 128), 96),  # Custom resolution
        (3, (64, 64), 192),  # Custom resolution
        (3, (32, 32), 384),  # Custom resolution
        (3, (16, 16), 768),  # Custom resolution
        (3, (8, 8), 1536),  # Custom resolution
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
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
