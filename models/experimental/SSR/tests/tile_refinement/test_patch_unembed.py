# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

from models.utility_functions import torch_random

from models.experimental.SSR.reference.SSR.model.tile_refinement import PatchUnEmbed
from models.experimental.SSR.tt.tile_refinement import TTPatchUnEmbed


@pytest.mark.parametrize(
    "batch_size, img_size, patch_size, in_chans, embed_dim",
    [
        (1, 64, 4, 3, 180),  # TR blk test
        (1, 64, 2, 3, 180),  # HAT blk test
    ],
)
def test_tt_patch_unembed(device, batch_size, img_size, patch_size, in_chans, embed_dim):
    """Test TTPatchUnEmbed against PyTorch reference implementation"""
    torch.manual_seed(0)

    # Calculate patch dimensions
    patches_resolution = [img_size // patch_size, img_size // patch_size]
    num_patches = patches_resolution[0] * patches_resolution[1]

    # Create reference PyTorch model
    torch_model = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    # Create TTNN model
    tt_model = TTPatchUnEmbed(
        mesh_device=device, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
    )

    # Generate random input tensor (batch_size, num_patches, embed_dim)
    torch_input = torch_random((batch_size, num_patches, embed_dim), -1, 1, dtype=torch.float32)

    # Run PyTorch reference
    torch_output = torch_model(torch_input, patches_resolution)

    # Convert input to TTNN format
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Run TTNN model
    ttnn_output = tt_model(ttnn_input, patches_resolution)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, 0.99)

    if does_pass:
        logger.info("TR PatchEmbed Passed!")
    else:
        logger.warning("TR PatchEmbed Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
