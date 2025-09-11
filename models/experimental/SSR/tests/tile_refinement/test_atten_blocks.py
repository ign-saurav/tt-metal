# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import AttenBlocks
from models.experimental.SSR.tt.tile_refinement import TTAttenBlocks
from models.experimental.SSR.tests.tile_refinement.test_HAB import (
    create_hab_preprocessor,
    create_relative_position_index,
)
from models.experimental.SSR.tests.tile_refinement.test_OCAB import create_ocab_preprocessor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_atten_blocks_preprocessor(device, depth, window_size, rpi_sa):
    """Preprocessor for AttenBlocks that handles multiple HAB blocks and one OCAB block"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Preprocess parameters for each HAB block
        params["blocks"] = {}
        hab_preprocessor = create_hab_preprocessor(device, window_size, rpi_sa)
        for i in range(depth):
            params["blocks"][i] = hab_preprocessor(torch_model.blocks[i], f"blocks_{i}", ttnn_module_args)

        # Preprocess parameters for OCAB
        ocab_preprocessor = create_ocab_preprocessor(device)
        params["overlap_attn"] = ocab_preprocessor(torch_model.overlap_attn, "overlap_attn")

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio",
    [
        (1, 64, 64, 180, 6, 16, 6, 0.5, 2),  # SSR config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_atten_blocks(device, batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio):
    torch.manual_seed(0)

    # Create reference model
    ref_model = AttenBlocks(
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    )
    ref_model.eval()

    # Create input tensors
    input_tensor = torch.randn(batch_size, height * width, dim)
    x_size = (height, width)

    # Create relative position indices
    rpi_sa = create_relative_position_index((window_size, window_size))

    # Create attention mask for shifted windows (simplified for testing)
    attn_mask = None

    # Create RPI for OCAB
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi_oca = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    # Create params dictionary
    params = {"rpi_sa": rpi_sa, "attn_mask": attn_mask, "rpi_oca": rpi_oca}

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, params)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_atten_blocks_preprocessor(device, depth, window_size, rpi_sa),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, params),
    )

    tt_model = TTAttenBlocks(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

    tt_params = {"rpi_sa": tt_rpi_sa, "attn_mask": None, "rpi_oca": tt_rpi_oca}

    # TTNN forward pass
    tt_output = tt_model(tt_input, x_size, tt_params)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.90)

    if does_pass:
        logger.info("AttenBlocks Passed!")
    else:
        logger.warning("AttenBlocks Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
