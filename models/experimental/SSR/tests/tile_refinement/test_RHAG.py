# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.tt.tile_refinement import TTRHAG
from models.experimental.SSR.reference.SSR.model.tile_refinement import RHAG
from models.experimental.SSR.tests.tile_refinement.test_HAB import create_relative_position_index
from models.experimental.SSR.tests.tile_refinement.test_atten_blocks import create_atten_blocks_preprocessor
from models.experimental.SSR.tests.tile_refinement.test_patch_embed_tile_refinement import (
    create_patch_embed_preprocessor_conv,
)

from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


def create_rhag_preprocessor(device, depth, window_size, rpi_sa):
    """Preprocessor for RHAG that handles all sub-components by importing existing preprocessors"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Import and use AttenBlocks preprocessor
        atten_blocks_preprocessor = create_atten_blocks_preprocessor(device, depth, window_size, rpi_sa)
        params["residual_group"] = atten_blocks_preprocessor(
            torch_model.residual_group, "residual_group", ttnn_module_args
        )

        # Preprocess conv layer parameters (if 1conv)
        if hasattr(torch_model, "conv") and hasattr(torch_model.conv, "weight"):
            conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
            params["conv"] = {
                "weight": ttnn.prepare_conv_weights(
                    weight_tensor=ttnn.from_torch(torch_model.conv.weight, dtype=ttnn.bfloat16),
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.TILE_LAYOUT,
                    weights_format="OIHW",
                    in_channels=torch_model.conv.in_channels,
                    out_channels=torch_model.conv.out_channels,
                    batch_size=1,
                    input_height=torch_model.input_resolution[0],
                    input_width=torch_model.input_resolution[1],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    has_bias=True,
                    groups=1,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_config,
                ),
                "bias": ttnn.prepare_conv_bias(
                    bias_tensor=ttnn.from_torch(torch_model.conv.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.TILE_LAYOUT,
                    in_channels=torch_model.conv.in_channels,
                    out_channels=torch_model.conv.out_channels,
                    batch_size=1,
                    input_height=torch_model.input_resolution[0],
                    input_width=torch_model.input_resolution[1],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_config,
                ),
            }

        if hasattr(torch_model, "patch_embed"):
            patch_embed_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.patch_embed,
                custom_preprocessor=create_patch_embed_preprocessor_conv(device),
                device=device,
            )
            params["patch_embed"] = patch_embed_params

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio, resi_connection",
    [
        (1, 64, 64, 180, 6, 16, 6, 0.5, 2, "1conv"),  # SSR config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_rhag(
    device, batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio, resi_connection
):
    torch.manual_seed(0)

    # Create reference model
    ref_model = RHAG(
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
        img_size=max(height, width),
        patch_size=4,
        resi_connection=resi_connection,
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
    # parameters = preprocess_model_parameters(
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_rhag_preprocessor(device, depth, window_size, rpi_sa),
        device=device,
    )

    tt_model = TTRHAG(
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
        img_size=max(height, width),
        patch_size=4,
        resi_connection=resi_connection,
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
    # tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.85)

    if does_pass:
        logger.info("RHAG Passed!")
    else:
        logger.warning("RHAG Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
