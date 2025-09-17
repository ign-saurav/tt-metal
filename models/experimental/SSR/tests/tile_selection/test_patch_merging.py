# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.SSR.reference.SSR.model.net_blocks import PatchMerging
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.SSR.tt.tile_selection import TTPatchMerging


def create_patch_merging_preprocessor(device, dim, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Create conv kernels for patch merging (same as in forward pass)
        kernel_top_left = torch.zeros(dim, 1, 2, 2, dtype=torch.bfloat16)
        kernel_top_left[:, 0, 0, 0] = 1.0

        kernel_bottom_left = torch.zeros(dim, 1, 2, 2, dtype=torch.bfloat16)
        kernel_bottom_left[:, 0, 1, 0] = 1.0

        kernel_top_right = torch.zeros(dim, 1, 2, 2, dtype=torch.bfloat16)
        kernel_top_right[:, 0, 0, 1] = 1.0

        kernel_bottom_right = torch.zeros(dim, 1, 2, 2, dtype=torch.bfloat16)
        kernel_bottom_right[:, 0, 1, 1] = 1.0

        # Convert to TTNN tensors
        params["conv_kernels"] = {
            "top_left": ttnn.from_torch(kernel_top_left, device=device, dtype=ttnn.bfloat16),
            "bottom_left": ttnn.from_torch(kernel_bottom_left, device=device, dtype=ttnn.bfloat16),
            "top_right": ttnn.from_torch(kernel_top_right, device=device, dtype=ttnn.bfloat16),
            "bottom_right": ttnn.from_torch(kernel_bottom_right, device=device, dtype=ttnn.bfloat16),
        }

        # Linear reduction layer
        params["reduction"] = {
            "weight": ttnn.from_torch(
                torch_model.reduction.weight.transpose(0, 1),  # Transpose for ttnn.linear
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        }

        # Layer normalization
        params["norm"] = {
            "weight": ttnn.from_torch(
                torch_model.norm.weight,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                torch_model.norm.bias,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, input_resolution, dim",
    (
        (3, (128, 128), 96),
        (3, (64, 64), 192),
        (3, (32, 32), 384),
        (3, (16, 16), 768),
        (3, (8, 8), 1536),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_patch_merging(device, batch_size, input_resolution, dim, input_dtype, weight_dtype):
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
        custom_preprocessor=create_patch_merging_preprocessor(device, dim, weight_dtype),
        device=device,
    )

    tt_layer = TTPatchMerging(
        device=device,
        parameters=params,
        input_resolution=input_resolution,
        dim=dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=input_dtype,
    )

    # Convert input to ttnn
    tt_input = ttnn.from_torch(
        input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # ttnn forward pass
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)
    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("PatchMerging Passed!")
    else:
        logger.warning("PatchMerging Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
