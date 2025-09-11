# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import HAB
from models.experimental.SSR.tt.tile_refinement import TTHAB
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.SSR.tests.tile_refinement.test_window_attn_tr import create_window_attention_preprocessor
from models.experimental.SSR.tests.tile_refinement.test_CAB import create_cab_preprocessor
from models.experimental.SSR.tests.common.test_mlp import create_mlp_preprocessor


def create_hab_preprocessor(device, window_size, rpi):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Norm layers
        params["norm1"] = {
            "weight": preprocess_linear_weight(torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }
        params["norm2"] = {
            "weight": preprocess_linear_weight(torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }
        relative_position_bias = torch_model.attn.relative_position_bias_table[rpi.view(-1)].view(
            window_size * window_size, window_size * window_size, -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        params["relative_position_bias"] = ttnn.from_torch(
            relative_position_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        # Window attention parameters
        window_attention_preprocessor = create_window_attention_preprocessor(device, (window_size, window_size), rpi)
        params["attn"] = window_attention_preprocessor(torch_model.attn, "attn", ttnn_module_args)

        # Conv block parameters
        cab_preprocessor = create_cab_preprocessor(device)
        params["conv_block"] = cab_preprocessor(torch_model.conv_block, "conv_block", ttnn_module_args)

        # MLP parameters
        mlp_preprocessor = create_mlp_preprocessor(device)
        params["mlp"] = mlp_preprocessor(torch_model.mlp, "mlp", ttnn_module_args)

        # Conv scale
        params["conv_scale"] = torch_model.conv_scale

        return params

    return custom_preprocessor


def create_relative_position_index(window_size):
    """Create relative position index for window attention"""
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    return relative_coords.sum(-1)


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio",
    [
        # SSR configurations
        (1, 64, 64, 180, 6, 16, 8, 2),  # With shift
        (1, 64, 64, 180, 6, 16, 0, 2),  # Without shift
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_hab_block(device, batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio):
    torch.manual_seed(0)

    # Create reference model
    ref_model = HAB(
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
    )
    ref_model.eval()

    # Create input tensors
    input_tensor = torch.randn(batch_size, height * width, dim)
    x_size = (height, width)

    # Create relative position index
    rpi_sa = create_relative_position_index((window_size, window_size))

    # Create attention mask for shifted windows
    if shift_size > 0:
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Create attention mask
        mask_windows = img_mask.view(1, height // window_size, window_size, width // window_size, window_size, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, float("0.0"))
    else:
        attn_mask = None

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, rpi_sa, attn_mask)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_hab_preprocessor(device, window_size, rpi_sa),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, rpi_sa, attn_mask),
    )

    memory_config = ttnn.L1_MEMORY_CONFIG

    tt_model = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        memory_config=memory_config,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(
        input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memory_config
    )

    tt_rpi = ttnn.from_torch(
        rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=memory_config
    )

    tt_attn_mask = None
    if attn_mask is not None:
        tt_attn_mask = ttnn.from_torch(
            attn_mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memory_config
        )

    # TTNN forward pass
    tt_output = tt_model(tt_input, x_size, tt_rpi, tt_attn_mask)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.95)

    logger.info(f"pcc: {pcc_message}")
    if does_pass:
        logger.info("HAB Block Passed!")
    else:
        logger.warning("HAB Block Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
