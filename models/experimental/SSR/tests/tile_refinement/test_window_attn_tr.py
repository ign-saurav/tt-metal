# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import WindowAttention
from models.experimental.SSR.tt.tile_refinement import TTWindowAttentionTR
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from tests.ttnn.utils_for_testing import check_with_pcc


def create_window_attention_preprocessor(device, window_size=None, rpi=None):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # QKV linear layer
        params["qkv"] = {
            "weight": preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if torch_model.qkv.bias is not None
            else None,
        }

        # Projection layer
        params["proj"] = {
            "weight": preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if torch_model.proj.bias is not None
            else None,
        }

        # Relative position bias table
        relative_position_bias = torch_model.relative_position_bias_table[rpi.view(-1)].view(
            window_size[0] * window_size[1], window_size[0] * window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        params["relative_position_bias"] = ttnn.from_torch(
            relative_position_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_windows, window_size, dim, num_heads",
    [
        (1, 16, (16, 16), 180, 6),  # SSR config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_window_attention(device, batch_size, num_windows, window_size, dim, num_heads):
    torch.manual_seed(0)

    # Create reference model
    ref_model = WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    ref_model.eval()

    # Create input tensors
    window_area = window_size[0] * window_size[1]
    input_tensor = torch.randn(batch_size * num_windows, window_area, dim)

    # Create relative position index
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    rpi = relative_coords.sum(-1)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, rpi=rpi, mask=None)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_window_attention_preprocessor(device, window_size, rpi),
        device=device,
        run_model=lambda model: model(input_tensor, rpi=rpi, mask=None),
    )

    tt_model = TTWindowAttentionTR(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(
        input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    # TTNN forward pass
    tt_output = tt_model(tt_input, rpi=tt_rpi, mask=None)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.97)

    if does_pass:
        logger.info("Window Attention Passed!")
    else:
        logger.warning("Window Attention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
