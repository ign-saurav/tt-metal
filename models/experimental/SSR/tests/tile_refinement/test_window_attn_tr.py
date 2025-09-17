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


def create_window_attention_preprocessor(device, window_size=None, rpi=None, tile_size=32, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # QKV linear layer
        num_heads = torch_model.num_heads
        head_size = torch_model.qkv.weight.shape[0] // (3 * num_heads)
        # nearest multiple of tile_size
        padded_head_size = ((head_size + tile_size - 1) // tile_size) * tile_size
        qkv_weight = torch_model.qkv.weight
        qkv_bias = torch_model.qkv.bias

        if padded_head_size != head_size:
            # Weight: [3*num_heads*head_size, in_features]
            qkv_weight = qkv_weight.view(3 * num_heads, head_size, -1)
            qkv_weight = torch.nn.functional.pad(qkv_weight, (0, 0, 0, padded_head_size - head_size), "constant", 0)
            qkv_weight = qkv_weight.reshape(3 * num_heads * padded_head_size, -1)

            if qkv_bias is not None:
                # Bias: [3*num_heads, head_size]
                qkv_bias = qkv_bias.view(3 * num_heads, head_size)
                qkv_bias = torch.nn.functional.pad(qkv_bias, (0, padded_head_size - head_size), "constant", 0)
                qkv_bias = qkv_bias.reshape(3 * num_heads * padded_head_size)

        params["qkv"] = {
            "weight": preprocess_linear_weight(qkv_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(qkv_bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            if qkv_bias is not None
            else None,
        }

        # Projection layer
        params["proj"] = {
            "weight": preprocess_linear_weight(torch_model.proj.weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.proj.bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            if torch_model.proj.bias is not None
            else None,
        }

        # Relative position bias table
        relative_position_bias = torch_model.relative_position_bias_table[rpi.view(-1)].view(
            window_size[0] * window_size[1], window_size[0] * window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        params["relative_position_bias"] = ttnn.from_torch(
            relative_position_bias.unsqueeze(0), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
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
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_window_attention(device, batch_size, num_windows, window_size, dim, num_heads, input_dtype, weight_dtype):
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
        custom_preprocessor=create_window_attention_preprocessor(device, window_size, rpi, weight_dtype=weight_dtype),
        device=device,
        run_model=lambda model: model(input_tensor, rpi=rpi, mask=None),
    )

    memory_config = ttnn.L1_MEMORY_CONFIG

    tt_model = TTWindowAttentionTR(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        memory_config=memory_config,
        dtype=input_dtype,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(
        input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=memory_config
    )

    tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    # TTNN forward pass
    tt_output = tt_model(tt_input, rpi=tt_rpi, mask=None)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.97)

    logger.info(f"pcc: {pcc_message}")

    if does_pass:
        logger.info("Window Attention Passed!")
    else:
        logger.warning("Window Attention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
