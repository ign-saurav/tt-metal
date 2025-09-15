# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import OCAB
from models.experimental.SSR.tt.tile_refinement import TTOCAB


def create_ocab_preprocessor(device, tile_size=32):
    """Create custom preprocessor for OCAB parameters"""

    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, OCAB):
            # Layer norm parameters
            dim = model.norm1.weight.size(0)
            padded_dim = ((dim + tile_size - 1) // tile_size) * tile_size

            norm1_weight_padded = torch.nn.functional.pad(model.norm1.weight, (0, padded_dim - dim))
            norm1_bias_padded = torch.nn.functional.pad(model.norm1.bias, (0, padded_dim - dim))

            norm1_weight = norm1_weight_padded.view(1, 1, padded_dim // tile_size, tile_size)
            norm1_bias = norm1_bias_padded.view(1, 1, padded_dim // tile_size, tile_size)

            parameters["norm1"] = {
                "weight": ttnn.from_torch(
                    norm1_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                ),
                "bias": ttnn.from_torch(norm1_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            }

            # QKV linear layer with padded heads
            qkv_weight = model.qkv.weight.T  # [embed_dim, 3*embed_dim]
            num_heads = model.num_heads
            head_size = qkv_weight.shape[1] // (3 * num_heads)
            padded_head_size = ((head_size + tile_size - 1) // tile_size) * tile_size

            if padded_head_size != head_size:
                # Pad each head separately
                qkv_chunks = torch.split(qkv_weight, head_size, dim=1)
                qkv_weight_padded = torch.cat(
                    [torch.nn.functional.pad(chunk, (0, padded_head_size - head_size)) for chunk in qkv_chunks], dim=1
                )

                if model.qkv.bias is not None:
                    qkv_bias_chunks = torch.split(model.qkv.bias, head_size, dim=0)
                    qkv_bias_padded = torch.cat(
                        [
                            torch.nn.functional.pad(chunk, (0, padded_head_size - head_size))
                            for chunk in qkv_bias_chunks
                        ],
                        dim=0,
                    )
                else:
                    qkv_bias_padded = None
            else:
                qkv_weight_padded = qkv_weight
                qkv_bias_padded = model.qkv.bias

            parameters["qkv"] = {
                "weight": ttnn.from_torch(
                    qkv_weight_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                ),
                "bias": ttnn.from_torch(qkv_bias_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if qkv_bias_padded is not None
                else None,
            }

            # Relative position bias table
            parameters["relative_position_bias_table"] = ttnn.from_torch(
                model.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Output projection
            proj_weight = model.proj.weight.T
            parameters["proj"] = {
                "weight": ttnn.from_torch(proj_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                "bias": ttnn.from_torch(model.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            }

            # Layer norm 2
            norm2_weight_padded = torch.nn.functional.pad(model.norm2.weight, (0, padded_dim - dim))
            norm2_bias_padded = torch.nn.functional.pad(model.norm2.bias, (0, padded_dim - dim))

            norm2_weight = norm2_weight_padded.view(1, 1, padded_dim // tile_size, tile_size)
            norm2_bias = norm2_bias_padded.view(1, 1, padded_dim // tile_size, tile_size)

            parameters["norm2"] = {
                "weight": ttnn.from_torch(
                    norm2_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                ),
                "bias": ttnn.from_torch(norm2_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            }

            # MLP parameters
            parameters["mlp"] = {
                "fc1": {
                    "weight": ttnn.from_torch(
                        model.mlp.fc1.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        model.mlp.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                },
                "fc2": {
                    "weight": ttnn.from_torch(
                        model.mlp.fc2.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        model.mlp.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                },
            }
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "dim, input_resolution, window_size, overlap_ratio, num_heads, input_shape",
    ((180, (64, 64), 16, 0.5, 6, (1, 4096, 180)),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ocab(device, dim, input_resolution, window_size, overlap_ratio, num_heads, input_shape):
    x = torch.randn(input_shape)

    # Create reference OCAB layer
    ref_layer = OCAB(
        dim=dim,
        input_resolution=input_resolution,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        num_heads=num_heads,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
    )

    h, w = input_resolution
    x_size = (h, w)
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    ref_output = ref_layer(x, x_size, rpi)

    ttnn.synchronize_device(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer, custom_preprocessor=create_ocab_preprocessor(device), device=device
    )

    tt_layer = TTOCAB(
        device=device,
        dim=dim,
        input_resolution=input_resolution,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        num_heads=num_heads,
        parameters=parameters,
    )

    tt_input = ttnn.from_torch(
        x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = tt_layer.forward(tt_input, x_size, tt_rpi)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"pcc: {pcc_message}")

    if does_pass:
        logger.info("OCAB Layer Passed!")
    else:
        logger.warning("OCAB Layer Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
