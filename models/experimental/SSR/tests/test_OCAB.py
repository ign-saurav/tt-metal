import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor, comp_pcc
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import OCAB
from models.experimental.SSR.tt.OCAB import TTOCAB


def create_ocab_preprocessor(device):
    """Create custom preprocessor for OCAB parameters"""

    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, OCAB):
            # Layer norm parameters (keep existing padding logic)
            dim = model.norm1.weight.size(0)  # 180
            padded_dim = ((dim + 31) // 32) * 32  # Round up to nearest multiple of 32 = 192

            norm1_weight_padded = torch.nn.functional.pad(model.norm1.weight, (0, padded_dim - dim))
            norm1_bias_padded = torch.nn.functional.pad(model.norm1.bias, (0, padded_dim - dim))

            norm1_weight = norm1_weight_padded.view(1, 1, padded_dim // 32, 32)
            norm1_bias = norm1_bias_padded.view(1, 1, padded_dim // 32, 32)

            parameters["norm1"] = {
                "weight": ttnn.from_torch(
                    norm1_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                ),
                "bias": ttnn.from_torch(norm1_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            }

            # QKV linear layer - ENSURE TILE LAYOUT
            qkv_weight = model.qkv.weight.T  # Transpose for linear operation
            parameters["qkv"] = {
                "weight": ttnn.from_torch(qkv_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                "bias": ttnn.from_torch(model.qkv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if model.qkv.bias is not None
                else None,
            }

            # Relative position bias table - ENSURE TILE LAYOUT
            parameters["relative_position_bias_table"] = ttnn.from_torch(
                model.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Output projection - ENSURE TILE LAYOUT
            proj_weight = model.proj.weight.T
            parameters["proj"] = {
                "weight": ttnn.from_torch(proj_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                "bias": ttnn.from_torch(model.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            }

            # Layer norm 2 (same padding logic as norm1)
            norm2_weight_padded = torch.nn.functional.pad(model.norm2.weight, (0, padded_dim - dim))
            norm2_bias_padded = torch.nn.functional.pad(model.norm2.bias, (0, padded_dim - dim))

            norm2_weight = norm2_weight_padded.view(1, 1, padded_dim // 32, 32)
            norm2_bias = norm2_bias_padded.view(1, 1, padded_dim // 32, 32)

            parameters["norm2"] = {
                "weight": ttnn.from_torch(
                    norm2_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                ),
                "bias": ttnn.from_torch(norm2_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            }

            # MLP parameters - ENSURE TILE LAYOUT
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
def test_ocab(dim, input_resolution, window_size, overlap_ratio, num_heads, input_shape):
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

    # Create dummy inputs for forward pass
    h, w = input_resolution
    x_size = (h, w)
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    ref_output = ref_layer(x, x_size, rpi)

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    ttnn.synchronize_device(device)

    try:
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
            x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = tt_layer.forward(tt_input, x_size, tt_rpi)
        tt_torch_output = tt2torch_tensor(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

        logger.info(pcc_message)

        if does_pass:
            logger.info("OCAB Layer Passed!")
        else:
            logger.warning("OCAB Layer Failed!")

    finally:
        ttnn.close_device(device)

    assert does_pass
