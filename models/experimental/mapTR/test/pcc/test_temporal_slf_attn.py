# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.mapTR.reference.pytorch_temporal_self_attention import TemporalSelfAttention
from models.experimental.mapTR.tt.temporal_sel_attention import TtTemporalSelfAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


def create_temporal_self_attention_preprocessor(device, weight_dtype=ttnn.bfloat16):
    """Custom preprocessor for TemporalSelfAttention parameters."""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if isinstance(torch_model, TemporalSelfAttention):
            # Preprocess sampling_offsets layer
            parameters["sampling_offsets"] = {
                "weight": preprocess_linear_weight(torch_model.sampling_offsets.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.sampling_offsets.bias, dtype=weight_dtype),
            }

            # Preprocess attention_weights layer
            parameters["attention_weights"] = {
                "weight": preprocess_linear_weight(torch_model.attention_weights.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.attention_weights.bias, dtype=weight_dtype),
            }

            # Preprocess value_proj layer
            parameters["value_proj"] = {
                "weight": preprocess_linear_weight(torch_model.value_proj.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.value_proj.bias, dtype=weight_dtype),
            }

            # Preprocess output_proj layer
            parameters["output_proj"] = {
                "weight": preprocess_linear_weight(torch_model.output_proj.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.output_proj.bias, dtype=weight_dtype),
            }

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "embed_dims, num_heads, num_levels, num_points, batch_size, num_query",
    [
        (256, 8, 1, 4, 1, 1000),  # Single level for BEV self-attention
        # (256, 8, 1, 4, 2, 500),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_temporal_self_attention_fp16(
    device, embed_dims, num_heads, num_levels, num_points, batch_size, num_query, input_dtype, weight_dtype
):
    """Test TemporalSelfAttention TT implementation against PyTorch reference with FP16."""
    torch.manual_seed(42)

    # Create reference PyTorch model and convert to FP16
    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_bev_queue=2,
        batch_first=True,
    ).eval()

    # Convert to FP16
    ref_model = ref_model.to_fp16()

    # Generate test inputs in FP16
    # For BEV temporal self-attention, spatial_shapes must sum to num_query
    query = torch.randn(batch_size, num_query, embed_dims).half()
    spatial_shapes = torch.tensor([[25, 40]])  # 25*40 = 1000 = num_query
    reference_points = torch.rand(batch_size, num_query, num_levels, 2).half()

    # Get reference output from PyTorch model
    with torch.no_grad():
        ref_output = ref_model(query=query, reference_points=reference_points, spatial_shapes=spatial_shapes)

    # Preprocess parameters for TT device
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_temporal_self_attention_preprocessor(device, weight_dtype),
        device=device,
    )

    # Create TT model
    tt_model = TtTemporalSelfAttention(
        device=device,
        params=parameters,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_bev_queue=2,
        batch_first=True,
    )

    # Convert inputs to TT tensors
    tt_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.int32)
    tt_reference_points = ttnn.from_torch(reference_points, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)

    # Run TT model
    with torch.no_grad():
        tt_output = tt_model(query=tt_query, reference_points=tt_reference_points, spatial_shapes=tt_spatial_shapes)

    # Convert TT output back to torch
    tt_torch_output = ttnn.to_torch(tt_output)

    # Check accuracy with PCC (slightly lower threshold for FP16)
    does_pass, pcc_message = assert_with_pcc(ref_output, tt_torch_output, 0.98)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("TemporalSelfAttention test passed!")
    else:
        logger.warning(f"TemporalSelfAttention test failed: {pcc_message}")

    assert does_pass, f"PCC check failed: {pcc_message}"
