# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.mapTR.reference.pytorch_spatial_cross_attention import SpatialCrossAttention
from models.experimental.mapTR.reference.pytorch_spatial_cross_attention import MSDeformableAttention3D
from models.experimental.mapTR.tt.spatial_cross_attention import TtSpatialCrossAttention
from models.experimental.mapTR.tt.spatial_cross_attention import TtMSDeformableAttention3D
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


def create_spatial_cross_attention_preprocessor(device, weight_dtype=ttnn.bfloat16):
    """Custom preprocessor for SpatialCrossAttention parameters."""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if isinstance(torch_model, SpatialCrossAttention):
            # Preprocess output_proj layer
            parameters["output_proj"] = {
                "weight": preprocess_linear_weight(torch_model.output_proj.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.output_proj.bias, dtype=weight_dtype),
            }

        return parameters

    return custom_preprocessor


def create_ms_deformable_attention_preprocessor(device, weight_dtype=ttnn.bfloat16):
    """Custom preprocessor for MSDeformableAttention3D parameters."""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if isinstance(torch_model, MSDeformableAttention3D):
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

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "embed_dims, num_cams, num_levels, num_points, batch_size, num_query, num_key",
    [
        # MapTR typical config:
        # - num_points = 4 (sampling points per level per head)
        # - num_Z_anchors = 4 (height anchors, defined below)
        # - num_key = sum of spatial_shapes: 37*37 + 19*19 + 10*10 + 5*5 = 1855
        # - num_query = BEV queries (e.g., 50x50=2500 or smaller for testing)
        # (256, 6, 4, 4, 1, 2500, 1855), #OOM
        (256, 6, 4, 4, 1, 500, 1855),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_spatial_cross_attention(
    device, embed_dims, num_cams, num_levels, num_points, batch_size, num_query, num_key, input_dtype, weight_dtype
):
    """Test SpatialCrossAttention TTNN implementation against PyTorch reference."""
    torch.manual_seed(42)

    # MapTR uses 4 height/depth anchors for BEV to camera projection
    num_Z_anchors = 4

    # Create deformable attention module
    # num_points here is total sampling points = num_points_per_anchor * num_Z_anchors
    deformable_attention = MSDeformableAttention3D(
        embed_dims=embed_dims,
        num_heads=8,
        num_levels=num_levels,
        num_points=num_points * num_Z_anchors,  # 4 * 4 = 16 total sampling points
        batch_first=True,
    ).eval()

    # Create reference PyTorch model
    ref_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        batch_first=True,
        deformable_attention=deformable_attention,
    ).eval()

    # Generate test inputs
    query = torch.randn(batch_size, num_query, embed_dims)
    key = torch.randn(num_cams, num_key, batch_size, embed_dims)
    value = torch.randn(num_cams, num_key, batch_size, embed_dims)
    # Feature pyramid spatial shapes (FPN levels)
    spatial_shapes = torch.tensor([[37, 37], [19, 19], [10, 10], [5, 5]])
    # reference_points_cam shape: (num_cams, bs, num_query, num_Z_anchors, 2)
    # Each BEV query projects to num_Z_anchors points at different heights in each camera
    reference_points_cam = torch.rand(num_cams, batch_size, num_query, num_Z_anchors, 2)
    bev_mask = torch.ones(num_cams, batch_size, num_query, num_Z_anchors).bool()

    # Get reference output from PyTorch model
    with torch.no_grad():
        ref_output = ref_model(
            query=query,
            key=key,
            value=value,
            reference_points_cam=reference_points_cam,
            spatial_shapes=spatial_shapes,
            bev_mask=bev_mask,
        )

    # Preprocess parameters for TTNN device
    spatial_params = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_spatial_cross_attention_preprocessor(device, weight_dtype),
        device=device,
    )

    deformable_params = preprocess_model_parameters(
        initialize_model=lambda: deformable_attention,
        custom_preprocessor=create_ms_deformable_attention_preprocessor(device, weight_dtype),
        device=device,
    )

    # Create TTNN deformable attention
    tt_deformable_attention = TtMSDeformableAttention3D(
        device=device,
        params=deformable_params,
        embed_dims=embed_dims,
        num_heads=8,
        num_levels=num_levels,
        num_points=num_points * num_Z_anchors,  # Must match PyTorch model
        batch_first=True,
    )

    # Create TTNN model
    tt_model = TtSpatialCrossAttention(
        device=device,
        params=spatial_params,
        embed_dims=embed_dims,
        num_cams=num_cams,
        batch_first=True,
        deformable_attention=tt_deformable_attention,
    )

    # Convert inputs to TTNN tensors
    tt_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_key = ttnn.from_torch(key, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_value = ttnn.from_torch(value, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.int32)
    tt_reference_points_cam = ttnn.from_torch(
        reference_points_cam, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype
    )
    # Keep bev_mask as torch tensor - TT impl converts it back to torch for indexing anyway

    # Run TTNN model
    with torch.no_grad():
        tt_output = tt_model(
            query=tt_query,
            key=tt_key,
            value=tt_value,
            reference_points_cam=tt_reference_points_cam,
            spatial_shapes=tt_spatial_shapes,
            bev_mask=bev_mask,  # Pass torch tensor directly
        )

    # Convert TTNN output back to torch
    tt_torch_output = ttnn.to_torch(tt_output)

    # Check accuracy with PCC
    does_pass, pcc_message = assert_with_pcc(ref_output, tt_torch_output, 0.98)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("SpatialCrossAttention test passed!")
    else:
        logger.warning(f"SpatialCrossAttention test failed: {pcc_message}")

    assert does_pass, f"PCC check failed: {pcc_message}"
