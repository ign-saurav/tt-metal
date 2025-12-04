# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.bevformerv2.reference.custom_base_transformer_layer import (
    MyCustomBaseTransformerLayer,
)
from models.experimental.bevformerv2.tt.tt_custom_base_transformer_layer import TtCustomBaseTransformerLayer
from models.experimental.vadv2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.vadv2.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.vadv2.tt.tt_temporal_self_attention import TtTemporalSelfAttention
from models.experimental.vadv2.tt.tt_spatial_cross_attention import TtSpatialCrossAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


def custom_preprocessor(model, name):
    """Custom preprocessor for the transformer layer parameters."""
    parameters = {}

    if isinstance(model, MyCustomBaseTransformerLayer):
        # Process temporal self attention
        if hasattr(model, "attentions") and len(model.attentions) >= 1:
            if isinstance(model.attentions[0], TemporalSelfAttention):
                parameters["attentions"] = {}
                parameters["attentions"]["temporal_self_attention"] = {}
                tsa = model.attentions[0]
                parameters["attentions"]["temporal_self_attention"]["sampling_offsets"] = {}
                parameters["attentions"]["temporal_self_attention"]["sampling_offsets"][
                    "weight"
                ] = preprocess_linear_weight(tsa.sampling_offsets.weight, dtype=ttnn.bfloat16)
                parameters["attentions"]["temporal_self_attention"]["sampling_offsets"][
                    "bias"
                ] = preprocess_linear_bias(tsa.sampling_offsets.bias, dtype=ttnn.bfloat16)
                parameters["attentions"]["temporal_self_attention"]["attention_weights"] = {}
                parameters["attentions"]["temporal_self_attention"]["attention_weights"][
                    "weight"
                ] = preprocess_linear_weight(tsa.attention_weights.weight, dtype=ttnn.bfloat16)
                parameters["attentions"]["temporal_self_attention"]["attention_weights"][
                    "bias"
                ] = preprocess_linear_bias(tsa.attention_weights.bias, dtype=ttnn.bfloat16)
                parameters["attentions"]["temporal_self_attention"]["value_proj"] = {}
                parameters["attentions"]["temporal_self_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
                    tsa.value_proj.weight, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["temporal_self_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
                    tsa.value_proj.bias, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["temporal_self_attention"]["output_proj"] = {}
                parameters["attentions"]["temporal_self_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
                    tsa.output_proj.weight, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["temporal_self_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
                    tsa.output_proj.bias, dtype=ttnn.bfloat16
                )

            # Process spatial cross attention
            if len(model.attentions) >= 2 and isinstance(model.attentions[1], SpatialCrossAttention):
                if "attentions" not in parameters:
                    parameters["attentions"] = {}
                parameters["attentions"]["spatial_cross_attention"] = {}
                sca = model.attentions[1]
                parameters["attentions"]["spatial_cross_attention"]["sampling_offsets"] = {}
                parameters["attentions"]["spatial_cross_attention"]["sampling_offsets"][
                    "weight"
                ] = preprocess_linear_weight(sca.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16)
                parameters["attentions"]["spatial_cross_attention"]["sampling_offsets"][
                    "bias"
                ] = preprocess_linear_bias(sca.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16)
                parameters["attentions"]["spatial_cross_attention"]["attention_weights"] = {}
                parameters["attentions"]["spatial_cross_attention"]["attention_weights"][
                    "weight"
                ] = preprocess_linear_weight(sca.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16)
                parameters["attentions"]["spatial_cross_attention"]["attention_weights"][
                    "bias"
                ] = preprocess_linear_bias(sca.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16)
                parameters["attentions"]["spatial_cross_attention"]["value_proj"] = {}
                parameters["attentions"]["spatial_cross_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
                    sca.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["spatial_cross_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
                    sca.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["spatial_cross_attention"]["output_proj"] = {}
                parameters["attentions"]["spatial_cross_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
                    sca.output_proj.weight, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["spatial_cross_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
                    sca.output_proj.bias, dtype=ttnn.bfloat16
                )

        # Process FFN layers
        if hasattr(model, "ffns") and len(model.ffns) > 0:
            parameters["ffns"] = {}
            for ffn_idx, ffn in enumerate(model.ffns):
                parameters["ffns"][f"ffn{ffn_idx}"] = {}
                # FFN has structure: layers[0][0] (linear1), layers[1] (linear2)
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"] = {}
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["0"] = {}
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["0"]["0"] = {}
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["0"]["0"]["weight"] = preprocess_linear_weight(
                    ffn.layers[0][0].weight, dtype=ttnn.bfloat16
                )
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["0"]["0"]["bias"] = preprocess_linear_bias(
                    ffn.layers[0][0].bias, dtype=ttnn.bfloat16
                )
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["1"] = {}
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["1"]["weight"] = preprocess_linear_weight(
                    ffn.layers[1].weight, dtype=ttnn.bfloat16
                )
                parameters["ffns"][f"ffn{ffn_idx}"]["layers"]["1"]["bias"] = preprocess_linear_bias(
                    ffn.layers[1].bias, dtype=ttnn.bfloat16
                )

        # Process normalization layers
        if hasattr(model, "norms") and len(model.norms) > 0:
            parameters["norms"] = {}
            for norm_idx, norm in enumerate(model.norms):
                parameters["norms"][f"norm{norm_idx}"] = {}
                parameters["norms"][f"norm{norm_idx}"]["weight"] = ttnn.from_torch(
                    norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                parameters["norms"][f"norm{norm_idx}"]["bias"] = ttnn.from_torch(
                    norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )

    return parameters


def create_transformer_layer_parameters(model, device=None):
    """Create preprocessed parameters for the transformer layer."""
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_custom_base_transformer_layer(device, reset_seeds):
    """Test the custom base transformer layer with temporal self attention and spatial cross attention."""

    # Configuration
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    embed_dims = 256
    batch_size = 1
    num_query = 10000
    num_cams = 6
    num_key = 240
    batch_first = True

    # Create attention modules
    temporal_self_attn = TemporalSelfAttention(embed_dims=embed_dims, num_levels=1)
    spatial_cross_attn = SpatialCrossAttention(
        embed_dims=embed_dims, pc_range=point_cloud_range, batch_first=batch_first
    )

    # Create reference transformer layer
    operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
    torch_model = MyCustomBaseTransformerLayer(
        attn_cfgs=[temporal_self_attn, spatial_cross_attn],
        ffn_cfgs=dict(embed_dims=embed_dims),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )
    torch_model.eval()

    # Create input tensors
    query = torch.randn(batch_size, num_query, embed_dims)
    key = torch.randn(num_cams, num_key, batch_size, embed_dims)
    value = torch.randn(num_cams, num_key, batch_size, embed_dims)
    query_pos = torch.randn(batch_size, num_query, embed_dims)

    # Temporal self attention inputs
    reference_points_tsa = torch.randn(2, num_query, 1, 2)
    spatial_shapes_tsa = torch.tensor([[100, 100]])
    level_start_index_tsa = torch.tensor([0])

    # Spatial cross attention inputs
    reference_points_sca = torch.randn(batch_size, 4, num_query, 3)
    spatial_shapes_sca = torch.tensor([[12, 20]])
    reference_points_cam = torch.randn(num_cams, batch_size, num_query, 4, 2)
    bev_mask = torch.randn(num_cams, batch_size, num_query, 4)
    level_start_index_sca = torch.tensor([0])

    # Run reference model
    with torch.no_grad():
        torch_output = torch_model(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points_sca,
            spatial_shapes=spatial_shapes_sca,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index_sca,
        )

    # Preprocess parameters
    parameters = create_transformer_layer_parameters(torch_model, device=device)

    # Create TTNN attention modules
    tt_temporal_self_attn = TtTemporalSelfAttention(
        params=parameters.attentions.temporal_self_attention,
        device=device,
        embed_dims=embed_dims,
        num_levels=1,
    )

    tt_spatial_cross_attn = TtSpatialCrossAttention(
        device=device,
        params=parameters.attentions.spatial_cross_attention,
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
    )

    # Create TTNN transformer layer
    tt_model = TtCustomBaseTransformerLayer(
        params=parameters,
        device=device,
        attn_cfgs=[tt_temporal_self_attn, tt_spatial_cross_attn],
        ffn_cfgs=dict(embed_dims=embed_dims),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Convert inputs to TTNN
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_sca_tt = ttnn.from_torch(spatial_shapes_sca, device=device, dtype=ttnn.bfloat16)
    bev_mask_tt = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    level_start_index_sca_tt = ttnn.from_torch(level_start_index_sca, device=device, dtype=ttnn.bfloat16)

    # Run TTNN model
    tt_output = tt_model(
        query_tt,
        key=key_tt,
        value=value_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_sca,
        spatial_shapes=spatial_shapes_sca_tt,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask_tt,
        level_start_index=level_start_index_sca_tt,
    )

    # Convert output to torch and compare
    ttnn_output = ttnn.to_torch(tt_output)

    # Check PCC
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.95)
    logger.info(f"Custom Base Transformer Layer PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
    logger.info("Custom Base Transformer Layer test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_custom_base_transformer_layer_simple(device, reset_seeds):
    """Test a simpler version with just self_attn, norm, and ffn."""

    # Configuration
    embed_dims = 256
    batch_size = 1
    num_query = 100
    batch_first = True

    # Create attention module
    temporal_self_attn = TemporalSelfAttention(embed_dims=embed_dims, num_levels=1)

    # Create reference transformer layer (simplified operation order)
    operation_order = ("self_attn", "norm", "ffn", "norm")
    torch_model = MyCustomBaseTransformerLayer(
        attn_cfgs=[temporal_self_attn],
        ffn_cfgs=dict(embed_dims=embed_dims),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )
    torch_model.eval()

    # Create input tensors
    query = torch.randn(batch_size, num_query, embed_dims)
    query_pos = torch.randn(batch_size, num_query, embed_dims)
    reference_points = torch.randn(2, num_query, 1, 2)
    spatial_shapes = torch.tensor([[10, 10]])
    level_start_index = torch.tensor([0])

    # Run reference model
    with torch.no_grad():
        torch_output = torch_model(
            query,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

    # Preprocess parameters
    parameters = create_transformer_layer_parameters(torch_model, device=device)

    # Create TTNN attention module
    tt_temporal_self_attn = TtTemporalSelfAttention(
        params=parameters.attentions.temporal_self_attention,
        device=device,
        embed_dims=embed_dims,
        num_levels=1,
    )

    # Create TTNN transformer layer
    tt_model = TtCustomBaseTransformerLayer(
        params=parameters,
        device=device,
        attn_cfgs=[tt_temporal_self_attn],
        ffn_cfgs=dict(embed_dims=embed_dims),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Convert inputs to TTNN
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # Run TTNN model
    tt_output = tt_model(
        query_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
    )

    # Convert output to torch and compare
    ttnn_output = ttnn.to_torch(tt_output)

    # Check PCC
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.95)
    logger.info(f"Simple Transformer Layer PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
    logger.info("Simple Transformer Layer test passed!")
