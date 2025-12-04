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
def test_custom_base_transformer_layer_with_checkpoint(device, reset_seeds):
    """Test the custom base transformer layer with checkpoint weights - BEVFormer config."""

    # Load checkpoint
    checkpoint_path = "/home/ubuntu/christyv1/tt-metal/models/experimental/bevformerv2/resources/mycustombase_encoder_epoch24_simple.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")

    # BEVFormer configuration from vadv2 test
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    embed_dims = 256
    _ffn_dim_ = 512
    batch_size = 1
    num_query = 10000  # BEV grid size (100x100=10000)
    num_cams = 6
    num_key = 240  # spatial feature map size for cross attention (12*20=240)
    batch_first = True

    # Create attention modules matching vadv2 test config
    # TemporalSelfAttention config
    temporal_self_attn = TemporalSelfAttention(
        embed_dims=embed_dims, num_heads=8, num_levels=1, num_points=4, num_bev_queue=2, batch_first=batch_first
    )

    # SpatialCrossAttention config
    spatial_cross_attn = SpatialCrossAttention(
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=embed_dims, num_heads=8, num_points=8, num_levels=1
        ),
    )

    # Create reference transformer layer with operation_order from BEVFormer config
    operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
    torch_model = MyCustomBaseTransformerLayer(
        attn_cfgs=[temporal_self_attn, spatial_cross_attn],
        ffn_cfgs=dict(embed_dims=embed_dims, feedforward_channels=_ffn_dim_),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Load weights from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Try to load the state dict
    try:
        torch_model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint weights into model (non-strict)")
    except Exception as e:
        logger.warning(f"Could not load checkpoint weights: {e}")
        logger.info("Using random weights for testing")

    torch_model.eval()

    # Create input tensors - use BEV grid size consistently
    # Both attentions will use the BEV grid spatial shape
    query = torch.randn(batch_size, num_query, embed_dims)

    # For this test, use same spatial shapes for both attentions
    # In real BEVFormer, key/value come from image features with different sizes
    # But for testing the transformer layer, we can use matching sizes
    spatial_shapes = torch.tensor([[100, 100]])  # BEV grid 100x100 = 10000
    level_start_index = torch.tensor([0])

    # Use num_key=10000 to match spatial_shapes for this test
    key = torch.randn(num_cams, num_query, batch_size, embed_dims)
    value = torch.randn(num_cams, num_query, batch_size, embed_dims)
    query_pos = torch.randn(batch_size, num_query, embed_dims)

    # Use 2D reference points in TemporalSelfAttention format
    # [num_bev_queue, num_query, num_levels, 2]
    reference_points = torch.randn(2, num_query, 1, 2)  # 2D points for temporal attention

    # Spatial cross attention specific inputs
    reference_points_cam = torch.randn(num_cams, batch_size, num_query, 4, 2)
    bev_mask = torch.randn(num_cams, batch_size, num_query, 4)

    # Run reference model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        torch_output = torch_model(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
        )

    logger.info(f"PyTorch output shape: {torch_output.shape}")

    # Preprocess parameters
    logger.info("Preprocessing parameters for TTNN...")
    parameters = create_transformer_layer_parameters(torch_model, device=device)

    # Create TTNN attention modules
    tt_temporal_self_attn = TtTemporalSelfAttention(
        params=parameters.attentions.temporal_self_attention,
        device=device,
        embed_dims=embed_dims,
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        batch_first=batch_first,
    )

    tt_spatial_cross_attn = TtSpatialCrossAttention(
        device=device,
        params=parameters.attentions.spatial_cross_attention,
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
    )

    # Create TTNN transformer layer
    logger.info("Creating TTNN transformer layer...")
    tt_model = TtCustomBaseTransformerLayer(
        params=parameters,
        device=device,
        attn_cfgs=[tt_temporal_self_attn, tt_spatial_cross_attn],
        ffn_cfgs=dict(embed_dims=embed_dims, feedforward_channels=_ffn_dim_),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Convert inputs to TTNN
    logger.info("Converting inputs to TTNN...")
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    bev_mask_tt = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # Run TTNN model
    logger.info("Running TTNN model...")
    tt_output = tt_model(
        query_tt,
        key=key_tt,
        value=value_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask_tt,
        level_start_index=level_start_index_tt,
    )

    # Convert output to torch and compare
    logger.info("Converting TTNN output to PyTorch...")
    ttnn_output = ttnn.to_torch(tt_output)

    logger.info(f"TTNN output shape: {ttnn_output.shape}")
    logger.info(f"PyTorch output stats - mean: {torch_output.mean():.6f}, std: {torch_output.std():.6f}")
    logger.info(f"TTNN output stats - mean: {ttnn_output.mean():.6f}, std: {ttnn_output.std():.6f}")

    # Check PCC
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.95)
    logger.info(f"Custom Base Transformer Layer PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
    logger.info("✅ Custom Base Transformer Layer test with checkpoint PASSED!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_custom_base_transformer_layer_simple_with_checkpoint(device, reset_seeds):
    """Test a simpler version with just self_attn, norm, and ffn using checkpoint weights."""

    # Load checkpoint
    checkpoint_path = "/home/ubuntu/christyv1/tt-metal/models/experimental/bevformerv2/resources/mycustombase_encoder_epoch24_simple.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Configuration - use same as vadv2 test for compatibility
    embed_dims = 256
    _ffn_dim_ = 512
    batch_size = 1
    num_query = 10000  # Use same as vadv2 test
    batch_first = True

    # Create attention module - explicitly set num_heads=8 so head_dim=32 (TILE_WIDTH compatible)
    temporal_self_attn = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=8,  # Explicitly set to ensure head_dim=256/8=32
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        batch_first=batch_first,
    )

    # Create reference transformer layer (simplified operation order)
    operation_order = ("self_attn", "norm", "ffn", "norm")
    torch_model = MyCustomBaseTransformerLayer(
        attn_cfgs=[temporal_self_attn],
        ffn_cfgs=dict(embed_dims=embed_dims, feedforward_channels=_ffn_dim_),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Try to load weights from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    try:
        torch_model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint weights into model (non-strict)")
    except Exception as e:
        logger.warning(f"Could not load checkpoint weights: {e}")
        logger.info("Using random weights for testing")

    torch_model.eval()

    # Create input tensors - match vadv2 test
    query = torch.randn(batch_size, num_query, embed_dims)
    query_pos = torch.randn(batch_size, num_query, embed_dims)
    reference_points = torch.randn(2, num_query, 1, 2)  # 2 is num_bev_queue
    spatial_shapes = torch.tensor([[100, 100]])  # H=100, W=100, total=10000
    level_start_index = torch.tensor([0])

    # Run reference model
    logger.info("Running PyTorch reference model...")
    logger.info(f"query shape: {query.shape}")
    logger.info(f"query_pos shape: {query_pos.shape}")
    logger.info(f"reference_points shape: {reference_points.shape}")
    logger.info(f"spatial_shapes: {spatial_shapes}")
    with torch.no_grad():
        torch_output = torch_model(
            query,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

    logger.info(f"PyTorch output shape: {torch_output.shape}")

    # Preprocess parameters
    logger.info("Preprocessing parameters for TTNN...")
    parameters = create_transformer_layer_parameters(torch_model, device=device)

    # Create TTNN attention module
    tt_temporal_self_attn = TtTemporalSelfAttention(
        params=parameters.attentions.temporal_self_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=8,  # Explicitly set to ensure head_dim=256/8=32
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        batch_first=batch_first,
    )

    # Create TTNN transformer layer
    logger.info("Creating TTNN transformer layer...")
    tt_model = TtCustomBaseTransformerLayer(
        params=parameters,
        device=device,
        attn_cfgs=[tt_temporal_self_attn],
        ffn_cfgs=dict(embed_dims=embed_dims, feedforward_channels=_ffn_dim_),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Convert inputs to TTNN
    logger.info("Converting inputs to TTNN...")
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # Run TTNN model
    logger.info("Running TTNN model...")
    tt_output = tt_model(
        query_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
    )

    # Convert output to torch and compare
    logger.info("Converting TTNN output to PyTorch...")
    ttnn_output = ttnn.to_torch(tt_output)

    logger.info(f"TTNN output shape: {ttnn_output.shape}")
    logger.info(f"PyTorch output stats - mean: {torch_output.mean():.6f}, std: {torch_output.std():.6f}")
    logger.info(f"TTNN output stats - mean: {ttnn_output.mean():.6f}, std: {ttnn_output.std():.6f}")

    # Check PCC
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.95)
    logger.info(f"Simple Transformer Layer PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
    logger.info("✅ Simple Transformer Layer test with checkpoint PASSED!")


if __name__ == "__main__":
    # For standalone testing
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
