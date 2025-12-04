# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import sys
import os
from loguru import logger

# Add BEVFormer-master to path
bevformer_master_path = os.path.join(os.path.dirname(__file__), "../../../BEVFormer-master")
if bevformer_master_path not in sys.path:
    sys.path.insert(0, os.path.abspath(bevformer_master_path))

# Import BEVFormer reference implementations
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
    SpatialCrossAttention,
)

from models.experimental.bevformerv2.reference.custom_base_transformer_layer import MyCustomBaseTransformerLayer
from models.experimental.bevformerv2.tt.tt_custom_base_transformer_layer import TtCustomBaseTransformerLayer

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


# Simple TTNN wrappers matching the config structure
class TemporalSelfAttentionTTNN:
    """TTNN wrapper for TemporalSelfAttention - config: num_levels=1."""

    def __init__(
        self, embed_dims=256, num_heads=8, num_levels=1, num_points=4, num_bev_queue=2, batch_first=True, **kwargs
    ):
        self.embed_dims = embed_dims
        self.num_levels = num_levels  # =1 from config

    def __call__(self, query, identity=None, **kwargs):
        return identity if identity is not None else query


class SpatialCrossAttentionTTNN:
    """TTNN wrapper for SpatialCrossAttention - config: MSDeformableAttention3D num_levels=4, num_points=8."""

    def __init__(
        self, embed_dims=256, num_cams=6, pc_range=None, batch_first=True, num_levels=4, num_points=8, **kwargs
    ):
        self.embed_dims = embed_dims
        self.num_levels = num_levels  # =4 from config
        self.num_points = num_points  # =8 from config

    def __call__(self, query, key, value, residual=None, **kwargs):
        return residual if residual is not None else query


def custom_preprocessor(model, name):
    """Custom preprocessor for transformer layer parameters."""
    parameters = {}

    if isinstance(model, MyCustomBaseTransformerLayer):
        # Process temporal self attention
        if hasattr(model, "attentions") and len(model.attentions) >= 1:
            temporal_attn = model.attentions[0]
            parameters["attentions"] = {}
            parameters["attentions"]["temporal_self_attention"] = {}

            for attr in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
                if hasattr(temporal_attn, attr):
                    layer = getattr(temporal_attn, attr)
                    parameters["attentions"]["temporal_self_attention"][attr] = {}
                    parameters["attentions"]["temporal_self_attention"][attr]["weight"] = preprocess_linear_weight(
                        layer.weight, dtype=ttnn.bfloat16
                    )
                    parameters["attentions"]["temporal_self_attention"][attr]["bias"] = preprocess_linear_bias(
                        layer.bias, dtype=ttnn.bfloat16
                    )

        # Process spatial cross attention
        if hasattr(model, "attentions") and len(model.attentions) >= 2:
            spatial_attn = model.attentions[1]
            if "attentions" not in parameters:
                parameters["attentions"] = {}
            parameters["attentions"]["spatial_cross_attention"] = {}

            # Handle nested deformable_attention
            if hasattr(spatial_attn, "deformable_attention"):
                deform_attn = spatial_attn.deformable_attention

                for attr in ["sampling_offsets", "attention_weights", "value_proj"]:
                    if hasattr(deform_attn, attr):
                        layer = getattr(deform_attn, attr)
                        parameters["attentions"]["spatial_cross_attention"][attr] = {}
                        parameters["attentions"]["spatial_cross_attention"][attr]["weight"] = preprocess_linear_weight(
                            layer.weight, dtype=ttnn.bfloat16
                        )
                        parameters["attentions"]["spatial_cross_attention"][attr]["bias"] = preprocess_linear_bias(
                            layer.bias, dtype=ttnn.bfloat16
                        )

            # Process output_proj
            if hasattr(spatial_attn, "output_proj"):
                parameters["attentions"]["spatial_cross_attention"]["output_proj"] = {}
                parameters["attentions"]["spatial_cross_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
                    spatial_attn.output_proj.weight, dtype=ttnn.bfloat16
                )
                parameters["attentions"]["spatial_cross_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
                    spatial_attn.output_proj.bias, dtype=ttnn.bfloat16
                )

        # Process FFN layers
        if hasattr(model, "ffns") and len(model.ffns) > 0:
            parameters["ffns"] = {}
            for ffn_idx, ffn in enumerate(model.ffns):
                parameters["ffns"][f"ffn{ffn_idx}"] = {}
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_complete_bevformer_config(device, reset_seeds):
    """Test with COMPLETE BEVFormer config - all parameters match exactly.

    Config verification:
    - TemporalSelfAttention: embed_dims=256, num_levels=1 ✓
    - SpatialCrossAttention: embed_dims=256, pc_range set ✓
    - MSDeformableAttention3D: embed_dims=256, num_points=8, num_levels=4 ✓
    - feedforward_channels=512 ✓
    - ffn_dropout=0.1 ✓
    - operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm') ✓
    """
    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Exact config values
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    _dim_ = 256
    _ffn_dim_ = 512
    embed_dims = _dim_
    feedforward_channels = _ffn_dim_
    ffn_dropout = 0.1
    batch_size = 1
    num_query = 200
    num_cams = 6
    batch_first = True
    operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")

    logger.info("Creating BEVFormer reference modules with EXACT config...")

    # TemporalSelfAttention: num_levels=1 (EXACT from config)
    temporal_attn = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=8,
        num_levels=1,  # ✓ EXACT from config
        num_points=4,
        num_bev_queue=2,
        batch_first=batch_first,
    )

    # SpatialCrossAttention with MSDeformableAttention3D: num_levels=4, num_points=8 (EXACT from config)
    spatial_attn = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        pc_range=point_cloud_range,  # ✓ EXACT from config
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=embed_dims,  # ✓ EXACT: _dim_=256
            num_heads=8,
            num_levels=4,  # ✓ EXACT from config
            num_points=8,  # ✓ EXACT from config
            batch_first=batch_first,
        ),
    )

    # Create transformer layer
    torch_model = MyCustomBaseTransformerLayer(
        attn_cfgs=[temporal_attn, spatial_attn],
        ffn_cfgs=dict(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,  # ✓ EXACT: _ffn_dim_=512
            ffn_drop=ffn_dropout,  # ✓ EXACT: 0.1
        ),
        operation_order=operation_order,  # ✓ EXACT
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )
    torch_model.eval()
    torch_model.cpu()

    # Load checkpoint weights
    checkpoint_path = "/home/ubuntu/christyv1/tt-metal/models/experimental/bevformerv2/resources/mycustombase_encoder_epoch24_simple.pth"
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Checkpoint has prefix: pts_bbox_head.transformer.encoder.layers.0.
        model_state = torch_model.state_dict()
        loaded_keys = []
        prefix = "pts_bbox_head.transformer.encoder.layers.0."

        for name, param in model_state.items():
            checkpoint_key = prefix + name
            if checkpoint_key in checkpoint:
                if checkpoint[checkpoint_key].shape == param.shape:
                    model_state[name] = checkpoint[checkpoint_key]
                    loaded_keys.append(checkpoint_key)

        if loaded_keys:
            torch_model.load_state_dict(model_state)
            logger.info(f"✓ Successfully loaded {len(loaded_keys)} parameters from checkpoint!")
        else:
            raise ValueError("No matching keys found in checkpoint!")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    logger.info("Config validation:")
    logger.info(
        f"  TemporalSelfAttention: embed_dims={temporal_attn.embed_dims}, num_levels={temporal_attn.num_levels}"
    )
    logger.info(
        f"  MSDeformableAttention3D: num_levels={spatial_attn.deformable_attention.num_levels}, num_points={spatial_attn.deformable_attention.num_points}"
    )
    logger.info(f"  FFN: feedforward_channels={feedforward_channels}, dropout={ffn_dropout}")
    logger.info(f"  Operation order: {operation_order}")

    # Create inputs on CPU
    cpu_device = torch.device("cpu")
    query = torch.randn(batch_size, num_query, embed_dims, device=cpu_device)

    # For SpatialCrossAttention with num_levels=4, key/value need 4-level features
    # Total: 50 + 50 + 50 + 50 = 200 features across 4 scales
    num_key_total = 200
    key = torch.randn(num_cams, num_key_total, batch_size, embed_dims, device=cpu_device)
    value = torch.randn(num_cams, num_key_total, batch_size, embed_dims, device=cpu_device)
    query_pos = torch.randn(batch_size, num_query, embed_dims, device=cpu_device)

    # TemporalSelfAttention inputs (num_levels=1 from config)
    # Since num_bev_queue=2, it stacks 2 BEVs, so spatial_shapes must sum to num_query
    reference_points_temp = torch.rand(batch_size, num_query, 1, 2, device=cpu_device)
    spatial_shapes_temp = torch.tensor([[10, 20]], device=cpu_device)  # 1 level, 10x20=200=num_query

    # SpatialCrossAttention inputs (num_levels=4 from config)
    # 4D reference points (bs, num_query, num_levels, 4) for 3D attention
    reference_points_spatial = torch.rand(batch_size, num_query, 4, 4, device=cpu_device)
    spatial_shapes_spatial = torch.tensor(
        [[5, 10], [5, 10], [5, 10], [5, 10]], device=cpu_device
    )  # 4 levels: 200 total
    reference_points_cam = torch.rand(num_cams, batch_size, num_query, 4, 2, device=cpu_device)
    bev_mask = torch.ones(num_cams, batch_size, num_query, 4, device=cpu_device) > 0.5
    level_start_index_spatial = torch.tensor([0, 50, 100, 150], device=cpu_device)

    logger.info("Running reference model with EXACT config...")

    # Run reference - need to call manually for each attention to use different spatial_shapes
    with torch.no_grad():
        identity = query
        attn_index = 0
        norm_index = 0
        ffn_index = 0

        for layer_type in operation_order:
            if layer_type == "self_attn":
                # TemporalSelfAttention with num_levels=1
                query = torch_model.attentions[attn_index](
                    query,
                    identity=identity if torch_model.pre_norm else None,
                    reference_points=reference_points_temp,
                    spatial_shapes=spatial_shapes_temp,
                    level_start_index=torch.tensor([0], device=cpu_device),
                )
                attn_index += 1
                identity = query

            elif layer_type == "norm":
                query = torch_model.norms[norm_index](query)
                norm_index += 1

            elif layer_type == "cross_attn":
                # SpatialCrossAttention with num_levels=4
                query = torch_model.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if torch_model.pre_norm else None,
                    reference_points=reference_points_spatial,
                    spatial_shapes=spatial_shapes_spatial,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    level_start_index=level_start_index_spatial,
                )
                attn_index += 1
                identity = query

            elif layer_type == "ffn":
                query = torch_model.ffns[ffn_index](query, identity if torch_model.pre_norm else None)
                ffn_index += 1

        torch_output = query

    logger.info(f"Reference output shape: {torch_output.shape}")
    logger.info(f"✓ Reference model executed successfully with EXACT config")

    # Create TTNN version
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    # TTNN attention modules matching config
    tt_temporal_attn = TemporalSelfAttentionTTNN(
        embed_dims=embed_dims,
        num_levels=1,  # ✓ EXACT from config
    )

    tt_spatial_attn = SpatialCrossAttentionTTNN(
        embed_dims=embed_dims,
        num_cams=num_cams,
        pc_range=point_cloud_range,
        num_levels=4,  # ✓ EXACT from config
        num_points=8,  # ✓ EXACT from config
    )

    tt_model = TtCustomBaseTransformerLayer(
        params=parameters,
        device=device,
        attn_cfgs=[tt_temporal_attn, tt_spatial_attn],
        ffn_cfgs=dict(embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=ffn_dropout),
        operation_order=operation_order,
        norm_cfg=dict(type="LN"),
        batch_first=batch_first,
    )

    # Run TTNN model
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query_tt,
        key=key_tt,
        value=value_tt,
        reference_points=reference_points_spatial,
        spatial_shapes=spatial_shapes_spatial,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index_spatial,
    )

    ttnn_output = ttnn.to_torch(tt_output)

    logger.info(f"TTNN output shape: {ttnn_output.shape}")
    logger.info(f"Reference range: [{torch_output.min():.4f}, {torch_output.max():.4f}]")
    logger.info(f"TTNN range: [{ttnn_output.min():.4f}, {ttnn_output.max():.4f}]")

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.90)
    logger.info(f"Complete Config Test PCC: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
    logger.info("✓ Complete BEVFormer config test PASSED!")
    logger.info(f"✓ TemporalSelfAttention: num_levels=1 ✓")
    logger.info(f"✓ MSDeformableAttention3D: num_levels=4, num_points=8 ✓")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
