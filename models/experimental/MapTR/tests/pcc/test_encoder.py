# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import numpy as np
import ttnn
from loguru import logger

from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefix for BEVFormer encoder
# pts_bbox_head.transformer.encoder.layers.X.attentions.0 = TemporalSelfAttention
# pts_bbox_head.transformer.encoder.layers.X.attentions.1 = GeometrySpatialCrossAttention
# pts_bbox_head.transformer.encoder.layers.X.ffns.0 = FFN
# pts_bbox_head.transformer.encoder.layers.X.norms.0/1/2 = LayerNorms
ENCODER_LAYER_PREFIX = "pts_bbox_head.transformer.encoder.layers."


def load_maptr_encoder_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}.")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
    else:
        full_state_dict = checkpoint

    encoder_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(ENCODER_LAYER_PREFIX):
            relative_key = key[len(ENCODER_LAYER_PREFIX) :]
            encoder_weights[relative_key] = value

    logger.info(f"Loaded {len(encoder_weights)} weight tensors for encoder")
    logger.info("=" * 60)
    logger.info("Encoder weight keys from checkpoint:")
    logger.info("=" * 60)
    for key in sorted(encoder_weights.keys()):
        shape = tuple(encoder_weights[key].shape)
        logger.info(f"  {key}: {shape}")
    logger.info("=" * 60)

    return encoder_weights


def load_torch_encoder(
    torch_model: BEVFormerEncoder,
    weights_path: str = MAPTR_WEIGHTS_PATH,
    num_layers: int = 1,
):
    encoder_weights = load_maptr_encoder_weights(weights_path)

    model_state_dict = torch_model.state_dict()

    logger.info("=" * 60)
    logger.info("Model state dict keys (expected):")
    logger.info("=" * 60)
    for key in sorted(model_state_dict.keys()):
        shape = tuple(model_state_dict[key].shape)
        logger.info(f"  {key}: {shape}")
    logger.info("=" * 60)

    new_state_dict = {}
    matched_keys = []
    missing_keys = []

    for model_key in model_state_dict.keys():
        # Map model keys to checkpoint keys
        # Model: layers.0.attentions.0.sampling_offsets.weight
        # Checkpoint: 0.attentions.0.sampling_offsets.weight
        # Only remove the leading "layers." prefix, not all occurrences
        if model_key.startswith("layers."):
            checkpoint_key = model_key[len("layers.") :]
        else:
            checkpoint_key = model_key

        # Handle FFN key mapping difference:
        # Model (flat Sequential): ffns.0.layers.0.weight, ffns.0.layers.3.weight
        # Checkpoint (nested Sequential): ffns.0.layers.0.0.weight, ffns.0.layers.1.weight
        if ".ffns." in checkpoint_key and ".layers." in checkpoint_key:
            # Map flat indices to nested structure
            # layers.0 -> layers.0.0 (first linear is nested in Sequential)
            # layers.3 -> layers.1 (second linear is at index 1)
            import re

            match = re.search(r"\.ffns\.(\d+)\.layers\.(\d+)\.(weight|bias)$", checkpoint_key)
            if match:
                ffn_idx = match.group(1)
                layer_idx = int(match.group(2))
                param_type = match.group(3)

                if layer_idx == 0:
                    # First linear: layers.0 -> layers.0.0
                    checkpoint_key = re.sub(
                        r"\.ffns\.(\d+)\.layers\.0\.(weight|bias)$",
                        f".ffns.{ffn_idx}.layers.0.0.{param_type}",
                        checkpoint_key,
                    )
                elif layer_idx == 3:
                    # Second linear: layers.3 -> layers.1
                    checkpoint_key = re.sub(
                        r"\.ffns\.(\d+)\.layers\.3\.(weight|bias)$",
                        f".ffns.{ffn_idx}.layers.1.{param_type}",
                        checkpoint_key,
                    )

        if checkpoint_key in encoder_weights:
            new_state_dict[model_key] = encoder_weights[checkpoint_key]
            matched_keys.append(model_key)
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key} (tried: {checkpoint_key})")
            new_state_dict[model_key] = model_state_dict[model_key]
            missing_keys.append(model_key)

    logger.info(f"Successfully matched {len(matched_keys)} weights")
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} weights: {missing_keys}")

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


def custom_preprocessor_encoder(model, name):
    parameters = {}

    if isinstance(model, BEVFormerEncoder):
        parameters["layers"] = {}

        for layer_idx, layer in enumerate(model.layers):
            layer_params = {}

            # Attention modules
            layer_params["attentions"] = {}

            # TemporalSelfAttention (attn0)
            tsa = layer.attentions[0]
            layer_params["attentions"]["attn0"] = {
                "sampling_offsets": {
                    "weight": preprocess_linear_weight(tsa.sampling_offsets.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(tsa.sampling_offsets.bias, dtype=ttnn.bfloat16),
                },
                "attention_weights": {
                    "weight": preprocess_linear_weight(tsa.attention_weights.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(tsa.attention_weights.bias, dtype=ttnn.bfloat16),
                },
                "value_proj": {
                    "weight": preprocess_linear_weight(tsa.value_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(tsa.value_proj.bias, dtype=ttnn.bfloat16),
                },
                "output_proj": {
                    "weight": preprocess_linear_weight(tsa.output_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(tsa.output_proj.bias, dtype=ttnn.bfloat16),
                },
            }

            # SpatialCrossAttention with MSDeformableAttention3D (attn1)
            sca = layer.attentions[1]
            layer_params["attentions"]["attn1"] = {
                "sampling_offsets": {
                    "weight": preprocess_linear_weight(
                        sca.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
                    ),
                    "bias": preprocess_linear_bias(sca.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16),
                },
                "attention_weights": {
                    "weight": preprocess_linear_weight(
                        sca.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
                    ),
                    "bias": preprocess_linear_bias(
                        sca.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16
                    ),
                },
                "value_proj": {
                    "weight": preprocess_linear_weight(sca.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(sca.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16),
                },
                "output_proj": {
                    "weight": preprocess_linear_weight(sca.output_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(sca.output_proj.bias, dtype=ttnn.bfloat16),
                },
            }

            # FFN
            layer_params["ffn"] = {}
            ffn = layer.ffns[0]
            # FFN has flat Sequential: [Linear, ReLU, Dropout, Linear, Dropout]
            # With num_fcs=2 (default):
            # layers[0] = Linear(embed_dims, feedforward_channels)
            # layers[1] = ReLU
            # layers[2] = Dropout
            # layers[3] = Linear(feedforward_channels, embed_dims)
            # layers[4] = Dropout
            layer_params["ffn"]["ffn0"] = {
                "linear1": {
                    "weight": preprocess_linear_weight(ffn.layers[0].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(ffn.layers[0].bias, dtype=ttnn.bfloat16),
                },
                "linear2": {
                    "weight": preprocess_linear_weight(ffn.layers[3].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(ffn.layers[3].bias, dtype=ttnn.bfloat16),
                },
            }

            # LayerNorms - use preprocess_layernorm_parameter which reshapes to (1, hidden_dim)
            layer_params["norms"] = {}
            for norm_idx, norm in enumerate(layer.norms):
                layer_params["norms"][f"norm{norm_idx}"] = {
                    "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                }

            parameters["layers"][f"layer{layer_idx}"] = layer_params

    return parameters


def create_encoder_parameters(model: BEVFormerEncoder, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_encoder,
        device=device,
    )
    return parameters


def create_dummy_img_metas(batch_size: int = 1, num_cams: int = 6):
    # Typical image shape: (H, W)
    img_h, img_w = 450, 800
    img_shape = [(img_h, img_w)] * num_cams

    # Create proper lidar2img matrices for 6 cameras around the vehicle
    # Each camera has: lidar2img = K @ [R | t] where K is intrinsic, R|t is extrinsic
    lidar2img_list = []

    # Camera intrinsics (shared)
    fx, fy = 400.0, 400.0  # focal length
    cx, cy = img_w / 2, img_h / 2  # principal point

    # Camera positions around vehicle (front, front-left, front-right, back, back-left, back-right)
    # Angles in radians from forward direction
    cam_angles = [0, np.pi / 3, -np.pi / 3, np.pi, 2 * np.pi / 3, -2 * np.pi / 3]

    for cam_idx in range(num_cams):
        angle = cam_angles[cam_idx % len(cam_angles)]

        # Rotation matrix (camera looks outward from vehicle center)
        # In lidar coords: x=right, y=forward, z=up
        # Camera convention: z=forward (depth), x=right, y=down
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotation from lidar to camera frame
        R = np.array(
            [
                [cos_a, sin_a, 0],  # camera x from lidar x,y
                [0, 0, -1],  # camera y from -lidar z (flip up to down)
                [sin_a, -cos_a, 0],  # camera z (depth) from lidar y,x
            ],
            dtype=np.float32,
        )

        # Translation: camera at height 1.5m, 2m from center in viewing direction
        t = np.array(
            [
                -2 * sin_a,  # x offset
                -2 * cos_a,  # y offset
                -1.5,  # z offset (height)
            ],
            dtype=np.float32,
        )

        # Intrinsic matrix
        K = np.array(
            [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Extrinsic matrix [R | t]
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R
        RT[:3, 3] = R @ t  # Transform translation to camera frame

        # Combined projection matrix
        lidar2img = K @ RT
        lidar2img_list.append(lidar2img)

    lidar2img = np.stack(lidar2img_list, axis=0)  # (num_cams, 4, 4)

    img_metas = [
        {
            "img_shape": img_shape,
            "lidar2img": lidar2img,
        }
        for _ in range(batch_size)
    ]

    return img_metas


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_bevformer_encoder(device, reset_seeds):
    # MapTR config parameters (reduced BEV size for memory constraints)
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    embed_dims = 256
    num_heads = 8  # MSDeformableAttention3D uses 8 heads
    feedforward_channels = 512
    num_levels = 1  # Single feature level
    num_points = 8  # MSDeformableAttention3D num_points
    num_layers = 1  # Test with 1 layer for simplicity
    bev_h = 50  # Reduced from 200 for memory
    bev_w = 25  # Reduced from 100 for memory
    num_cams = 6
    batch_size = 1

    # Create PyTorch model using config-based approach (matching MapTR reference)
    transformerlayers_cfg = dict(
        type="BEVFormerLayer",
        attn_cfgs=[
            dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
            dict(
                type="SpatialCrossAttention",
                pc_range=point_cloud_range,
                deformable_attention=dict(
                    type="MSDeformableAttention3D",
                    embed_dims=embed_dims,
                    num_points=num_points,
                    num_levels=num_levels,
                ),
                embed_dims=embed_dims,
            ),
        ],
        feedforward_channels=feedforward_channels,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    )

    torch_model = BEVFormerEncoder(
        transformerlayer=transformerlayers_cfg,  # Note: singular 'transformerlayer'
        num_layers=num_layers,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
    )

    # Load mapTR weights
    torch_model = load_torch_encoder(torch_model, num_layers=num_layers)

    # Create input tensors
    num_query = bev_h * bev_w

    # BEV query: (num_query, bs, embed_dims) - before permute in forward
    bev_query = torch.randn(num_query, batch_size, embed_dims)

    # Key/Value from backbone: (num_cams, num_feat, bs, embed_dims)
    num_feat_per_level = 12 * 20  # Example: 12x20 feature map
    key = torch.randn(num_cams, num_feat_per_level, batch_size, embed_dims)
    value = torch.randn(num_cams, num_feat_per_level, batch_size, embed_dims)

    # BEV positional encoding: (num_query, bs, embed_dims)
    bev_pos = torch.randn(num_query, batch_size, embed_dims)

    # Spatial shapes for feature levels
    spatial_shapes = torch.tensor([[12, 20]])
    level_start_index = torch.tensor([0])

    # Shift for temporal alignment
    shift = torch.zeros(batch_size, 2)

    # Image metadata
    img_metas = create_dummy_img_metas(batch_size, num_cams)

    # Run PyTorch model
    with torch.no_grad():
        torch_output = torch_model(
            bev_query,
            key,
            value,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=shift,
            img_metas=img_metas,
        )

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Prepare TT model parameters
    parameters = create_encoder_parameters(torch_model, device)

    # Create TT model
    tt_model = TtBEVFormerEncoder(
        params=parameters,
        device=device,
        num_layers=num_layers,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=embed_dims,
        num_heads=num_heads,
        feedforward_channels=feedforward_channels,
        num_levels=num_levels,
        num_points=num_points,
    )

    # Convert inputs to TT tensors
    bev_query_tt = ttnn.from_torch(bev_query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    bev_pos_tt = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)
    shift_tt = ttnn.from_torch(shift, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    tt_output = tt_model(
        bev_query_tt,
        key_tt,
        value_tt,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
        prev_bev=None,
        shift=shift_tt,
        img_metas=img_metas,
    )

    # Compare outputs
    ttnn_output = ttnn.to_torch(tt_output)
    logger.info(f"TT output shape: {ttnn_output.shape}")
    logger.info(
        f"TT output stats: min={ttnn_output.min():.4f}, max={ttnn_output.max():.4f}, mean={ttnn_output.mean():.4f}"
    )

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.97)
    logger.info(f"PCC Result: {pcc_message}")
