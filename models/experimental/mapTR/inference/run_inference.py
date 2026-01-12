# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Full Inference Pipeline

This script runs the complete MapTR inference pipeline with:
- Image loading and preprocessing
- Model weight loading
- Multi-camera inference
- Result visualization

Based on official MapTR repository: https://github.com/hustvl/MapTR

Usage:
    python models/experimental/mapTR/inference/run_inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --images /path/to/images/ \
        --output /path/to/output/

    # Or with demo mode (generates dummy data):
    python models/experimental/mapTR/inference/run_inference.py --demo
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR components
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_maptr import MapTR
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.pytorch_temporal_self_attention import multi_scale_deformable_attn_pytorch


# ============================================================================
# Configuration Classes
# ============================================================================


class MapTRConfig:
    """Configuration for MapTR model."""

    def __init__(
        self,
        # Image settings
        img_height: int = 900,
        img_width: int = 1600,
        num_cameras: int = 6,
        # Model architecture
        embed_dims: int = 256,
        num_classes: int = 3,  # divider, ped_crossing, boundary
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        # BEV settings
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        # Transformer settings
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        # Backbone settings
        backbone_depth: int = 50,
        fpn_out_channels: int = 256,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.num_cameras = num_cameras

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels

        self.backbone_depth = backbone_depth
        self.fpn_out_channels = fpn_out_channels

    @classmethod
    def from_nuScenes(cls):
        """Create config for nuScenes dataset."""
        return cls(
            img_height=900,
            img_width=1600,
            num_cameras=6,
            num_classes=3,
            num_vec=50,
            num_pts_per_vec=20,
            bev_h=200,
            bev_w=100,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        )

    @classmethod
    def from_maptr_tiny(cls):
        """Create config for MapTR Tiny R50 checkpoint (maptr_tiny_r50_24e.pth).

        Based on official MapTR config: https://github.com/hustvl/MapTR
        - Only 1 encoder layer (tiny variant)
        - 6 decoder layers
        - 512 FFN channels
        """
        return cls(
            img_height=480,
            img_width=800,
            num_cameras=6,
            num_classes=3,
            num_vec=50,
            num_pts_per_vec=20,
            bev_h=200,
            bev_w=100,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_encoder_layers=1,  # Tiny variant has only 1 encoder layer
            num_decoder_layers=6,
            feedforward_channels=512,  # Matches checkpoint
        )

    @classmethod
    def from_small(cls):
        """Create small config for testing/demo (no pretrained weights)."""
        return cls(
            img_height=224,
            img_width=400,
            num_cameras=6,
            num_classes=3,
            num_vec=20,
            num_pts_per_vec=10,
            bev_h=50,
            bev_w=50,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_encoder_layers=2,
            num_decoder_layers=2,
        )


# ============================================================================
# Model Builder
# ============================================================================


class MapTRTransformerWithBEVFormer(nn.Module):
    """MapTR Transformer with BEVFormer encoder for proper weight loading.

    Based on official MapTR: https://github.com/hustvl/MapTR
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points_in_pillar: int = 4,
        num_cams: int = 6,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        feedforward_channels: int = 512,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        # BEVFormer Encoder - checkpoint uses 512 FFN channels
        self.encoder = BEVFormerEncoder(
            num_layers=num_encoder_layers,
            pc_range=self.pc_range,
            num_points_in_pillar=num_points_in_pillar,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=512,  # Matches checkpoint
        )

        # Decoder - using proper deformable attention (matches checkpoint)
        # Note: checkpoint uses 512 FFN channels for decoder
        self.decoder = MapTRDecoder(
            embed_dims=embed_dims,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            feedforward_channels=512,  # Matches checkpoint: 512 not 2048
            num_points=4,  # Matches checkpoint
            return_intermediate=True,
        )

        # Additional layers matching checkpoint keys
        self.level_embeds = nn.Parameter(torch.randn(4, embed_dims))
        self.cams_embeds = nn.Parameter(torch.randn(num_cams, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        object_query_embeds: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: tuple,
        bev_pos: torch.Tensor,
        reg_branches: nn.ModuleList = None,
        cls_branches: nn.ModuleList = None,
        img_metas: List[Dict] = None,
        prev_bev: torch.Tensor = None,
    ):
        """Forward pass through BEVFormer encoder and decoder."""
        bs = mlvl_feats[0].size(0)

        # Prepare BEV queries
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos_flat = bev_pos.flatten(2).permute(2, 0, 1)

        # Add CAN bus info
        if img_metas is not None:
            can_bus = torch.zeros(bs, 18, device=bev_queries.device, dtype=bev_queries.dtype)
            for i, meta in enumerate(img_metas):
                if "can_bus" in meta:
                    can_bus_data = meta["can_bus"]
                    if isinstance(can_bus_data, np.ndarray):
                        can_bus[i] = torch.from_numpy(can_bus_data[:18]).to(can_bus.device)
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

        # Compute shift for temporal fusion
        shift = torch.zeros(bs, 2, device=bev_queries.device, dtype=bev_queries.dtype)

        # Prepare multi-level features
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs_f, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cam, bs, h*w, c)
            if self.cams_embeds is not None:
                feat = feat + self.cams_embeds[:num_cam, None, None, :].to(feat.dtype)
            if lvl < len(self.level_embeds):
                feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, h*w, bs, c)

        # Run BEVFormer encoder
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_flat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas,
        )

        # Prepare decoder queries
        query_embeds = object_query_embeds
        query = query_embeds[..., : self.embed_dims]
        query_pos = query_embeds[..., self.embed_dims :]

        query = query.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)

        # Reference points
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # Spatial shapes for decoder attention
        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)

        # Run decoder with deformable attention
        hs, inter_references = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        return bev_embed.permute(1, 0, 2), hs, init_reference_out, inter_references

    def get_bev_features(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: tuple,
        bev_pos: torch.Tensor,
        img_metas: List[Dict],
        prev_bev: torch.Tensor = None,
    ):
        """Get BEV features only."""
        bs = mlvl_feats[0].size(0)
        bev_embed = bev_queries.unsqueeze(0).repeat(bs, 1, 1)
        return bev_embed


class DeformableSelfAttention(nn.Module):
    """Deformable Self Attention for decoder (matches MapTR checkpoint structure).

    This uses the same deformable attention mechanism as the encoder's temporal
    self-attention but without the BEV queue for historical frames.

    Based on: https://github.com/hustvl/MapTR
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}")

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function."""
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(f"reference_points last dim must be 2, got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output + identity


class CustomMSDeformableAttention(nn.Module):
    """Cross attention using deformable attention for decoder (matches checkpoint)."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # Handle reference points
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(f"reference_points last dim must be 2, got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output + identity


class MapTRDecoderLayer(nn.Module):
    """MapTR Decoder layer matching checkpoint structure.

    Based on official MapTR: https://github.com/hustvl/MapTR

    Checkpoint structure (decoder.layers.X):
    - attentions.0 = Standard MultiheadAttention (self-attention) with attn.in_proj_*, attn.out_proj.*
    - attentions.1 = Deformable Attention (cross-attention) with sampling_offsets, attention_weights, etc.
    - ffns.0.layers.0.0/1 = FFN layers
    - norms.0/1/2 = LayerNorms
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        num_points: int = 4,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # Self attention - Standard MultiheadAttention (matches checkpoint: attentions.0.attn.*)
        # We wrap it to match checkpoint key structure: self_attn.attn.*
        self.self_attn = SelfAttnWrapper(embed_dims, num_heads)

        # Cross attention - Deformable (matches checkpoint: attentions.1.*)
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
        )

        # FFN - matches checkpoint ffns.0.layers.0.0/1.*
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_channels, embed_dims),
        )

        # Layer norms - matches checkpoint norms.0/1/2.*
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self attention (standard MultiheadAttention)
        q = k = query + query_pos if query_pos is not None else query
        query2 = self.self_attn(q, k, query)
        query = self.norm1(query + query2)

        # Cross attention (deformable)
        query = self.cross_attn(
            query=query,
            key=value,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)

        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)

        return query


class SelfAttnWrapper(nn.Module):
    """Wrapper for MultiheadAttention to match checkpoint key structure.

    Checkpoint has: decoder.layers.X.attentions.0.attn.in_proj_weight, etc.
    This creates: self_attn.attn.in_proj_weight (after remapping attentions.0 -> self_attn)
    """

    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)

    def forward(self, query, key, value):
        return self.attn(query, key, value)[0]


class MapTRDecoder(nn.Module):
    """MapTR Decoder with iterative refinement (matches checkpoint).

    Based on official MapTR: https://github.com/hustvl/MapTR
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        num_points: int = 4,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList(
            [
                MapTRDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    num_points=num_points,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        reg_branches: nn.ModuleList = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function with iterative refinement."""
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Prepare reference points for attention
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # (bs, num_query, 1, 2)

            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(output.permute(1, 0, 2))  # (num_query, bs, embed_dims)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def build_maptr_model(config: MapTRConfig, device: torch.device = None) -> MapTR:
    """Build MapTR model from config.

    Args:
        config: Model configuration.
        device: Target device.

    Returns:
        MapTR model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build backbone (ResNet50 by default)
    if config.backbone_depth == 50:
        layers = [3, 4, 6, 3]
    elif config.backbone_depth == 101:
        layers = [3, 4, 23, 3]
    else:
        layers = [3, 4, 6, 3]

    backbone = ResNet(
        block=Bottleneck,
        layers=layers,
        out_indices=(3,),  # Output from layer4 (2048 channels) to match checkpoint
    )

    # Build FPN
    fpn = FPN(
        in_channels=[2048],  # layer4 output has 2048 channels
        out_channels=config.fpn_out_channels,
        num_outs=1,
    )

    # Build transformer with BEVFormer encoder
    transformer = MapTRTransformerWithBEVFormer(
        embed_dims=config.embed_dims,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        num_points_in_pillar=4,
        num_cams=config.num_cameras,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        feedforward_channels=config.feedforward_channels,
    )

    # Build positional encoding
    pos_enc = LearnedPositionalEncoding(
        num_feats=config.embed_dims // 2,
        row_num_embed=config.bev_h,
        col_num_embed=config.bev_w,
    )

    # Build head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=pos_enc,
        embed_dims=config.embed_dims,
        num_classes=config.num_classes,
        num_reg_fcs=2,
        code_size=2,  # (x, y) per point
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        query_embed_type="instance_pts",  # Match checkpoint: separate instance and pts embeddings
        transform_method="minmax",
    )

    # Build full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    return model.to(device)


# ============================================================================
# Image Processing
# ============================================================================


class ImageProcessor:
    """Process images for MapTR inference."""

    def __init__(
        self,
        img_height: int = 900,
        img_width: int = 1600,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean or [123.675, 116.28, 103.53]
        self.std = std or [58.395, 57.12, 57.375]

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        return np.array(img)

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess images for model input.

        Args:
            images: List of images as numpy arrays (H, W, C) in RGB format.

        Returns:
            Tensor of shape (1, num_cams, 3, H, W).
        """
        processed = []
        for img in images:
            # Normalize
            img = img.astype(np.float32)
            img = (img - np.array(self.mean)) / np.array(self.std)
            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            processed.append(img)

        # Stack cameras: (num_cams, 3, H, W)
        imgs = np.stack(processed, axis=0)
        # Add batch dimension: (1, num_cams, 3, H, W)
        imgs = imgs[np.newaxis, ...]
        return torch.from_numpy(imgs).float()

    def generate_dummy_images(self, num_cameras: int = 6) -> torch.Tensor:
        """Generate dummy images for testing."""
        return torch.randn(1, num_cameras, 3, self.img_height, self.img_width)


# ============================================================================
# Camera Calibration
# ============================================================================


class CameraCalibration:
    """Handle camera calibration for MapTR."""

    def __init__(self, num_cameras: int = 6):
        self.num_cameras = num_cameras

    def create_dummy_calibration(
        self,
        img_height: int = 900,
        img_width: int = 1600,
    ) -> Dict[str, np.ndarray]:
        """Create dummy camera calibration for testing.

        Returns:
            Dictionary with calibration matrices.
        """
        # Create identity transforms for dummy data
        lidar2img = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera2ego = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera_intrinsics = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        img_aug_matrix = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        lidar2ego = np.eye(4)

        # Set reasonable intrinsics
        fx, fy = 1000.0, 1000.0
        cx, cy = img_width / 2, img_height / 2
        for i in range(self.num_cameras):
            camera_intrinsics[i, 0, 0] = fx
            camera_intrinsics[i, 1, 1] = fy
            camera_intrinsics[i, 0, 2] = cx
            camera_intrinsics[i, 1, 2] = cy

        return {
            "lidar2img": lidar2img,
            "camera2ego": camera2ego,
            "camera_intrinsics": camera_intrinsics,
            "img_aug_matrix": img_aug_matrix,
            "lidar2ego": lidar2ego,
        }

    def load_calibration(self, calibration_path: str) -> Dict[str, np.ndarray]:
        """Load calibration from JSON file."""
        with open(calibration_path, "r") as f:
            calib = json.load(f)

        return {
            "lidar2img": np.array(calib.get("lidar2img", np.eye(4))),
            "camera2ego": np.array(calib.get("camera2ego", np.eye(4))),
            "camera_intrinsics": np.array(calib.get("camera_intrinsics", np.eye(4))),
            "img_aug_matrix": np.array(calib.get("img_aug_matrix", np.eye(4))),
            "lidar2ego": np.array(calib.get("lidar2ego", np.eye(4))),
        }


# ============================================================================
# nuScenes Data Loader (Official MapTR approach)
# ============================================================================


class NuScenesDataLoader:
    """nuScenes data loader following official MapTR repository approach.

    Based on: https://github.com/hustvl/MapTR

    This loader handles:
    - Loading images from all 6 cameras
    - Loading camera calibration from nuScenes metadata
    - Loading CAN bus data for temporal fusion
    - Proper image preprocessing

    Directory structure expected:
        data_root/
        ├── nuscenes/
        │   ├── samples/
        │   │   ├── CAM_FRONT/
        │   │   ├── CAM_FRONT_LEFT/
        │   │   ├── CAM_FRONT_RIGHT/
        │   │   ├── CAM_BACK/
        │   │   ├── CAM_BACK_LEFT/
        │   │   └── CAM_BACK_RIGHT/
        │   ├── v1.0-mini/ (or v1.0-trainval)
        │   └── maps/
        └── can_bus/
            ├── scene-0001/
            │   ├── pose.json
            │   ├── ms_imu.json
            │   └── ...
            └── ...
    """

    # Camera names in nuScenes order
    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-mini",
        img_height: int = 480,
        img_width: int = 800,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        """Initialize nuScenes data loader.

        Args:
            data_root: Root directory containing nuscenes/ and can_bus/ folders.
            version: nuScenes version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test').
            img_height: Target image height after resize.
            img_width: Target image width after resize.
            mean: Image normalization mean (default: ImageNet).
            std: Image normalization std (default: ImageNet).
        """
        self.data_root = Path(data_root)
        self.version = version
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean or [123.675, 116.28, 103.53]
        self.std = std or [58.395, 57.12, 57.375]

        # Paths
        self.nuscenes_root = self.data_root / "nuscenes"
        self.canbus_root = self.data_root / "can_bus"

        # Initialize nuScenes API if available
        self.nusc = None
        self._init_nuscenes()

    def _init_nuscenes(self):
        """Initialize nuScenes devkit."""
        try:
            from nuscenes.nuscenes import NuScenes

            nuscenes_dataroot = str(self.nuscenes_root)
            if (self.nuscenes_root / self.version).exists():
                # Standard structure
                self.nusc = NuScenes(
                    version=self.version,
                    dataroot=nuscenes_dataroot,
                    verbose=True,
                )
                print(f"✓ Loaded nuScenes {self.version} with {len(self.nusc.sample)} samples")
            else:
                print(f"⚠ nuScenes metadata not found at {self.nuscenes_root / self.version}")
                print("  Will use manual file loading mode.")

        except ImportError:
            print("⚠ nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
            print("  Will use manual file loading mode.")
        except Exception as e:
            print(f"⚠ Could not initialize nuScenes: {e}")
            print("  Will use manual file loading mode.")

    def load_can_bus(self, scene_token: str, sample_token: str) -> np.ndarray:
        """Load CAN bus data for a sample.

        CAN bus data contains 18 values:
        - [0:3]: translation (x, y, z)
        - [3:7]: rotation quaternion (w, x, y, z)
        - [7:10]: velocity (vx, vy, vz)
        - [10:13]: acceleration (ax, ay, az)
        - [13:16]: angular velocity (wx, wy, wz)
        - [16]: steering angle
        - [17]: yaw rate

        Args:
            scene_token: Scene token.
            sample_token: Sample token.

        Returns:
            CAN bus data array of shape (18,).
        """
        can_bus = np.zeros(18)

        if self.nusc is None:
            return can_bus

        try:
            # Get scene name from token
            scene = self.nusc.get("scene", scene_token)
            scene_name = scene["name"]

            # Load pose data from can_bus folder
            pose_file = self.canbus_root / scene_name / "pose.json"

            if pose_file.exists():
                with open(pose_file, "r") as f:
                    pose_data = json.load(f)

                # Get sample timestamp
                sample = self.nusc.get("sample", sample_token)
                timestamp = sample["timestamp"]

                # Find closest pose by timestamp
                closest_pose = None
                min_diff = float("inf")

                for pose in pose_data:
                    diff = abs(pose["utime"] - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pose = pose

                if closest_pose:
                    # Translation
                    can_bus[0] = closest_pose.get("pos", [0, 0, 0])[0]
                    can_bus[1] = closest_pose.get("pos", [0, 0, 0])[1]
                    can_bus[2] = closest_pose.get("pos", [0, 0, 0])[2]

                    # Rotation (quaternion to euler yaw)
                    orientation = closest_pose.get("orientation", [1, 0, 0, 0])
                    # Convert quaternion to yaw
                    w, x, y, z = orientation
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    can_bus[16] = yaw  # steering angle proxy
                    can_bus[17] = closest_pose.get("rotation_rate", [0, 0, 0])[2]  # yaw rate

            # Load IMU data
            imu_file = self.canbus_root / scene_name / "ms_imu.json"

            if imu_file.exists():
                with open(imu_file, "r") as f:
                    imu_data = json.load(f)

                sample = self.nusc.get("sample", sample_token)
                timestamp = sample["timestamp"]

                # Find closest IMU reading
                closest_imu = None
                min_diff = float("inf")

                for imu in imu_data:
                    diff = abs(imu["utime"] - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest_imu = imu

                if closest_imu:
                    # Linear acceleration
                    accel = closest_imu.get("linear_accel", [0, 0, 0])
                    can_bus[10:13] = accel[:3]

                    # Angular velocity
                    rot_rate = closest_imu.get("rotation_rate", [0, 0, 0])
                    can_bus[13:16] = rot_rate[:3]

        except Exception as e:
            print(f"⚠ Could not load CAN bus data: {e}")

        return can_bus

    def get_lidar2img(self, sample_token: str) -> np.ndarray:
        """Get lidar to image projection matrices for all cameras.

        Args:
            sample_token: Sample token.

        Returns:
            Array of shape (6, 4, 4) with lidar2img matrices for each camera.
        """
        lidar2img_list = []

        if self.nusc is None:
            # Return identity matrices if nuScenes not available
            return np.tile(np.eye(4), (6, 1, 1))

        try:
            sample = self.nusc.get("sample", sample_token)

            # Get lidar pose
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            lidar_ego = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

            # Lidar to ego
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = self._quat_to_rot(lidar_calib["rotation"])
            lidar2ego[:3, 3] = lidar_calib["translation"]

            for cam_name in self.CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
                cam_ego = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

                # Camera intrinsics
                intrinsic = np.eye(4)
                intrinsic[:3, :3] = np.array(cam_calib["camera_intrinsic"])

                # Camera to ego
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = self._quat_to_rot(cam_calib["rotation"])
                cam2ego[:3, 3] = cam_calib["translation"]
                ego2cam = np.linalg.inv(cam2ego)

                # Compute lidar2img
                lidar2cam = ego2cam @ lidar2ego
                lidar2img = intrinsic @ lidar2cam
                lidar2img_list.append(lidar2img)

        except Exception as e:
            print(f"⚠ Could not compute lidar2img: {e}")
            return np.tile(np.eye(4), (6, 1, 1))

        return np.stack(lidar2img_list, axis=0)

    def _quat_to_rot(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [w, x, y, z].

        Returns:
            3x3 rotation matrix.
        """
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    def load_sample_images(self, sample_token: str) -> List[np.ndarray]:
        """Load all camera images for a sample.

        Args:
            sample_token: Sample token.

        Returns:
            List of 6 images as numpy arrays (H, W, 3).
        """
        images = []

        if self.nusc is not None:
            sample = self.nusc.get("sample", sample_token)

            for cam_name in self.CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                img_path = self.nuscenes_root / cam_data["filename"]

                img = Image.open(img_path).convert("RGB")
                img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                images.append(np.array(img))
        else:
            # Manual mode - try to find images by pattern
            print("⚠ Using manual image loading mode")
            samples_dir = self.nuscenes_root / "samples"

            for cam_name in self.CAMERA_NAMES:
                cam_dir = samples_dir / cam_name
                if cam_dir.exists():
                    # Get first image in directory
                    img_files = list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png"))
                    if img_files:
                        img = Image.open(img_files[0]).convert("RGB")
                        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                        images.append(np.array(img))
                    else:
                        # Create dummy image
                        images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                else:
                    images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))

        return images

    def preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess images for model input.

        Args:
            images: List of images as numpy arrays (H, W, C) in RGB.

        Returns:
            Tensor of shape (1, num_cams, 3, H, W).
        """
        processed = []
        for img in images:
            # Normalize
            img = img.astype(np.float32)
            img = (img - np.array(self.mean)) / np.array(self.std)
            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            processed.append(img)

        # Stack cameras: (num_cams, 3, H, W)
        imgs = np.stack(processed, axis=0)
        # Add batch dimension: (1, num_cams, 3, H, W)
        imgs = imgs[np.newaxis, ...]
        return torch.from_numpy(imgs).float()

    def create_img_metas(
        self,
        sample_token: str = None,
        scene_token: str = None,
    ) -> List[Dict]:
        """Create image metadata for inference.

        Args:
            sample_token: Sample token (optional).
            scene_token: Scene token (optional).

        Returns:
            List of metadata dictionaries.
        """
        # Get calibration
        if sample_token and self.nusc:
            lidar2img = self.get_lidar2img(sample_token)
            can_bus = self.load_can_bus(scene_token, sample_token) if scene_token else np.zeros(18)
        else:
            lidar2img = np.tile(np.eye(4), (6, 1, 1))
            can_bus = np.zeros(18)

        meta = {
            "scene_token": scene_token or "inference",
            "sample_token": sample_token or "inference",
            "can_bus": can_bus,
            "lidar2img": lidar2img,
            "img_shape": [(self.img_height, self.img_width)] * 6,
            "prev_bev_exists": False,
        }

        return [meta]

    def get_sample_list(self) -> List[Dict]:
        """Get list of all samples in dataset.

        Returns:
            List of dicts with 'sample_token' and 'scene_token'.
        """
        samples = []

        if self.nusc is not None:
            for sample in self.nusc.sample:
                samples.append(
                    {
                        "sample_token": sample["token"],
                        "scene_token": sample["scene_token"],
                    }
                )
        else:
            # Manual mode - return empty list
            print("⚠ nuScenes API not available, cannot enumerate samples")

        return samples

    def load_sample(self, sample_token: str, scene_token: str = None) -> Tuple[torch.Tensor, List[Dict]]:
        """Load a complete sample for inference.

        Args:
            sample_token: Sample token.
            scene_token: Scene token (optional, will be looked up if not provided).

        Returns:
            Tuple of (images_tensor, img_metas).
        """
        # Get scene token if not provided
        if scene_token is None and self.nusc:
            sample = self.nusc.get("sample", sample_token)
            scene_token = sample["scene_token"]

        # Load and preprocess images
        images = self.load_sample_images(sample_token)
        images_tensor = self.preprocess_images(images)

        # Create metadata
        img_metas = self.create_img_metas(sample_token, scene_token)

        return images_tensor, img_metas


# ============================================================================
# Weight Loading
# ============================================================================


def remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap checkpoint keys to match model key names.

    Based on official MapTR checkpoint structure: https://github.com/hustvl/MapTR

    Args:
        state_dict: Original checkpoint state dict.

    Returns:
        Remapped state dict with keys matching the model.
    """
    remapped = {}

    for k, v in state_dict.items():
        new_key = k

        # ============================================================
        # FPN keys - checkpoint uses ModuleList index, our model doesn't
        # ============================================================
        # Checkpoint: img_neck.fpn_convs.0.conv.weight -> Model: img_neck.fpn_convs.conv.weight
        if "img_neck.fpn_convs.0." in new_key:
            new_key = new_key.replace("img_neck.fpn_convs.0.", "img_neck.fpn_convs.")
        if "img_neck.lateral_convs.0." in new_key:
            new_key = new_key.replace("img_neck.lateral_convs.0.", "img_neck.lateral_convs.")

        # ============================================================
        # Encoder keys - BEVFormerLayer structure
        # ============================================================
        # The encoder uses: attentions[0]=TemporalSelfAttention, attentions[1]=SpatialCrossAttention
        # Checkpoint has: layers.X.attentions.0.* (temporal) and layers.X.attentions.1.* (spatial)

        # Encoder FFN: checkpoint uses ffns.0.layers.0.0/1 -> our FFN uses layers.0/2
        if "encoder.layers" in new_key and "ffns.0.layers" in new_key:
            new_key = new_key.replace("ffns.0.layers.0.0.", "ffns.0.layers.0.")
            new_key = new_key.replace("ffns.0.layers.1.", "ffns.0.layers.2.")

        # ============================================================
        # Decoder keys - deformable attention structure
        # ============================================================
        # Checkpoint decoder: decoder.layers.X.attentions.0.* (self attn - done first in MapTR)
        #                     decoder.layers.X.attentions.1.* (cross attn)
        # Our decoder: decoder.layers.X.self_attn.* and decoder.layers.X.cross_attn.*

        if "decoder.layers" in new_key:
            # Self attention (attentions.0.attn.* in checkpoint) -> self_attn.attn.*
            # Standard MultiheadAttention with in_proj_*, out_proj.*
            if ".attentions.0." in new_key:
                new_key = new_key.replace(".attentions.0.", ".self_attn.")
            # Cross attention (attentions.1.* in checkpoint) -> cross_attn.*
            # Deformable attention with sampling_offsets, attention_weights, etc.
            elif ".attentions.1." in new_key:
                new_key = new_key.replace(".attentions.1.", ".cross_attn.")

            # FFN structure: ffns.0.layers.0.0 -> ffn.0, ffns.0.layers.1 -> ffn.2
            if "ffns.0.layers.0.0." in new_key:
                new_key = new_key.replace("ffns.0.layers.0.0.", "ffn.0.")
            elif "ffns.0.layers.1." in new_key:
                new_key = new_key.replace("ffns.0.layers.1.", "ffn.2.")

            # Norms
            if ".norms.0." in new_key:
                new_key = new_key.replace(".norms.0.", ".norm1.")
            elif ".norms.1." in new_key:
                new_key = new_key.replace(".norms.1.", ".norm2.")
            elif ".norms.2." in new_key:
                new_key = new_key.replace(".norms.2.", ".norm3.")

        # ============================================================
        # Transformer-level keys
        # ============================================================
        # CAN bus MLP norm
        if "transformer.can_bus_mlp.norm." in new_key:
            new_key = new_key.replace("transformer.can_bus_mlp.norm.", "transformer.can_bus_mlp.4.")

        remapped[new_key] = v

    return remapped


def load_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to strictly enforce weight matching.

    Returns:
        Model with loaded weights.
    """
    print(f"Loading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    # Remap checkpoint keys to match model structure
    new_state_dict = remap_checkpoint_keys(new_state_dict)

    # Check for size mismatches and filter them out if not strict
    model_state = model.state_dict()
    filtered_state_dict = {}
    size_mismatches = []

    for k, v in new_state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                size_mismatches.append(f"  {k}: checkpoint {v.shape} vs model {model_state[k].shape}")
                if not strict:
                    continue  # Skip mismatched weights
        filtered_state_dict[k] = v

    if size_mismatches:
        print(f"\n⚠ Size mismatches found ({len(size_mismatches)}):")
        for msg in size_mismatches[:10]:
            print(msg)
        if len(size_mismatches) > 10:
            print(f"  ... and {len(size_mismatches) - 10} more")
        print("\nHINT: Make sure to use the correct config for your checkpoint:")
        print("  --config maptr_tiny   for maptr_tiny_r50_24e.pth (BEV: 200x100)")
        print("  --config nuScenes     for full nuScenes config (BEV: 200x100)")
        print("  --config small        for testing without weights (BEV: 50x50)")

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        print(f"\nMissing keys ({len(missing_keys)}) - Keys expected by MODEL but NOT in checkpoint:")
        print("-" * 80)
        for key in sorted(missing_keys):
            print(f"  - {key}")

    if unexpected_keys:
        print(f"\nUnexpected keys ({len(unexpected_keys)}) - Keys in CHECKPOINT but NOT in model:")
        print("-" * 80)
        for key in sorted(unexpected_keys):
            print(f"  - {key}")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("WEIGHT KEY COMPARISON SUMMARY")
    print("=" * 80)
    model_keys = set(model_state.keys())
    checkpoint_keys = set(filtered_state_dict.keys())
    matched_keys = model_keys & checkpoint_keys
    print(f"Model keys:       {len(model_keys)}")
    print(f"Checkpoint keys:  {len(checkpoint_keys)}")
    print(f"Matched keys:     {len(matched_keys)}")
    print(f"Missing keys:     {len(missing_keys)} (model expects, checkpoint missing)")
    print(f"Unexpected keys:  {len(unexpected_keys)} (checkpoint has, model missing)")

    print("\n✓ Weights loaded successfully!")
    return model


def convert_mmcv_checkpoint(checkpoint_path: str, output_path: str):
    """Convert MMCV checkpoint to pure PyTorch format.

    Args:
        checkpoint_path: Path to MMCV checkpoint.
        output_path: Path to save converted checkpoint.
    """
    print(f"Converting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Key mapping from MMCV to PyTorch
    key_mapping = {
        # Backbone
        "img_backbone.": "img_backbone.",
        # Neck
        "img_neck.": "img_neck.",
        # Head
        "pts_bbox_head.": "pts_bbox_head.",
    }

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in key_mapping.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix) :]
                break
        new_state_dict[new_key] = v

    torch.save({"state_dict": new_state_dict}, output_path)
    print(f"Converted checkpoint saved to: {output_path}")


# ============================================================================
# Inference Pipeline
# ============================================================================


class MapTRInference:
    """MapTR inference pipeline."""

    def __init__(
        self,
        config: MapTRConfig,
        checkpoint_path: Optional[str] = None,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Build model
        print("Building model...")
        self.model = build_maptr_model(config, self.device)

        # Load weights if provided
        if checkpoint_path is not None:
            self.model = load_weights(self.model, checkpoint_path, strict=False)

        self.model.eval()

        # Initialize processors
        self.image_processor = ImageProcessor(
            img_height=config.img_height,
            img_width=config.img_width,
        )
        self.calibration = CameraCalibration(num_cameras=config.num_cameras)

        # Class names
        self.class_names = ["divider", "ped_crossing", "boundary"]

    def create_img_metas(
        self,
        calibration: Dict[str, np.ndarray],
        scene_token: str = "inference",
    ) -> List[List[Dict]]:
        """Create image metadata for inference."""
        meta = {
            "scene_token": scene_token,
            "can_bus": np.zeros(18),
            "lidar2img": calibration["lidar2img"],
            "camera2ego": calibration.get("camera2ego", np.eye(4)),
            "camera_intrinsics": calibration.get("camera_intrinsics", np.eye(4)),
            "img_aug_matrix": calibration.get("img_aug_matrix", np.eye(4)),
            "lidar2ego": calibration.get("lidar2ego", np.eye(4)),
            "img_shape": [(self.config.img_height, self.config.img_width)] * self.config.num_cameras,
            "prev_bev_exists": False,
        }
        return [[meta]]

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        img_metas: List[List[Dict]],
    ) -> List[Dict]:
        """Run inference on images.

        Args:
            images: Input images tensor of shape (1, num_cams, 3, H, W).
            img_metas: Image metadata.

        Returns:
            List of detection results.
        """
        images = images.to(self.device)
        results = self.model(img_metas=img_metas, img=images)
        return results

    def predict_from_paths(
        self,
        image_paths: List[str],
        calibration: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """Run inference on images from file paths.

        Args:
            image_paths: List of paths to camera images.
            calibration: Camera calibration (uses dummy if None).

        Returns:
            List of detection results.
        """
        # Load and preprocess images
        images = [self.image_processor.load_image(p) for p in image_paths]
        images_tensor = self.image_processor.preprocess(images)

        # Use dummy calibration if not provided
        if calibration is None:
            calibration = self.calibration.create_dummy_calibration(self.config.img_height, self.config.img_width)

        img_metas = self.create_img_metas(calibration)

        return self.predict(images_tensor, img_metas)

    def predict_demo(self) -> List[Dict]:
        """Run inference on dummy data for demo/testing."""
        images = self.image_processor.generate_dummy_images(self.config.num_cameras)
        calibration = self.calibration.create_dummy_calibration(self.config.img_height, self.config.img_width)
        img_metas = self.create_img_metas(calibration)
        return self.predict(images, img_metas)

    def format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Format results for output.

        Args:
            results: Raw model output.

        Returns:
            Formatted results dictionary.
        """
        formatted = {
            "num_detections": 0,
            "detections": [],
        }

        for i, result in enumerate(results):
            if "pts_bbox" not in result:
                continue

            pts_bbox = result["pts_bbox"]
            bboxes = pts_bbox["boxes_3d"]
            scores = pts_bbox["scores_3d"]
            labels = pts_bbox["labels_3d"]
            pts = pts_bbox["pts_3d"]

            num_dets = bboxes.shape[0]
            formatted["num_detections"] += num_dets

            for j in range(num_dets):
                det = {
                    "sample_idx": i,
                    "class": self.class_names[labels[j].item()]
                    if labels[j].item() < len(self.class_names)
                    else f"class_{labels[j].item()}",
                    "class_id": labels[j].item(),
                    "score": scores[j].item(),
                    "bbox": bboxes[j].cpu().numpy().tolist(),
                    "points": pts[j].cpu().numpy().tolist(),
                }
                formatted["detections"].append(det)

        return formatted

    def print_results(self, results: List[Dict]):
        """Print detection results."""
        formatted = self.format_results(results)

        print("\n" + "=" * 60)
        print("Detection Results")
        print("=" * 60)
        print(f"Total detections: {formatted['num_detections']}")

        for det in formatted["detections"][:10]:  # Print first 10
            print(f"\n  Class: {det['class']}")
            print(f"  Score: {det['score']:.4f}")
            print(f"  BBox: {det['bbox']}")
            print(f"  Points: {len(det['points'])} vertices")

        if len(formatted["detections"]) > 10:
            print(f"\n  ... and {len(formatted['detections']) - 10} more detections")

    def visualize_hd_map(
        self,
        results: List[Dict],
        output_path: str = None,
        score_threshold: float = 0.0,
        show: bool = False,
    ):
        """Visualize HD map output as polylines in BEV view.

        Args:
            results: Detection results from predict().
            output_path: Path to save the visualization.
            score_threshold: Minimum score to display.
            show: Whether to display the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        formatted = self.format_results(results)

        # Color map for different classes
        colors = {
            "divider": "#FF6B6B",  # Red
            "ped_crossing": "#4ECDC4",  # Teal
            "boundary": "#45B7D1",  # Blue
        }

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Set BEV range from pc_range
        x_min, y_min = self.config.pc_range[0], self.config.pc_range[1]
        x_max, y_max = self.config.pc_range[3], self.config.pc_range[4]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        # Draw grid
        ax.grid(True, alpha=0.3, color="white", linestyle="--")

        # Draw ego vehicle (rectangle at origin)
        ego_width, ego_length = 2.0, 4.5
        ego = plt.Rectangle(
            (-ego_width / 2, -ego_length / 2),
            ego_width,
            ego_length,
            fill=True,
            facecolor="#FFD93D",
            edgecolor="white",
            linewidth=2,
            zorder=10,
        )
        ax.add_patch(ego)

        # Track counts per class
        class_counts = {name: 0 for name in colors.keys()}

        # Draw polylines for each detection
        for det in formatted["detections"]:
            if det["score"] < score_threshold:
                continue

            class_name = det["class"]
            points = np.array(det["points"])

            if len(points) < 2:
                continue

            color = colors.get(class_name, "#FFFFFF")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Draw polyline
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=2.5,
                alpha=0.8,
                marker="o",
                markersize=3,
                markerfacecolor=color,
            )

        # Create legend
        legend_handles = []
        for class_name, color in colors.items():
            count = class_counts.get(class_name, 0)
            if count > 0:
                patch = mpatches.Patch(color=color, label=f"{class_name} ({count})")
                legend_handles.append(patch)

        ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

        # Labels
        ax.set_xlabel("X (meters)", fontsize=12, color="white")
        ax.set_ylabel("Y (meters)", fontsize=12, color="white")
        ax.set_title("MapTR HD Map Output (BEV)", fontsize=14, fontweight="bold", color="white")
        ax.tick_params(colors="white")

        # Style spines
        for spine in ax.spines.values():
            spine.set_color("white")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
            print(f"\n✓ HD Map saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def print_hd_map_polylines(self, results: List[Dict], score_threshold: float = 0.0):
        """Print HD map polylines in a readable format.

        Args:
            results: Detection results from predict().
            score_threshold: Minimum score to display.
        """
        formatted = self.format_results(results)

        print("\n" + "=" * 60)
        print("HD Map Polylines (Vectorized Output)")
        print("=" * 60)

        # Group by class
        by_class = {}
        for det in formatted["detections"]:
            if det["score"] < score_threshold:
                continue
            class_name = det["class"]
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)

        for class_name, detections in by_class.items():
            print(f"\n{'─' * 50}")
            print(f"  {class_name.upper()} ({len(detections)} instances)")
            print(f"{'─' * 50}")

            for i, det in enumerate(detections[:5]):  # Show first 5 per class
                points = np.array(det["points"])
                print(f"\n  [{i+1}] Score: {det['score']:.3f}")
                print(f"      Vertices: {len(points)} points")
                print(f"      Coordinates (x, y):")
                for j, pt in enumerate(points):
                    print(f"        P{j+1}: ({pt[0]:7.2f}, {pt[1]:7.2f})")

            if len(detections) > 5:
                print(f"\n      ... and {len(detections) - 5} more {class_name} instances")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MapTR Inference Pipeline")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Paths to input images (one per camera)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing camera images",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with dummy data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["small", "nuScenes", "maptr_tiny"],
        help="Model configuration preset (use 'maptr_tiny' for maptr_tiny_r50_24e.pth checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to save HD map visualization (e.g., hd_map.png)",
    )
    parser.add_argument(
        "--show_polylines",
        action="store_true",
        help="Print detailed polyline coordinates",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for detections",
    )

    # nuScenes dataset arguments
    parser.add_argument(
        "--nuscenes",
        type=str,
        default=None,
        help="Path to nuScenes data root (containing nuscenes/ and can_bus/ folders)",
    )
    parser.add_argument(
        "--nuscenes_version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="nuScenes dataset version",
    )
    parser.add_argument(
        "--sample_token",
        type=str,
        default=None,
        help="Specific sample token to process (nuScenes mode)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to process (nuScenes mode, default: 0)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to process (nuScenes mode)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config
    if args.config == "nuScenes":
        config = MapTRConfig.from_nuScenes()
    elif args.config == "maptr_tiny":
        config = MapTRConfig.from_maptr_tiny()
    else:
        config = MapTRConfig.from_small()

    print("\n" + "=" * 60)
    print("MapTR Inference Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"BEV size: {config.bev_h} x {config.bev_w}")
    print(f"Num vectors: {config.num_vec}")
    print(f"Points per vector: {config.num_pts_per_vec}")

    # Create inference pipeline
    inference = MapTRInference(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Run inference
    if args.nuscenes:
        # ============================================================
        # nuScenes Dataset Mode (Official MapTR approach)
        # ============================================================
        print(f"\n{'='*60}")
        print("nuScenes Dataset Mode")
        print(f"{'='*60}")
        print(f"Data root: {args.nuscenes}")
        print(f"Version: {args.nuscenes_version}")

        # Initialize nuScenes data loader
        nuscenes_loader = NuScenesDataLoader(
            data_root=args.nuscenes,
            version=args.nuscenes_version,
            img_height=config.img_height,
            img_width=config.img_width,
        )

        # Get samples to process
        if args.sample_token:
            # Process specific sample
            samples_to_process = [{"sample_token": args.sample_token, "scene_token": None}]
        else:
            # Get sample list
            all_samples = nuscenes_loader.get_sample_list()
            if all_samples:
                start_idx = args.sample_idx
                end_idx = min(start_idx + args.num_samples, len(all_samples))
                samples_to_process = all_samples[start_idx:end_idx]
                print(f"Processing samples {start_idx} to {end_idx-1} ({len(samples_to_process)} samples)")
            else:
                print("⚠ No samples found in dataset")
                samples_to_process = []

        # Process each sample
        all_results = []
        for i, sample_info in enumerate(samples_to_process):
            sample_token = sample_info["sample_token"]
            scene_token = sample_info.get("scene_token")

            print(f"\n--- Sample {i+1}/{len(samples_to_process)} ---")
            print(
                f"Sample token: {sample_token[:20]}..." if len(sample_token) > 20 else f"Sample token: {sample_token}"
            )

            # Load sample data with CAN bus
            images_tensor, img_metas = nuscenes_loader.load_sample(sample_token, scene_token)
            images_tensor = images_tensor.to(inference.device)

            # Show CAN bus info
            can_bus = img_metas[0].get("can_bus", np.zeros(18))
            print(f"CAN bus - Position: ({can_bus[0]:.2f}, {can_bus[1]:.2f}, {can_bus[2]:.2f})")
            print(f"CAN bus - Yaw: {can_bus[16]:.4f} rad, Yaw rate: {can_bus[17]:.4f}")

            # Run inference
            results = inference.predict(images_tensor, [img_metas])
            all_results.append(
                {
                    "sample_token": sample_token,
                    "results": results,
                }
            )

            # Print results for this sample
            inference.print_results(results)

            # Visualize if requested
            if args.visualize:
                if len(samples_to_process) == 1:
                    viz_path = args.visualize
                else:
                    # Add sample index to filename
                    viz_base = Path(args.visualize)
                    viz_path = str(viz_base.parent / f"{viz_base.stem}_{i:04d}{viz_base.suffix}")

                inference.visualize_hd_map(
                    results,
                    output_path=viz_path,
                    score_threshold=args.score_threshold,
                )

        # Summary
        print(f"\n{'='*60}")
        print(f"Processed {len(all_results)} samples from nuScenes dataset")
        print(f"{'='*60}")

        # Use last result for final output
        results = all_results[-1]["results"] if all_results else []

    elif args.demo:
        print("\nRunning demo inference with dummy data...")
        results = inference.predict_demo()

    elif args.images:
        print(f"\nRunning inference on {len(args.images)} images...")
        calibration = None
        if args.calibration:
            calibration = inference.calibration.load_calibration(args.calibration)
        results = inference.predict_from_paths(args.images, calibration)

    elif args.image_dir:
        # Load all images from directory
        image_dir = Path(args.image_dir)
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(sorted(image_dir.glob(f"*{ext}")))
            image_paths.extend(sorted(image_dir.glob(f"*{ext.upper()}")))

        if len(image_paths) < config.num_cameras:
            print(f"Warning: Found only {len(image_paths)} images, expected {config.num_cameras}")

        image_paths = [str(p) for p in image_paths[: config.num_cameras]]
        print(f"\nRunning inference on images from {args.image_dir}...")
        calibration = None
        if args.calibration:
            calibration = inference.calibration.load_calibration(args.calibration)
        results = inference.predict_from_paths(image_paths, calibration)

    else:
        print("\nNo input provided, running demo mode...")
        results = inference.predict_demo()

    # Print results (skip if already printed in nuScenes mode)
    if not args.nuscenes:
        inference.print_results(results)

    # Print detailed polyline coordinates if requested
    if args.show_polylines:
        inference.print_hd_map_polylines(results, score_threshold=args.score_threshold)

    # Visualize HD map if path provided (skip if already done in nuScenes mode)
    if args.visualize and not args.nuscenes:
        inference.visualize_hd_map(
            results,
            output_path=args.visualize,
            score_threshold=args.score_threshold,
        )

    # Save results if output path provided
    if args.output:
        formatted = inference.format_results(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
