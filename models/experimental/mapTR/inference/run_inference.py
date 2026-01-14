# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Inference and Visualization Pipeline

Based on official MapTR repository: https://github.com/hustvl/MapTR
Visualization code adapted from: https://github.com/hustvl/MapTR/blob/main/tools/maptr/vis_pred.py

This script runs the complete MapTR inference pipeline with:
- Image loading and preprocessing
- Model weight loading
- Multi-camera inference
- Result visualization (BEV map, surround view cameras)

Usage:
    # Demo mode (generates dummy data):
    python models/experimental/mapTR/inference/run_inference.py --demo

    # With checkpoint and images:
    python models/experimental/mapTR/inference/run_inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --image_dir /path/to/images/ \
        --show-dir /path/to/output/

    # With nuScenes dataset:
    python models/experimental/mapTR/inference/run_inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --nuscenes /path/to/nuscenes/ \
        --show-dir /path/to/output/ \
        --score-thresh 0.4
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
import os
import os.path as osp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid.

    Args:
        x: Tensor with values in [0, 1]
        eps: Small value to prevent numerical instability

    Returns:
        Tensor with values in (-inf, inf)
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


# Import MapTR components
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_maptr import MapTR
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.pytorch_temporal_self_attention import multi_scale_deformable_attn_pytorch


# ============================================================================
# Constants (from official vis_pred.py)
# ============================================================================

# Camera names in nuScenes order (matching official MapTR)
CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# Candidate samples for visualization (challenging samples from nuScenes)
# From: https://github.com/hustvl/MapTR/blob/main/tools/maptr/vis_pred.py
CANDIDATE = [
    "n008-2018-08-01-15-16-36-0400_1533151184047036",
    "n008-2018-08-01-15-16-36-0400_1533151200646853",
    "n008-2018-08-01-15-16-36-0400_1533151274047332",
    "n008-2018-08-01-15-16-36-0400_1533151369947807",
    "n008-2018-08-01-15-16-36-0400_1533151581047647",
    "n008-2018-08-01-15-16-36-0400_1533151585447531",
    "n008-2018-08-01-15-16-36-0400_1533151741547700",
    "n008-2018-08-01-15-16-36-0400_1533151854947676",
    "n008-2018-08-22-15-53-49-0400_1534968048946931",
    "n008-2018-08-22-15-53-49-0400_1534968255947662",
    "n008-2018-08-01-15-16-36-0400_1533151616447606",
    "n015-2018-07-18-11-41-49+0800_1531885617949602",
    "n008-2018-08-28-16-43-51-0400_1535489136547616",
    "n008-2018-08-28-16-43-51-0400_1535489145446939",
    "n008-2018-08-28-16-43-51-0400_1535489152948944",
    "n008-2018-08-28-16-43-51-0400_1535489299547057",
    "n008-2018-08-28-16-43-51-0400_1535489317946828",
    "n008-2018-09-18-15-12-01-0400_1537298038950431",
    "n008-2018-09-18-15-12-01-0400_1537298047650680",
    "n008-2018-09-18-15-12-01-0400_1537298056450495",
    "n008-2018-09-18-15-12-01-0400_1537298074700410",
    "n008-2018-09-18-15-12-01-0400_1537298088148941",
    "n008-2018-09-18-15-12-01-0400_1537298101700395",
    "n015-2018-11-21-19-21-35+0800_1542799330198603",
    "n015-2018-11-21-19-21-35+0800_1542799345696426",
    "n015-2018-11-21-19-21-35+0800_1542799353697765",
    "n015-2018-11-21-19-21-35+0800_1542799525447813",
    "n015-2018-11-21-19-21-35+0800_1542799676697935",
    "n015-2018-11-21-19-21-35+0800_1542799758948001",
]

# Colors for map elements (matching official MapTR visualization)
# Class order: divider (0), ped_crossing (1), boundary (2)
COLORS_PLT = {
    0: "orange",  # divider
    1: "blue",  # ped_crossing
    2: "green",  # boundary
}

CLASS_NAMES = ["divider", "ped_crossing", "boundary"]


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
            img_height=448,  # Official: 900*0.5=450, padded to divisor 32 = 448
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

        # BEVFormer Encoder
        self.encoder = BEVFormerEncoder(
            num_layers=num_encoder_layers,
            pc_range=self.pc_range,
            num_points_in_pillar=num_points_in_pillar,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=512,
        )

        # Decoder
        self.decoder = MapTRDecoder(
            embed_dims=embed_dims,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            feedforward_channels=512,
            num_points=4,
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
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.cams_embeds is not None:
                feat = feat + self.cams_embeds[:num_cam, None, None, :].to(feat.dtype)
            if lvl < len(self.level_embeds):
                feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

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

        # Prepare decoder queries - CRITICAL: query_pos is FIRST half, query is SECOND half
        query_pos, query = torch.split(object_query_embeds, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        # Reference points computed from query_pos
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # Permute to (seq_len, batch, embed_dims) format for decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # Spatial shapes for decoder attention
        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)

        # Run decoder with reg_branches for iterative refinement
        hs, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        # bev_embed is already in (seq_len, batch, embed_dims) format
        return bev_embed, hs, init_reference_out, inter_references

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


class CustomMSDeformableAttention(nn.Module):
    """Cross attention using deformable attention for decoder."""

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


class SelfAttnWrapper(nn.Module):
    """Wrapper for MultiheadAttention to match checkpoint key structure."""

    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)

    def forward(self, query, key, value):
        return self.attn(query, key, value)[0]


class MapTRDecoderLayer(nn.Module):
    """MapTR Decoder layer matching checkpoint structure."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        num_points: int = 4,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        self.self_attn = SelfAttnWrapper(embed_dims, num_heads)
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_channels, embed_dims),
        )

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
        # Input is sequence-first (num_query, bs, embed_dims)
        # Self-attention uses sequence-first format
        q = k = query + query_pos if query_pos is not None else query
        query2 = self.self_attn(q, k, query)
        query = self.norm1(query + query2)

        # Cross-attention expects batch-first (bs, num_query, embed_dims)
        query_bf = query.permute(1, 0, 2)  # (num_q, bs, dim) -> (bs, num_q, dim)
        value_bf = value.permute(1, 0, 2)  # (num_v, bs, dim) -> (bs, num_v, dim)
        query_pos_bf = query_pos.permute(1, 0, 2) if query_pos is not None else None

        # reference_points: (bs, num_query, num_levels, 2)
        query_bf = self.cross_attn(
            query=query_bf,
            key=value_bf,
            value=value_bf,
            query_pos=query_pos_bf,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        # Convert back to sequence-first
        query = query_bf.permute(1, 0, 2)
        query = self.norm2(query)

        query = query + self.ffn(query)
        query = self.norm3(query)

        return query


class MapTRDecoder(nn.Module):
    """MapTR Decoder with iterative refinement."""

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
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)

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

            # Permute for reg_branches: (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
            output = output.permute(1, 0, 2)

            # CRITICAL: Iterative refinement - update reference points using reg_branches
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # Permute back: (bs, num_query, embed_dims) -> (num_query, bs, embed_dims)
            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                # Official decoder stores output in (num_query, bs, embed_dims) format
                # Head will permute it to (bs, num_query, embed_dims)
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def build_maptr_model(config: MapTRConfig, device: torch.device = None) -> MapTR:
    """Build MapTR model from config."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.backbone_depth == 50:
        layers = [3, 4, 6, 3]
    elif config.backbone_depth == 101:
        layers = [3, 4, 23, 3]
    else:
        layers = [3, 4, 6, 3]

    backbone = ResNet(
        block=Bottleneck,
        layers=layers,
        out_indices=(3,),
    )

    fpn = FPN(
        in_channels=[2048],
        out_channels=config.fpn_out_channels,
        num_outs=1,
    )

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

    pos_enc = LearnedPositionalEncoding(
        num_feats=config.embed_dims // 2,
        row_num_embed=config.bev_h,
        col_num_embed=config.bev_w,
    )

    head = MapTRHead(
        transformer=transformer,
        positional_encoding=pos_enc,
        embed_dims=config.embed_dims,
        num_classes=config.num_classes,
        num_reg_fcs=2,
        code_size=2,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        query_embed_type="instance_pts",
        transform_method="minmax",
    )

    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    return model.to(device)


# ============================================================================
# Visualization (adapted from official vis_pred.py)
# ============================================================================


class MapTRVisualizer:
    """Visualizer for MapTR predictions.

    Based on: https://github.com/hustvl/MapTR/blob/main/tools/maptr/vis_pred.py
    """

    def __init__(self, pc_range: List[float], car_img_path: str = None):
        """Initialize visualizer.

        Args:
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
            car_img_path: Path to car icon image for BEV visualization.
        """
        self.pc_range = pc_range
        self.car_img = None

        if car_img_path and osp.exists(car_img_path):
            self.car_img = Image.open(car_img_path)
        else:
            # Create a simple car icon
            self.car_img = self._create_car_icon()

    def _create_car_icon(self) -> np.ndarray:
        """Create a simple car icon for BEV visualization."""
        # Create 60x30 car icon
        car = np.ones((60, 30, 4), dtype=np.uint8) * 255
        # Body (gray)
        car[5:55, 3:27, :3] = [100, 100, 100]
        # Windshield (blue)
        car[8:20, 5:25, :3] = [50, 50, 150]
        # Rear window
        car[40:50, 5:25, :3] = [50, 50, 150]
        return car

    def create_surround_view(
        self,
        cam_images: Dict[str, np.ndarray],
        output_path: str,
        jpeg_quality: int = 70,
    ):
        """Create surround view by concatenating camera images.

        Args:
            cam_images: Dictionary mapping camera name to image array.
            output_path: Path to save the surround view image.
            jpeg_quality: JPEG compression quality.
        """
        try:
            import cv2
        except ImportError:
            print("cv2 not available, skipping surround view creation")
            return

        # Row 1: Front left, Front, Front right
        row_1_list = []
        for cam in CAMS[:3]:
            if cam in cam_images:
                row_1_list.append(cam_images[cam])
            else:
                # Create placeholder
                row_1_list.append(np.zeros((480, 800, 3), dtype=np.uint8))

        # Row 2: Back left, Back, Back right
        row_2_list = []
        for cam in CAMS[3:]:
            if cam in cam_images:
                row_2_list.append(cam_images[cam])
            else:
                row_2_list.append(np.zeros((480, 800, 3), dtype=np.uint8))

        if row_1_list and row_2_list:
            row_1_img = cv2.hconcat(row_1_list)
            row_2_img = cv2.hconcat(row_2_list)
            cams_img = cv2.vconcat([row_1_img, row_2_img])
            cv2.imwrite(output_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            print(f"✓ Surround view saved: {output_path}")

    def visualize_predictions(
        self,
        results: Dict,
        output_path: str,
        score_thresh: float = 0.4,
        vis_format: str = "fixed_num_pts",
        dpi: int = 1200,
    ):
        """Visualize model predictions in BEV.

        Based on official vis_pred.py visualization.

        Args:
            results: Detection results dictionary with boxes_3d, scores_3d, labels_3d, pts_3d.
            output_path: Path to save the visualization.
            score_thresh: Score threshold for filtering predictions.
            vis_format: Visualization format ('fixed_num_pts', 'bbox', 'polyline_pts').
            dpi: DPI for output image.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        pc_range = self.pc_range

        # Create figure (2:4 aspect ratio like official code)
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis("off")

        # Extract predictions
        boxes_3d = results.get("boxes_3d", results.get("bboxes", None))
        scores_3d = results.get("scores_3d", results.get("scores", None))
        labels_3d = results.get("labels_3d", results.get("labels", None))
        pts_3d = results.get("pts_3d", results.get("pts", None))

        if boxes_3d is None or scores_3d is None:
            print("No predictions to visualize")
            plt.close()
            return

        # Convert to numpy if tensors
        if torch.is_tensor(boxes_3d):
            boxes_3d = boxes_3d.cpu().numpy()
        if torch.is_tensor(scores_3d):
            scores_3d = scores_3d.cpu().numpy()
        if torch.is_tensor(labels_3d):
            labels_3d = labels_3d.cpu().numpy()
        if pts_3d is not None and torch.is_tensor(pts_3d):
            pts_3d = pts_3d.cpu().numpy()

        # Apply score threshold
        keep = scores_3d > score_thresh

        # Draw predictions
        for i, (score, bbox, label) in enumerate(zip(scores_3d[keep], boxes_3d[keep], labels_3d[keep])):
            color = COLORS_PLT.get(int(label), "white")

            if vis_format in ["fixed_num_pts", "polyline_pts"] and pts_3d is not None:
                # Draw polyline with points
                pts = pts_3d[keep][i] if len(pts_3d[keep]) > i else None
                if pts is not None:
                    pts_x = pts[:, 0]
                    pts_y = pts[:, 1]
                    plt.plot(pts_x, pts_y, color=color, linewidth=1, alpha=0.8, zorder=-1)
                    plt.scatter(pts_x, pts_y, color=color, s=1, alpha=0.8, zorder=-1)

            if vis_format == "bbox":
                # Draw bounding box
                xy = (bbox[0], bbox[1])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                plt.gca().add_patch(Rectangle(xy, width, height, linewidth=0.4, edgecolor=color, facecolor="none"))

        # Add car icon at center
        if self.car_img is not None:
            try:
                plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
            except Exception:
                pass

        # Save figure
        plt.savefig(output_path, bbox_inches="tight", format="png", dpi=dpi)
        plt.close()
        print(f"✓ Prediction map saved: {output_path}")

    def visualize_with_legend(
        self,
        results: Dict,
        output_path: str,
        score_thresh: float = 0.4,
        title: str = "MapTR HD Map Prediction",
    ):
        """Visualize predictions with legend and labels.

        Args:
            results: Detection results dictionary.
            output_path: Path to save the visualization.
            score_thresh: Score threshold for filtering predictions.
            title: Title for the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not installed")
            return

        pc_range = self.pc_range

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 16))
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        ax.grid(True, alpha=0.3, linestyle="--")

        # Extract predictions
        boxes_3d = results.get("boxes_3d", results.get("bboxes", None))
        scores_3d = results.get("scores_3d", results.get("scores", None))
        labels_3d = results.get("labels_3d", results.get("labels", None))
        pts_3d = results.get("pts_3d", results.get("pts", None))

        if boxes_3d is None:
            plt.close()
            return

        # Convert to numpy
        if torch.is_tensor(boxes_3d):
            boxes_3d = boxes_3d.cpu().numpy()
        if torch.is_tensor(scores_3d):
            scores_3d = scores_3d.cpu().numpy()
        if torch.is_tensor(labels_3d):
            labels_3d = labels_3d.cpu().numpy()
        if pts_3d is not None and torch.is_tensor(pts_3d):
            pts_3d = pts_3d.cpu().numpy()

        keep = scores_3d > score_thresh
        class_counts = {0: 0, 1: 0, 2: 0}

        # Draw predictions
        for i, (score, bbox, label) in enumerate(zip(scores_3d[keep], boxes_3d[keep], labels_3d[keep])):
            label_int = int(label)
            color = COLORS_PLT.get(label_int, "gray")
            class_counts[label_int] = class_counts.get(label_int, 0) + 1

            if pts_3d is not None and len(pts_3d[keep]) > i:
                pts = pts_3d[keep][i]
                pts_x = pts[:, 0]
                pts_y = pts[:, 1]
                ax.plot(pts_x, pts_y, color=color, linewidth=2, alpha=0.8)
                ax.scatter(pts_x, pts_y, color=color, s=8, alpha=0.8)

        # Draw ego vehicle
        ego_rect = plt.Rectangle(
            (-1, -2.25), 2, 4.5, fill=True, facecolor="yellow", edgecolor="black", linewidth=2, zorder=10
        )
        ax.add_patch(ego_rect)

        # Create legend
        legend_handles = []
        for label_id, color in COLORS_PLT.items():
            if label_id < len(CLASS_NAMES):
                count = class_counts.get(label_id, 0)
                patch = mpatches.Patch(color=color, label=f"{CLASS_NAMES[label_id]} ({count})")
                legend_handles.append(patch)

        ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Visualization with legend saved: {output_path}")


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
        """Preprocess images for model input."""
        processed = []
        for img in images:
            img = img.astype(np.float32)
            img = (img - np.array(self.mean)) / np.array(self.std)
            img = img.transpose(2, 0, 1)
            processed.append(img)

        imgs = np.stack(processed, axis=0)
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
        """Create dummy camera calibration."""
        lidar2img = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera2ego = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera_intrinsics = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        img_aug_matrix = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        lidar2ego = np.eye(4)

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
# nuScenes Data Loader
# ============================================================================


class NuScenesDataLoader:
    """nuScenes data loader following official MapTR approach."""

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
        self.data_root = Path(data_root)
        self.version = version
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean or [123.675, 116.28, 103.53]
        self.std = std or [58.395, 57.12, 57.375]

        self.nuscenes_root = self.data_root / "nuscenes"
        self.canbus_root = self.data_root / "can_bus"

        self.nusc = None
        self._init_nuscenes()

    def _init_nuscenes(self):
        """Initialize nuScenes devkit."""
        try:
            from nuscenes.nuscenes import NuScenes

            nuscenes_dataroot = str(self.nuscenes_root)
            if (self.nuscenes_root / self.version).exists():
                self.nusc = NuScenes(
                    version=self.version,
                    dataroot=nuscenes_dataroot,
                    verbose=True,
                )
                print(f"✓ Loaded nuScenes {self.version} with {len(self.nusc.sample)} samples")
            else:
                print(f"⚠ nuScenes metadata not found at {self.nuscenes_root / self.version}")
        except ImportError:
            print("⚠ nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
        except Exception as e:
            print(f"⚠ Could not initialize nuScenes: {e}")

    def load_can_bus(self, scene_token: str, sample_token: str) -> np.ndarray:
        """Load CAN bus data for a sample."""
        can_bus = np.zeros(18)
        if self.nusc is None:
            return can_bus

        try:
            scene = self.nusc.get("scene", scene_token)
            scene_name = scene["name"]
            # Try both path formats: scene-0001_pose.json and scene-0001/pose.json
            pose_file = self.canbus_root / f"{scene_name}_pose.json"
            if not pose_file.exists():
                pose_file = self.canbus_root / scene_name / "pose.json"

            if pose_file.exists():
                with open(pose_file, "r") as f:
                    pose_data = json.load(f)

                sample = self.nusc.get("sample", sample_token)
                timestamp = sample["timestamp"]

                closest_pose = None
                min_diff = float("inf")
                for pose in pose_data:
                    diff = abs(pose["utime"] - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pose = pose

                if closest_pose:
                    can_bus[0] = closest_pose.get("pos", [0, 0, 0])[0]
                    can_bus[1] = closest_pose.get("pos", [0, 0, 0])[1]
                    can_bus[2] = closest_pose.get("pos", [0, 0, 0])[2]

                    orientation = closest_pose.get("orientation", [1, 0, 0, 0])
                    w, x, y, z = orientation
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    can_bus[16] = yaw
                    can_bus[17] = closest_pose.get("rotation_rate", [0, 0, 0])[2]

        except Exception as e:
            print(f"⚠ Could not load CAN bus data: {e}")

        return can_bus

    def get_lidar2img(self, sample_token: str) -> np.ndarray:
        """Get lidar to image projection matrices."""
        lidar2img_list = []

        if self.nusc is None:
            return np.tile(np.eye(4), (6, 1, 1))

        try:
            sample = self.nusc.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = self._quat_to_rot(lidar_calib["rotation"])
            lidar2ego[:3, 3] = lidar_calib["translation"]

            for cam_name in self.CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

                intrinsic = np.eye(4)
                cam_intrinsic = np.array(cam_calib["camera_intrinsic"])

                # Scale intrinsics for resized images
                # Original nuScenes image size is 1600x900
                original_w, original_h = 1600, 900
                scale_w = self.img_width / original_w
                scale_h = self.img_height / original_h

                # Scale fx, cx by width ratio; fy, cy by height ratio
                cam_intrinsic[0, :] *= scale_w  # fx, 0, cx
                cam_intrinsic[1, :] *= scale_h  # 0, fy, cy

                intrinsic[:3, :3] = cam_intrinsic

                cam2ego = np.eye(4)
                cam2ego[:3, :3] = self._quat_to_rot(cam_calib["rotation"])
                cam2ego[:3, 3] = cam_calib["translation"]
                ego2cam = np.linalg.inv(cam2ego)

                lidar2cam = ego2cam @ lidar2ego
                lidar2img = intrinsic @ lidar2cam
                lidar2img_list.append(lidar2img)

        except Exception as e:
            print(f"⚠ Could not compute lidar2img: {e}")
            return np.tile(np.eye(4), (6, 1, 1))

        return np.stack(lidar2img_list, axis=0)

    def _quat_to_rot(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    def load_sample_images(self, sample_token: str) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """Load all camera images for a sample.

        Returns:
            Tuple of (list of images, dict mapping cam name to image)
        """
        images = []
        cam_images = {}

        if self.nusc is not None:
            sample = self.nusc.get("sample", sample_token)

            for cam_name in self.CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                img_path = self.nuscenes_root / cam_data["filename"]

                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                img_array = np.array(img_resized)
                images.append(img_array)

                # Also store BGR for OpenCV visualization
                cam_images[cam_name] = img_array[:, :, ::-1].copy()
        else:
            samples_dir = self.nuscenes_root / "samples"
            for cam_name in self.CAMERA_NAMES:
                cam_dir = samples_dir / cam_name
                if cam_dir.exists():
                    img_files = list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png"))
                    if img_files:
                        img = Image.open(img_files[0]).convert("RGB")
                        img_resized = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                        img_array = np.array(img_resized)
                        images.append(img_array)
                        cam_images[cam_name] = img_array[:, :, ::-1].copy()
                    else:
                        images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                else:
                    images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))

        return images, cam_images

    def preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess images for model input."""
        processed = []
        for img in images:
            img = img.astype(np.float32)
            img = (img - np.array(self.mean)) / np.array(self.std)
            img = img.transpose(2, 0, 1)
            processed.append(img)

        imgs = np.stack(processed, axis=0)
        imgs = imgs[np.newaxis, ...]
        return torch.from_numpy(imgs).float()

    def create_img_metas(self, sample_token: str = None, scene_token: str = None) -> List[Dict]:
        """Create image metadata for inference."""
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
        """Get list of all samples in dataset."""
        samples = []
        if self.nusc is not None:
            for sample in self.nusc.sample:
                samples.append(
                    {
                        "sample_token": sample["token"],
                        "scene_token": sample["scene_token"],
                    }
                )
        return samples

    def load_sample(
        self, sample_token: str, scene_token: str = None
    ) -> Tuple[torch.Tensor, List[Dict], Dict[str, np.ndarray]]:
        """Load a complete sample for inference."""
        if scene_token is None and self.nusc:
            sample = self.nusc.get("sample", sample_token)
            scene_token = sample["scene_token"]

        images, cam_images = self.load_sample_images(sample_token)
        images_tensor = self.preprocess_images(images)
        img_metas = self.create_img_metas(sample_token, scene_token)

        return images_tensor, img_metas, cam_images


# ============================================================================
# Weight Loading
# ============================================================================


def remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap checkpoint keys to match model key names."""
    remapped = {}

    for k, v in state_dict.items():
        new_key = k

        if "img_neck.fpn_convs.0." in new_key:
            new_key = new_key.replace("img_neck.fpn_convs.0.", "img_neck.fpn_convs.")
        if "img_neck.lateral_convs.0." in new_key:
            new_key = new_key.replace("img_neck.lateral_convs.0.", "img_neck.lateral_convs.")

        # Encoder FFN: checkpoint uses ffns.0.layers.0/2, model uses ffns.0.layers.0.0/1
        if "encoder.layers" in new_key and "ffns.0.layers" in new_key:
            # Must do .2 first to avoid double replacement
            new_key = new_key.replace("ffns.0.layers.2.", "ffns.0.layers.1.")
            if "ffns.0.layers.0." in new_key and "ffns.0.layers.0.0" not in new_key:
                new_key = new_key.replace("ffns.0.layers.0.", "ffns.0.layers.0.0.")

        if "decoder.layers" in new_key:
            if ".attentions.0." in new_key:
                new_key = new_key.replace(".attentions.0.", ".self_attn.")
            elif ".attentions.1." in new_key:
                new_key = new_key.replace(".attentions.1.", ".cross_attn.")

            if "ffns.0.layers.0.0." in new_key:
                new_key = new_key.replace("ffns.0.layers.0.0.", "ffn.0.")
            elif "ffns.0.layers.1." in new_key:
                new_key = new_key.replace("ffns.0.layers.1.", "ffn.2.")

            if ".norms.0." in new_key:
                new_key = new_key.replace(".norms.0.", ".norm1.")
            elif ".norms.1." in new_key:
                new_key = new_key.replace(".norms.1.", ".norm2.")
            elif ".norms.2." in new_key:
                new_key = new_key.replace(".norms.2.", ".norm3.")

        if "transformer.can_bus_mlp.norm." in new_key:
            new_key = new_key.replace("transformer.can_bus_mlp.norm.", "transformer.can_bus_mlp.4.")

        remapped[new_key] = v

    return remapped


def load_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load model weights from checkpoint."""
    print(f"Loading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    new_state_dict = remap_checkpoint_keys(new_state_dict)

    model_state = model.state_dict()
    filtered_state_dict = {}
    size_mismatches = []

    for k, v in new_state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                size_mismatches.append(f"  {k}: checkpoint {v.shape} vs model {model_state[k].shape}")
                if not strict:
                    continue
        filtered_state_dict[k] = v

    if size_mismatches:
        print(f"\n⚠ Size mismatches found ({len(size_mismatches)}):")
        for msg in size_mismatches[:10]:
            print(msg)

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    print(f"\n✓ Weights loaded: {len(filtered_state_dict)} keys matched")
    if missing_keys:
        print(f"  Missing: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"  Unexpected: {len(unexpected_keys)} keys")

    return model


# ============================================================================
# Inference Pipeline
# ============================================================================


class MapTRInference:
    """MapTR inference pipeline with visualization."""

    def __init__(
        self,
        config: MapTRConfig,
        checkpoint_path: Optional[str] = None,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        print("Building model...")
        self.model = build_maptr_model(config, self.device)

        if checkpoint_path is not None:
            self.model = load_weights(self.model, checkpoint_path, strict=False)

        self.model.eval()

        self.image_processor = ImageProcessor(
            img_height=config.img_height,
            img_width=config.img_width,
        )
        self.calibration = CameraCalibration(num_cameras=config.num_cameras)
        self.visualizer = MapTRVisualizer(pc_range=config.pc_range)

        self.class_names = CLASS_NAMES

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
        """Run inference on images."""
        images = images.to(self.device)
        results = self.model(img_metas=img_metas, img=images)
        return results

    def predict_demo(self) -> List[Dict]:
        """Run inference on dummy data."""
        images = self.image_processor.generate_dummy_images(self.config.num_cameras)
        calibration = self.calibration.create_dummy_calibration(self.config.img_height, self.config.img_width)
        img_metas = self.create_img_metas(calibration)
        return self.predict(images, img_metas)

    def format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Format results for output."""
        formatted = {"num_detections": 0, "detections": []}

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

    def print_results(self, results: List[Dict], score_thresh: float = 0.0):
        """Print detection results."""
        print("\n" + "=" * 60)
        print("Detection Results")
        print("=" * 60)

        for result in results:
            if "pts_bbox" not in result:
                continue

            pts_bbox = result["pts_bbox"]
            scores = pts_bbox["scores_3d"]
            labels = pts_bbox["labels_3d"]

            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            keep = scores > score_thresh
            print(f"Total detections (score > {score_thresh}): {keep.sum()}")

            class_counts = {}
            for label in labels[keep]:
                label_int = int(label)
                class_name = self.class_names[label_int] if label_int < len(self.class_names) else f"class_{label_int}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")


# ============================================================================
# Main (based on official vis_pred.py)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MapTR Inference and Visualization Pipeline")

    # Core arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="maptr_tiny",
        choices=["small", "nuScenes", "maptr_tiny"],
        help="Model configuration preset",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, or auto)")

    # Input arguments
    parser.add_argument("--demo", action="store_true", help="Run demo with dummy data")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing camera images")
    parser.add_argument("--nuscenes", type=str, default=None, help="Path to nuScenes data root")
    parser.add_argument(
        "--nuscenes_version", type=str, default="v1.0-mini", choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"]
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to process")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to process")

    # Visualization arguments (matching official vis_pred.py)
    parser.add_argument("--score-thresh", default=0.4, type=float, help="Score threshold for visualization")
    parser.add_argument("--show-dir", type=str, default=None, help="Directory for visualization outputs")
    parser.add_argument("--show-cam", action="store_true", help="Save surround camera view")
    parser.add_argument(
        "--gt-format",
        type=str,
        nargs="+",
        default=["fixed_num_pts"],
        help="Visualization format: fixed_num_pts, bbox, polyline_pts, se_pts",
    )

    # Output arguments
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")

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
    print("MapTR Inference and Visualization Pipeline")
    print("Based on: https://github.com/hustvl/MapTR")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Score threshold: {args.score_thresh}")

    # Create output directory
    if args.show_dir is None:
        args.show_dir = "./work_dirs/vis_pred"
    os.makedirs(args.show_dir, exist_ok=True)
    print(f"Output directory: {args.show_dir}")

    # Create inference pipeline
    inference = MapTRInference(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Run inference
    if args.nuscenes:
        # nuScenes dataset mode
        print(f"\n{'='*60}")
        print("nuScenes Dataset Mode")
        print(f"{'='*60}")

        nuscenes_loader = NuScenesDataLoader(
            data_root=args.nuscenes,
            version=args.nuscenes_version,
            img_height=config.img_height,
            img_width=config.img_width,
        )

        all_samples = nuscenes_loader.get_sample_list()
        if all_samples:
            start_idx = args.sample_idx
            end_idx = min(start_idx + args.num_samples, len(all_samples))
            samples_to_process = all_samples[start_idx:end_idx]
            print(f"Processing samples {start_idx} to {end_idx-1}")
        else:
            print("⚠ No samples found")
            samples_to_process = []

        for i, sample_info in enumerate(samples_to_process):
            sample_token = sample_info["sample_token"]
            scene_token = sample_info.get("scene_token")

            print(f"\n--- Sample {i+1}/{len(samples_to_process)} ---")
            print(f"Token: {sample_token[:30]}...")

            # Create sample output directory
            sample_dir = osp.join(args.show_dir, f"sample_{i:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            # Load sample
            images_tensor, img_metas, cam_images = nuscenes_loader.load_sample(sample_token, scene_token)
            images_tensor = images_tensor.to(inference.device)

            # Save camera images
            if args.show_cam and cam_images:
                for cam_name, cam_img in cam_images.items():
                    try:
                        import cv2

                        cam_path = osp.join(sample_dir, f"{cam_name}.jpg")
                        cv2.imwrite(cam_path, cam_img)
                    except ImportError:
                        pass

                # Create surround view
                surround_path = osp.join(sample_dir, "surround_view.jpg")
                inference.visualizer.create_surround_view(cam_images, surround_path)

            # Run inference
            results = inference.predict(images_tensor, [img_metas])

            # Print results
            inference.print_results(results, score_thresh=args.score_thresh)

            # Visualize predictions
            if results and "pts_bbox" in results[0]:
                pts_bbox = results[0]["pts_bbox"]

                # Save prediction map (matching official format)
                for vis_format in args.gt_format:
                    pred_map_path = osp.join(sample_dir, f"PRED_MAP_{vis_format}.png")
                    inference.visualizer.visualize_predictions(
                        pts_bbox,
                        pred_map_path,
                        score_thresh=args.score_thresh,
                        vis_format=vis_format,
                        dpi=1200,
                    )

                # Save visualization with legend
                legend_path = osp.join(sample_dir, "PRED_MAP_legend.png")
                inference.visualizer.visualize_with_legend(
                    pts_bbox,
                    legend_path,
                    score_thresh=args.score_thresh,
                )

        print(f"\n{'='*60}")
        print(f"Processed {len(samples_to_process)} samples")
        print(f"Results saved to: {args.show_dir}")
        print(f"{'='*60}")

    elif args.demo:
        print("\nRunning demo inference with dummy data...")
        results = inference.predict_demo()
        inference.print_results(results)

        if results and "pts_bbox" in results[0]:
            pts_bbox = results[0]["pts_bbox"]
            demo_path = osp.join(args.show_dir, "demo_pred.png")
            inference.visualizer.visualize_with_legend(pts_bbox, demo_path)

    elif args.image_dir:
        print(f"\nLoading images from: {args.image_dir}")
        image_dir = Path(args.image_dir)
        image_paths = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            image_paths.extend(sorted(image_dir.glob(f"*{ext}")))

        if len(image_paths) < config.num_cameras:
            print(f"⚠ Found only {len(image_paths)} images, expected {config.num_cameras}")

        image_paths = [str(p) for p in image_paths[: config.num_cameras]]
        images = [inference.image_processor.load_image(p) for p in image_paths]
        images_tensor = inference.image_processor.preprocess(images)

        calibration = inference.calibration.create_dummy_calibration(config.img_height, config.img_width)
        img_metas = inference.create_img_metas(calibration)

        results = inference.predict(images_tensor, img_metas)
        inference.print_results(results, score_thresh=args.score_thresh)

        if results and "pts_bbox" in results[0]:
            pts_bbox = results[0]["pts_bbox"]
            pred_path = osp.join(args.show_dir, "pred_map.png")
            inference.visualizer.visualize_with_legend(
                pts_bbox,
                pred_path,
                score_thresh=args.score_thresh,
            )

    else:
        print("\nNo input provided, running demo mode...")
        results = inference.predict_demo()
        inference.print_results(results)

    # Save JSON results
    if args.output and results:
        formatted = inference.format_results(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
