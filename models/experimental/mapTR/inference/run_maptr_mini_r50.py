#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Mini with ResNet50 and BEVFormer Encoder Inference Pipeline

This script runs inference with:
- ResNet50 backbone
- FPN neck
- BEVFormer encoder (attention-based BEV generation)
- MapTR decoder head

Usage:
    # Demo mode
    python models/experimental/mapTR/inference/run_maptr_mini_r50.py --demo

    # With images
    python models/experimental/mapTR/inference/run_maptr_mini_r50.py \
        --image_dir /path/to/images/ \
        --checkpoint /path/to/maptr_mini_r50.pth

    # With sample data
    python models/experimental/mapTR/inference/run_maptr_mini_r50.py \
        --image_dir models/experimental/mapTR/resources/data/sample/images/
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR components
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_maptr import MapTR


# ============================================================================
# MapTR Mini R50 Configuration
# ============================================================================


class MapTRMiniR50Config:
    """Configuration for MapTR Mini with ResNet50 backbone."""

    # Image settings
    img_height: int = 480
    img_width: int = 800
    num_cameras: int = 6

    # Model architecture
    embed_dims: int = 256
    num_classes: int = 3  # divider, ped_crossing, boundary
    num_vec: int = 50
    num_pts_per_vec: int = 20

    # BEV settings
    bev_h: int = 50
    bev_w: int = 50
    pc_range: List[float] = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

    # BEVFormer encoder settings
    num_encoder_layers: int = 3
    num_decoder_layers: int = 6
    num_heads: int = 8
    feedforward_channels: int = 512
    num_points_in_pillar: int = 4

    # Backbone settings (ResNet50)
    backbone_depth: int = 50
    fpn_in_channels: List[int] = [512, 1024, 2048]  # C3, C4, C5
    fpn_out_channels: int = 256
    fpn_num_outs: int = 4


# ============================================================================
# BEVFormer Components (Simplified for Inference)
# ============================================================================


class TemporalSelfAttention(nn.Module):
    """Temporal Self Attention for BEVFormer (inference-only)."""

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
        key_pos: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if key is None:
            key = query
        if value is None:
            value = key

        bs, num_query, _ = query.shape
        value = self.value_proj(value)

        # Simplified attention for inference
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(bs, num_query, self.num_heads, -1)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Simple aggregation
        value = value.view(bs, -1, self.num_heads, self.embed_dims // self.num_heads)
        output = torch.einsum("bqhn,bnhd->bqhd", attn_weights, value)
        output = output.reshape(bs, num_query, -1)
        output = self.output_proj(output)

        return output + identity


class MSDeformableAttention3D(nn.Module):
    """Multi-Scale Deformable Attention for 3D (inference-only)."""

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
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if value is None:
            value = query

        bs, num_query, _ = query.shape
        value = self.value_proj(value)

        # Compute attention weights
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(bs, num_query, self.num_heads, -1)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Simple aggregation
        bs_v, num_value, _ = value.shape
        value = value.view(bs_v, num_value, self.num_heads, -1)

        # Average pooling over value for simplicity
        output = value.mean(dim=1, keepdim=True).expand(-1, num_query, -1, -1)
        output = (output * attn_weights.unsqueeze(-1)).sum(dim=-2)
        output = output.reshape(bs, num_query, -1)
        output = self.output_proj(output)

        return output + identity


class SpatialCrossAttention(nn.Module):
    """Spatial Cross Attention for BEVFormer (inference-only)."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        pc_range: List[float] = None,
        batch_first: bool = True,
        deformable_attention: nn.Module = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.batch_first = batch_first

        if deformable_attention is None:
            deformable_attention = MSDeformableAttention3D(embed_dims=embed_dims)
        self.deformable_attention = deformable_attention
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        reference_points_cam: torch.Tensor = None,
        bev_mask: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        # Simplified cross attention
        output = self.deformable_attention(
            query=query,
            value=value.reshape(bs, -1, self.embed_dims) if value.dim() > 3 else value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        output = self.output_proj(output)
        return output + identity


class FFN(nn.Module):
    """Feed Forward Network."""

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 512,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, identity: torch.Tensor = None) -> torch.Tensor:
        if identity is None:
            identity = x
        return self.layers(x) + identity


class BEVFormerLayer(nn.Module):
    """Single BEVFormer encoder layer."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        num_cams: int = 6,
        pc_range: List[float] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # Self attention (temporal)
        self.self_attn = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        # Cross attention (spatial)
        self.cross_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            pc_range=pc_range,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # FFN
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bev_pos: torch.Tensor = None,
        ref_2d: torch.Tensor = None,
        ref_3d: torch.Tensor = None,
        bev_h: int = None,
        bev_w: int = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        reference_points_cam: torch.Tensor = None,
        bev_mask: torch.Tensor = None,
        prev_bev: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self attention
        query = self.self_attn(
            query,
            prev_bev if prev_bev is not None else query,
            prev_bev if prev_bev is not None else query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )
        query = self.norm1(query)

        # Cross attention
        query = self.cross_attn(
            query,
            key,
            value,
            query_pos=bev_pos,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)

        # FFN
        query = self.ffn(query)
        query = self.norm3(query)

        return query


class BEVFormerEncoder(nn.Module):
    """BEVFormer Encoder with attention-based BEV generation."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_channels: int = 512,
        num_cams: int = 6,
        num_points_in_pillar: int = 4,
        pc_range: List[float] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.layers = nn.ModuleList(
            [
                BEVFormerLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    num_cams=num_cams,
                    pc_range=self.pc_range,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        H: int,
        W: int,
        Z: float = 8,
        num_points_in_pillar: int = 4,
        dim: str = "3d",
        bs: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate reference points for BEV queries."""
        if device is None:
            device = torch.device("cpu")

        if dim == "3d":
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
            zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z

            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            xs = xs.view(1, 1, W).expand(num_points_in_pillar, H, W) / W

            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
            ys = ys.view(1, H, 1).expand(num_points_in_pillar, H, W) / H

            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def forward(
        self,
        bev_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        prev_bev: torch.Tensor = None,
        shift: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through BEVFormer encoder."""
        bs = bev_query.size(1)
        device = bev_query.device
        dtype = bev_query.dtype

        # Generate reference points
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            Z=self.pc_range[5] - self.pc_range[2],
            num_points_in_pillar=self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            device=device,
            dtype=dtype,
        )

        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bs,
            device=device,
            dtype=dtype,
        )

        # Permute for processing
        bev_query = bev_query.permute(1, 0, 2)  # (bs, num_query, embed_dims)
        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)

        # Process previous BEV for temporal fusion
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)

        output = bev_query
        for layer in self.layers:
            output = layer(
                output,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                **kwargs,
            )

        return output


# ============================================================================
# MapTR Transformer with BEVFormer
# ============================================================================


class MapTRTransformerBEVFormer(nn.Module):
    """MapTR Transformer with BEVFormer encoder."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 6,
        feedforward_channels: int = 512,
        num_cams: int = 6,
        num_points_in_pillar: int = 4,
        bev_h: int = 50,
        bev_w: int = 50,
        pc_range: List[float] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        # BEVFormer encoder
        self.encoder = BEVFormerEncoder(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            feedforward_channels=feedforward_channels,
            num_cams=num_cams,
            num_points_in_pillar=num_points_in_pillar,
            pc_range=self.pc_range,
        )

        # Decoder
        self.decoder = SimpleDecoder(
            embed_dims=embed_dims,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
        )

        # Additional components
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
        """Forward pass."""
        bs = mlvl_feats[0].size(0)

        # Prepare BEV queries
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos_flat = bev_pos.flatten(2).permute(2, 0, 1)

        # Add CAN bus info if available
        if img_metas is not None:
            can_bus = bev_queries.new_tensor([meta.get("can_bus", np.zeros(18))[:18] for meta in img_metas])
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

        # Prepare multi-level features
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cam, bs, h*w, c)
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

        # Run decoder
        hs, inter_references = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
        )

        return bev_embed.permute(1, 0, 2), hs, reference_points, inter_references

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


class SimpleDecoder(nn.Module):
    """Simple decoder for MapTR."""

    def __init__(self, embed_dims: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.layers = nn.ModuleList([SimpleDecoderLayer(embed_dims, num_heads) for _ in range(num_layers)])

    def forward(self, query, key, value, query_pos, reference_points):
        output = query.permute(1, 0, 2)  # (num_query, bs, embed_dims)
        key = key.permute(1, 0, 2) if key.dim() == 3 else key
        value = value.permute(1, 0, 2) if value.dim() == 3 else value
        query_pos_perm = query_pos.permute(1, 0, 2)

        intermediate = []
        intermediate_reference_points = []

        for layer in self.layers:
            output_t = output.permute(1, 0, 2)
            key_t = key.permute(1, 0, 2) if key.dim() == 3 else key
            value_t = value.permute(1, 0, 2) if value.dim() == 3 else value
            query_pos_t = query_pos_perm.permute(1, 0, 2)

            output_t = layer(output_t, key_t, value_t, query_pos_t)
            output = output_t.permute(1, 0, 2)
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        hs = torch.stack(intermediate)
        inter_references = torch.stack(intermediate_reference_points)
        return hs, inter_references


class SimpleDecoderLayer(nn.Module):
    """Simple decoder layer."""

    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, query_pos):
        q = k = query + query_pos
        query2 = self.self_attn(q, k, query)[0]
        query = self.norm1(query + query2)

        query2 = self.cross_attn(query + query_pos, key, value)[0]
        query = self.norm2(query + query2)

        query2 = self.ffn(query)
        query = self.norm3(query + query2)
        return query


# ============================================================================
# Model Builder
# ============================================================================


def build_maptr_mini_r50(config: MapTRMiniR50Config = None, device: torch.device = None) -> MapTR:
    """Build MapTR Mini with R50 backbone and BEVFormer encoder."""
    if config is None:
        config = MapTRMiniR50Config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet50 backbone
    backbone = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        out_indices=(2,),  # Only C4
    )

    # FPN neck
    fpn = FPN(
        in_channels=[1024],
        out_channels=config.fpn_out_channels,
        num_outs=1,
    )

    # MapTR Transformer with BEVFormer
    transformer = MapTRTransformerBEVFormer(
        embed_dims=config.embed_dims,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        feedforward_channels=config.feedforward_channels,
        num_cams=config.num_cameras,
        num_points_in_pillar=config.num_points_in_pillar,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
    )

    # Positional encoding
    pos_enc = LearnedPositionalEncoding(
        num_feats=config.embed_dims // 2,
        row_num_embed=config.bev_h,
        col_num_embed=config.bev_w,
    )

    # MapTR head
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
        query_embed_type="all_pts",
        transform_method="minmax",
    )

    # Full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    return model.to(device)


# ============================================================================
# Inference Pipeline
# ============================================================================


class MapTRMiniInference:
    """MapTR Mini inference pipeline."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = MapTRMiniR50Config()

        print(f"Building MapTR Mini R50 with BEVFormer encoder...")
        print(f"  Device: {self.device}")
        print(f"  BEV size: {self.config.bev_h} x {self.config.bev_w}")
        print(f"  Encoder layers: {self.config.num_encoder_layers}")
        print(f"  Decoder layers: {self.config.num_decoder_layers}")

        self.model = build_maptr_mini_r50(self.config, self.device)

        if checkpoint_path:
            self.load_weights(checkpoint_path)

        self.model.eval()
        self.class_names = ["divider", "ped_crossing", "boundary"]

    def load_weights(self, checkpoint_path: str):
        """Load model weights."""
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

        # Remove module prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """Load and preprocess images."""
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]

        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.config.img_width, self.config.img_height), Image.BILINEAR)
            img = np.array(img).astype(np.float32)
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)
            images.append(img)

        images = np.stack(images, axis=0)
        return torch.from_numpy(images[np.newaxis, ...]).float()

    def create_img_metas(self, num_cams: int = 6) -> List[List[Dict]]:
        """Create image metadata."""
        meta = {
            "scene_token": "inference",
            "can_bus": np.zeros(18),
            "lidar2img": np.eye(4)[np.newaxis, :, :].repeat(num_cams, axis=0),
            "img_shape": [(self.config.img_height, self.config.img_width)] * num_cams,
            "prev_bev_exists": False,
        }
        return [[meta]]

    @torch.no_grad()
    def predict(self, image_paths: List[str]) -> List[Dict]:
        """Run inference."""
        images = self.preprocess_images(image_paths).to(self.device)
        img_metas = self.create_img_metas(len(image_paths))
        results = self.model(img_metas=img_metas, img=images)
        return results

    @torch.no_grad()
    def predict_demo(self) -> List[Dict]:
        """Run demo with dummy data."""
        images = torch.randn(1, self.config.num_cameras, 3, self.config.img_height, self.config.img_width).to(
            self.device
        )
        img_metas = self.create_img_metas(self.config.num_cameras)
        return self.model(img_metas=img_metas, img=images)

    def print_results(self, results: List[Dict]):
        """Print detection results."""
        print("\n" + "=" * 60)
        print("Detection Results")
        print("=" * 60)

        total = 0
        for result in results:
            if "pts_bbox" not in result:
                continue
            pts_bbox = result["pts_bbox"]
            num_dets = pts_bbox["boxes_3d"].shape[0]
            total += num_dets

            print(f"Detections: {num_dets}")
            for i in range(min(5, num_dets)):
                label = pts_bbox["labels_3d"][i].item()
                score = pts_bbox["scores_3d"][i].item()
                cls_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                print(f"  [{i}] {cls_name}: score={score:.4f}")

        print(f"\nTotal detections: {total}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MapTR Mini R50 with BEVFormer Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory with camera images")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Image paths")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--demo", action="store_true", help="Run demo with dummy data")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("MapTR Mini R50 with BEVFormer Encoder")
    print("=" * 60)

    # Create inference pipeline
    inference = MapTRMiniInference(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Run inference
    if args.demo:
        print("\nRunning demo inference...")
        results = inference.predict_demo()

    elif args.images:
        print(f"\nProcessing {len(args.images)} images...")
        results = inference.predict(args.images)

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = sorted([str(p) for p in image_dir.glob("*.jpg")] + [str(p) for p in image_dir.glob("*.png")])[:6]
        print(f"\nProcessing images from: {image_dir}")
        print(f"  Found {len(image_paths)} images")
        results = inference.predict(image_paths)

    else:
        print("\nNo input provided, running demo...")
        results = inference.predict_demo()

    # Print results
    inference.print_results(results)

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        formatted = {"detections": []}
        for result in results:
            if "pts_bbox" in result:
                pts_bbox = result["pts_bbox"]
                for i in range(pts_bbox["boxes_3d"].shape[0]):
                    formatted["detections"].append(
                        {
                            "class": inference.class_names[pts_bbox["labels_3d"][i].item()],
                            "score": pts_bbox["scores_3d"][i].item(),
                            "bbox": pts_bbox["boxes_3d"][i].cpu().numpy().tolist(),
                            "points": pts_bbox["pts_3d"][i].cpu().numpy().tolist(),
                        }
                    )

        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
