# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Perception Transformer Reference Implementation (Inference-only)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from typing import List, Optional, Tuple


class ConvFuser(nn.Sequential):
    """Convolution-based feature fuser for multi-modal fusion."""

    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


class MapTRPerceptionTransformer(nn.Module):
    """MapTR Perception Transformer (Inference-only).

    Standalone PyTorch implementation derived from the original MapTRPerceptionTransformer.

    Args:
        encoder (nn.Module): BEVFormerEncoder module.
        decoder (nn.Module): MapTRDecoder module.
        embed_dims (int): Embedding dimensions. Default: 256.
        num_feature_levels (int): Number of feature maps from FPN. Default: 1.
        num_cams (int): Number of cameras. Default: 6.
        rotate_prev_bev (bool): Whether to rotate previous BEV features. Default: True.
        use_shift (bool): Whether to use shift for temporal alignment. Default: True.
        use_can_bus (bool): Whether to use CAN bus signals. Default: True.
        len_can_bus (int): Length of CAN bus signals. Default: 18.
        can_bus_norm (bool): Whether to normalize CAN bus signals. Default: True.
        use_cams_embeds (bool): Whether to use camera embeddings. Default: True.
        rotate_center (List[int]): Center for BEV rotation. Default: [100, 100].
        fuser (nn.Module, optional): Feature fuser for multi-modal fusion. Default: None.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        embed_dims: int = 256,
        num_feature_levels: int = 1,
        num_cams: int = 6,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        len_can_bus: int = 18,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        rotate_center: List[int] = None,
        fuser: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rotate_center is None:
            rotate_center = [100, 100]

        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center
        self.fuser = fuser

        # Check if using attention-based BEV encoder (BEVFormerEncoder has 'layers')
        self.use_attn_bev = hasattr(encoder, "layers")

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(self.len_can_bus, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

        # Initialize parameters
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def attn_bev_encode(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: torch.Tensor = None,
        prev_bev: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode BEV features using attention-based encoder (BEVFormerEncoder)."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # Obtain rotation angle and shift with ego motion
        img_metas = kwargs.get("img_metas", [{}])
        delta_x = np.array([each.get("can_bus", np.zeros(18))[0] for each in img_metas])
        delta_y = np.array([each.get("can_bus", np.zeros(18))[1] for each in img_metas])
        ego_angle = np.array([each.get("can_bus", np.zeros(18))[-2] / np.pi * 180 for each in img_metas])

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = img_metas[i].get("can_bus", np.zeros(18))[-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # Add can bus signals
        can_bus = bev_queries.new_tensor([each.get("can_bus", np.zeros(18)) for each in img_metas])
        can_bus = self.can_bus_mlp(can_bus[:, : self.len_can_bus])[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )
        return bev_embed

    def lss_bev_encode(
        self,
        mlvl_feats: List[torch.Tensor],
        prev_bev: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode BEV features using LSS-based encoder."""
        assert len(mlvl_feats) == 1, "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs.get("img_metas", [])
        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: torch.Tensor = None,
        prev_bev: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Obtain BEV features."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            bev_embed = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs,
            )

        # Fuse with LiDAR features if available
        if lidar_feat is not None and self.fuser is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = F.interpolate(lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev

        return bev_embed

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        object_query_embed: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: torch.Tensor = None,
        reg_branches: Optional[nn.ModuleList] = None,
        cls_branches: Optional[nn.ModuleList] = None,
        prev_bev: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            mlvl_feats: Input features from different levels, each with shape (bs, num_cams, c, h, w).
            lidar_feat: LiDAR features for fusion (optional).
            bev_queries: BEV queries with shape (bev_h*bev_w, embed_dims).
            object_query_embed: Query embedding for decoder with shape (num_query, embed_dims*2).
            bev_h: BEV height.
            bev_w: BEV width.
            grid_length: Grid length for spatial alignment.
            bev_pos: BEV positional encoding with shape (bs, embed_dims, bev_h, bev_w).
            reg_branches: Regression branches for box refinement.
            cls_branches: Classification branches.
            prev_bev: Previous BEV features for temporal fusion.

        Returns:
            Tuple of:
                - bev_embed: BEV features.
                - inter_states: Outputs from decoder.
                - init_reference_out: Initial reference points.
                - inter_references_out: Intermediate reference points from decoder.
        """
        if grid_length is None:
            grid_length = [0.512, 0.512]

        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
