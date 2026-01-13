# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import torch.nn as nn
import copy
import warnings

# Use local MapTR implementations
from models.experimental.mapTR.reference.pytorch_temporal_self_attention import TemporalSelfAttention
from models.experimental.mapTR.reference.pytorch_spatial_cross_attention import (
    SpatialCrossAttention,
    MSDeformableAttention3D,
)
from models.experimental.mapTR.reference.pytorch_custom_base_transformer_layer import FFN


class BEVFormerEncoder(nn.Module):
    def __init__(
        self,
        num_layers=3,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=256,
        num_heads=8,
        num_points=8,
        im2col_step=192,
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ):
        super(BEVFormerEncoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.num_layers = num_layers
        self.fp16_enabled = False

        transformer_layers = dict(
            attn_cfgs=[
                # TemporalSelfAttention uses num_heads=8
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_heads=8, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    attention=dict(
                        embed_dims=embed_dims,
                        num_heads=num_heads,  # MSDeformableAttention3D uses 8 heads
                        num_levels=1,
                        num_points=num_points,
                        im2col_step=im2col_step,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
        )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(BEVFormerLayer(**transformer_layers))

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device="cuda", dtype=torch.float):
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d

    def point_sampling(self, reference_points, pc_range, img_metas):  # TODO Handle fp32
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # Expected: (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        # lidar2img should be (B, num_cam, 4, 4)
        # If it's (num_cam, 4, 4) without batch dim, need to handle correctly
        if lidar2img.dim() == 3:
            # Shape is (num_cam, 4, 4), need to unsqueeze and repeat for batch
            num_cam = lidar2img.size(0)
            lidar2img = lidar2img.unsqueeze(0).expand(B, -1, -1, -1)  # (B, num_cam, 4, 4)
        else:
            num_cam = lidar2img.size(1)
            # Verify batch size matches
            if lidar2img.size(0) != B:
                lidar2img = lidar2img.expand(B, -1, -1, -1)  # Expand to batch size

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]
        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        # TODO handle
        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    # TODO Handle fp16
    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        output = bev_query
        intermediate = []

        # Note: bev_query comes in as (num_bev, bs, embed_dims), permute happens later
        # So use size(1) for bs before permute
        batch_size = bev_query.size(1)

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=batch_size,
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=batch_size, device=bev_query.device, dtype=bev_query.dtype
        )

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, kwargs["img_metas"])

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class BEVFormerLayer(nn.Module):
    def __init__(self, attn_cfgs, feedforward_channels, ffn_dropout=0.0, operation_order=None, ffn_num_fcs=2, **kwargs):
        super(BEVFormerLayer, self).__init__()

        self.attn_cfgs = attn_cfgs
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.ffn_num_fcs = ffn_num_fcs
        self.fp16_enabled = False
        self.batch_first = True
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in self.operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                cfg = attn_cfgs[index]
                if "batch_first" not in cfg:
                    cfg["batch_first"] = self.batch_first

                if cfg["type"] == "TemporalSelfAttention":
                    # Build TemporalSelfAttention
                    # Checkpoint uses num_heads=8, num_levels=1, num_points=4
                    attention = TemporalSelfAttention(
                        embed_dims=cfg.get("embed_dims", 256),
                        num_heads=cfg.get("num_heads", 8),
                        num_levels=cfg.get("num_levels", 1),
                        num_points=cfg.get("num_points", 4),
                        num_bev_queue=cfg.get("num_bev_queue", 2),
                        batch_first=cfg.get("batch_first", True),
                    )
                elif cfg["type"] == "SpatialCrossAttention":
                    # Build MSDeformableAttention3D (as used in BEVFormer checkpoint)
                    deform_cfg = cfg.get("attention", {})
                    deformable_attention = MSDeformableAttention3D(
                        embed_dims=deform_cfg.get("embed_dims", 256),
                        num_heads=deform_cfg.get("num_heads", 8),
                        num_levels=deform_cfg.get("num_levels", 1),
                        num_points=deform_cfg.get("num_points", 8),
                        im2col_step=deform_cfg.get("im2col_step", 64),
                        batch_first=deform_cfg.get("batch_first", True),
                    )
                    # Build SpatialCrossAttention with MSDeformableAttention3D
                    attention = SpatialCrossAttention(
                        embed_dims=cfg.get("embed_dims", 256),
                        num_cams=cfg.get("num_cams", 6),
                        pc_range=cfg.get("pc_range"),
                        batch_first=cfg.get("batch_first", True),
                    )
                    # Replace default attention with configured one
                    attention.deformable_attention = deformable_attention
                else:
                    raise ValueError(f"Unknown attention type: {cfg['type']}")

                self.attentions.append(attention)
                index += 1

        self.pre_norm = operation_order[0] == "norm"

        self.embed_dims = self.attentions[0].embed_dims

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")

        for ffn_index in range(num_ffns):
            self.ffns.append(FFN(self.embed_dims, feedforward_channels=self.feedforward_channels))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(
        self,
        query,
        key,
        value,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1

                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
