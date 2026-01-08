# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """PyTorch implementation of multi-scale deformable attention.

    Args:
        value: (bs, num_value, num_heads, embed_dims//num_heads)
        spatial_shapes: (num_levels, 2) - (h, w) for each level
        sampling_locations: (bs, num_query, num_heads, num_levels, num_points, 2)
        attention_weights: (bs, num_query, num_heads, num_levels, num_points)

    Returns:
        output: (bs, num_query, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([int(h * w) for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level_idx, (h, w) in enumerate(spatial_shapes):
        h, w = int(h), int(w)
        # (bs, h*w, num_heads, embed_dims) -> (bs*num_heads, embed_dims, h, w)
        value_l = value_list[level_idx].permute(0, 2, 3, 1).reshape(bs * num_heads, embed_dims, h, w)
        # (bs, num_query, num_heads, num_points, 2) -> (bs*num_heads, num_query, num_points, 2)
        sampling_grid_l = (
            sampling_grids[:, :, :, level_idx].permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_query, num_points, 2)
        )
        # (bs*num_heads, embed_dims, num_query, num_points)
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l)

    # (bs, num_query, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_query, num_levels*num_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * num_heads, 1, num_query, num_levels * num_points
    )
    # (bs*num_heads, embed_dims, num_query, num_levels, num_points) -> (bs*num_heads, embed_dims, num_query, num_levels*num_points)
    output = (
        torch.stack(sampling_value_list, dim=-2).reshape(bs * num_heads, embed_dims, num_query, num_levels * num_points)
        * attention_weights
    ).sum(-1)
    # (bs*num_heads, embed_dims, num_query) -> (bs, num_query, num_heads*embed_dims)
    output = output.reshape(bs, num_heads * embed_dims, num_query).permute(0, 2, 1)
    return output


class SpatialCrossAttention(nn.Module):
    """Spatial Cross Attention module used in BEVFormer

    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_cams (int): The number of cameras. Default: 6.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim). Default: False.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        batch_first: bool = False,
        deformable_attention: nn.Module = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        self.deformable_attention = deformable_attention
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        reference_points_cam: torch.Tensor = None,
        bev_mask: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of SpatialCrossAttention.

        Args:
            query (Tensor): Query with shape (bs, num_query, embed_dims).
            key (Tensor): Key tensor with shape (num_cams, num_key, bs, embed_dims).
            value (Tensor): Value tensor with shape (num_cams, num_key, bs, embed_dims).
            residual (Tensor): Tensor for residual connection. Default None.
            query_pos (Tensor): Positional encoding for query. Default: None.
            reference_points_cam (Tensor): Reference points in camera coordinates.
            bev_mask (Tensor): BEV mask for valid queries per camera.
            spatial_shapes (Tensor): Spatial shape of features (num_levels, 2).
            level_start_index (Tensor): Start index of each level.

        Returns:
            Tensor: forwarded results with shape (bs, num_query, embed_dims).
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # Each camera only interacts with its corresponding BEV queries
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        # key shape: (num_cams, l, bs, embed_dims)
        num_cams_from_key, l, bs_from_key, embed_dims_from_key = key.shape

        # Use bs from query (already computed above) for consistency
        key = key.permute(2, 0, 1, 3).reshape(bs_from_key * num_cams_from_key, l, embed_dims_from_key)
        value = value.permute(2, 0, 1, 3).reshape(bs_from_key * num_cams_from_key, l, embed_dims_from_key)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return slots + inp_residual

    def to_fp16(self) -> "SpatialCrossAttention":
        """Convert model parameters to fp16."""
        return self.half()


class MSDeformableAttention3D(nn.Module):
    """Multi-Scale Deformable Attention for 3D (inference-only, fp16).

    Based on Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>

    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default: 8.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of MSDeformableAttention3D.

        Args:
            query (Tensor): Query with shape (bs, num_query, embed_dims).
            key (Tensor): Key tensor with shape (bs, num_key, embed_dims).
            value (Tensor): Value tensor with shape (bs, num_key, embed_dims).
            identity (Tensor): Tensor for residual, same shape as query. Default None.
            query_pos (Tensor): Positional encoding for query. Default: None.
            key_padding_mask (Tensor): ByteTensor for query, shape [bs, num_key].
            reference_points (Tensor): Normalized reference points with shape
                (bs, num_query, num_Z_anchors, 2), all elements in range [0, 1].
            spatial_shapes (Tensor): Spatial shape of features (num_levels, 2).
            level_start_index (Tensor): Start index of each level.

        Returns:
            Tensor: forwarded results with shape (bs, num_query, embed_dims).
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
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
            # For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            # After projecting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
            )
            sampling_locations = reference_points + sampling_offsets

            bs, num_query, num_heads, num_levels, num_points_per_z, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points_per_z * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(f"Last dim of reference_points must be 2, but got {reference_points.shape[-1]} instead.")

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    def to_fp16(self) -> "MSDeformableAttention3D":
        """Convert model parameters to fp16."""
        return self.half()


class MSIPM3D(nn.Module):
    """Multi-Scale Image Projection Module for 3D with fixed sampling offsets (inference-only, fp16).

    Based on Deformable DETR with fixed/uniform sampling offsets and attention weights.

    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default: 8.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
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

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # Fixed sampling offsets (not learned)
        self._init_fixed_offsets()

    def _init_fixed_offsets(self):
        """Initialize fixed sampling offsets."""
        import math

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.register_buffer("fixed_sampling_offsets", grid_init.view(-1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of MSIPM3D.

        Args:
            query (Tensor): Query with shape (bs, num_query, embed_dims).
            key (Tensor): Key tensor with shape (bs, num_key, embed_dims).
            value (Tensor): Value tensor with shape (bs, num_key, embed_dims).
            identity (Tensor): Tensor for residual, same shape as query. Default None.
            query_pos (Tensor): Positional encoding for query. Default: None.
            key_padding_mask (Tensor): ByteTensor for query, shape [bs, num_key].
            reference_points (Tensor): Normalized reference points with shape
                (bs, num_query, num_Z_anchors, 2), all elements in range [0, 1].
            spatial_shapes (Tensor): Spatial shape of features (num_levels, 2).
            level_start_index (Tensor): Start index of each level.

        Returns:
            Tensor: forwarded results with shape (bs, num_query, embed_dims).
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # Use fixed sampling offsets (expanded for batch and query)
        sampling_offsets = self.fixed_sampling_offsets.view(
            1, 1, self.num_heads, self.num_levels, self.num_points, 2
        ).expand(bs, num_query, -1, -1, -1, -1)

        # Uniform attention weights
        attention_weights = query.new_ones((bs, num_query, self.num_heads, self.num_levels * self.num_points))
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
            )
            sampling_locations = reference_points + sampling_offsets

            bs, num_query, num_heads, num_levels, num_points_per_z, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points_per_z * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(f"Last dim of reference_points must be 2, but got {reference_points.shape[-1]} instead.")

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    def to_fp16(self) -> "MSIPM3D":
        """Convert model parameters to fp16."""
        return self.half()
