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

    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level_idx, (h, w) in enumerate(spatial_shapes):
        # (bs, h*w, num_heads, embed_dims) -> (bs, num_heads, embed_dims, h, w)
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


class TemporalSelfAttention(nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr (inference-only, fp16).

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default: 4.
        num_bev_queue (int): Length of BEV queue (history + current). Default: 2.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_bev_queue: int = 2,
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
        self.num_bev_queue = num_bev_queue

        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

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
        """Forward Function of TemporalSelfAttention.

        Args:
            query (Tensor): Query of Transformer with shape (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape (bs, num_key, embed_dims).
            value (Tensor): The value tensor with shape (bs, num_key, embed_dims).
            identity (Tensor): The tensor used for residual, same shape as `query`. Default None.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].
            reference_points (Tensor): Normalized reference points with shape
                (bs, num_query, num_levels, 2), all elements in range [0, 1].
            spatial_shapes (Tensor): Spatial shape of features in different levels.
                With shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.

        Returns:
            Tensor: forwarded results with shape (bs, num_query, embed_dims).
        """
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )

        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims) -> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue) -> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs) -> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output + identity

    def to_fp16(self) -> "TemporalSelfAttention":
        """Convert model parameters to fp16."""
        return self.half()
