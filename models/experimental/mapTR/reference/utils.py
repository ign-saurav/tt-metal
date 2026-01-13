# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """PyTorch implementation of multi-scale deformable attention.

    Args:
        value: Value tensor with shape (bs, num_value, num_heads, embed_dims)
        value_spatial_shapes: Spatial shapes tensor with shape (num_levels, 2)
        sampling_locations: Sampling locations with shape (bs, num_query, num_heads, num_levels, num_points, 2)
        attention_weights: Attention weights with shape (bs, num_query, num_heads, num_levels, num_points)

    Returns:
        Output tensor with shape (bs, num_query, num_heads * embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
