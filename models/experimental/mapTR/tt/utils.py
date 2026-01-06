# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def multi_scale_deformable_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    device,
) -> torch.Tensor:
    """TT implementation of multi-scale deformable attention.

    Args:
        value: (bs, num_value, num_heads, embed_dims//num_heads)
        spatial_shapes: (num_levels, 2) - (h, w) for each level
        sampling_locations: (bs, num_query, num_heads, num_levels, num_points, 2)
        attention_weights: (bs, num_query, num_heads, num_levels, num_points)
        device: TT device

    Returns:
        output: (bs, num_query, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape

    # Convert to torch for grid_sample operation
    value_torch = ttnn.to_torch(value)
    sampling_locations_torch = ttnn.to_torch(sampling_locations)
    attention_weights_torch = ttnn.to_torch(attention_weights)
    spatial_shapes_torch = ttnn.to_torch(spatial_shapes)

    # Perform multi-scale deformable attention in PyTorch
    value_list = value_torch.split([h * w for h, w in spatial_shapes_torch], dim=1)
    sampling_grids = 2 * sampling_locations_torch - 1
    sampling_value_list = []

    for level_idx, (h, w) in enumerate(spatial_shapes_torch):
        value_l = value_list[level_idx].permute(0, 2, 3, 1).reshape(bs * num_heads, embed_dims, h, w)
        sampling_grid_l = (
            sampling_grids[:, :, :, level_idx].permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_query, num_points, 2)
        )
        sampling_value_l = torch.nn.functional.grid_sample(
            value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights_torch.permute(0, 2, 1, 3, 4).reshape(
        bs * num_heads, 1, num_query, num_levels * num_points
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).reshape(bs * num_heads, embed_dims, num_query, num_levels * num_points)
        * attention_weights
    ).sum(-1)
    output = output.reshape(bs, num_heads * embed_dims, num_query).permute(0, 2, 1)

    # Convert back to TT tensor
    return ttnn.from_torch(output, device=device, layout=ttnn.TILE_LAYOUT)
