# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.mapTR.tt.utils import multi_scale_deformable_attn


class TtTemporalSelfAttention:
    """TT implementation of TemporalSelfAttention for mapTR."""

    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        batch_first=True,
    ):
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.device = device
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first
        self.params = params

        dim_per_head = embed_dims // num_heads
        if not (dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0:
            warnings.warn(
                "For optimal performance with TTNN, embed_dims should be set "
                "so that dimension of each attention head is a power of 2"
            )

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward pass of TT TemporalSelfAttention."""
        params = self.params

        # Handle value input
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = ttnn.stack([query, query], dim=1)
            value = ttnn.reshape(value, (bs * 2, len_bev, c))

        # Handle identity
        if identity is None:
            identity = query

        # Add positional encoding if provided
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        # Handle batch_first
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue == 2

        # Concatenate value and query
        query = ttnn.concat([value[:bs], query], dim=-1)

        # Value projection
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)

        # Handle key padding mask
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        # Reshape value for attention
        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))

        # Compute sampling offsets
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )
        sampling_offsets = ttnn.reallocate(sampling_offsets)

        # Compute attention weights
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(params.attention_weights.weight)
        ttnn.deallocate(params.attention_weights.bias)
        ttnn.deallocate(query)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        )

        # Permute and reshape for multi-scale attention
        attention_weights = ttnn.permute(attention_weights, (0, 3, 1, 2, 4, 5))
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        sampling_offsets = ttnn.permute(sampling_offsets, (0, 3, 1, 2, 4, 5, 6))
        sampling_offsets = ttnn.reallocate(sampling_offsets)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r = reference_points.shape[0]

            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            ttnn.deallocate(offset_normalizer)

            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)

            sampling_offsets_shape = sampling_offsets.shape
            # Reshape to 4D for division (TTNN broadcast limit is rank 5)
            sampling_offsets_4d = ttnn.reshape(
                sampling_offsets, (sampling_offsets.shape[0], -1, sampling_offsets.shape[4], sampling_offsets.shape[5])
            )
            offset_normalizer_4d = ttnn.reshape(
                offset_normalizer_xy,
                (
                    offset_normalizer_xy.shape[0],
                    offset_normalizer_xy.shape[1],
                    offset_normalizer_xy.shape[2],
                    offset_normalizer_xy.shape[-1],
                ),
            )
            sampling_locations = ttnn.div(sampling_offsets_4d, offset_normalizer_4d)
            ttnn.deallocate(offset_normalizer_xy)
            ttnn.deallocate(offset_normalizer_4d)
            ttnn.deallocate(sampling_offsets_4d)
            sampling_locations = ttnn.reshape(sampling_locations, sampling_offsets_shape)

            # Expand reference_points to match sampling_locations shape for addition
            # TTNN only supports broadcasting up to rank 5, so we reshape to 4D
            # sampling_locations: (bs*num_bev_queue, num_query, num_heads, num_levels, num_points, 2)
            #                  -> (bs*num_bev_queue, num_query, num_heads*num_levels*num_points, 2)
            sl_shape = sampling_locations.shape
            sampling_locations_4d = ttnn.reshape(
                sampling_locations, (sl_shape[0], sl_shape[1], sl_shape[2] * sl_shape[3] * sl_shape[4], sl_shape[5])
            )
            # reference_points: (bs_r, num_query, num_levels, 2)
            # Repeat to match sampling_locations shape before adding
            ref_pts_4d = ttnn.repeat(reference_points, (sl_shape[0] // bs_r, 1, sl_shape[2] * sl_shape[4], 1))
            ref_pts_4d = ttnn.to_layout(ref_pts_4d, ttnn.TILE_LAYOUT)

            sampling_locations = ttnn.add(ref_pts_4d, sampling_locations_4d)
            ttnn.deallocate(ref_pts_4d)
            ttnn.deallocate(sampling_locations_4d)
            sampling_locations = ttnn.reshape(sampling_locations, sl_shape)
        else:
            raise ValueError(f"Last dim of reference_points must be 2, but got {reference_points.shape[-1]} .")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)

        # Clean up intermediate tensors
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(sampling_offsets)
        ttnn.deallocate(value)

        # Reshape and fuse history
        output = ttnn.permute(output, (1, 2, 0))
        output = ttnn.reshape(output, (num_query, embed_dims, bs, self.num_bev_queue))
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.mean(output, dim=-1)
        output = ttnn.permute(output, (2, 0, 1))

        # Output projection
        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        ttnn.deallocate(params.output_proj.weight)
        ttnn.deallocate(params.output_proj.bias)

        # Handle batch_first
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        # Add residual connection
        output = ttnn.add(output, identity)
        ttnn.deallocate(identity)

        return output
