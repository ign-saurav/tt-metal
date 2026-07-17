# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.BEVFormerV2.tt.ttnn_utils import multi_scale_deformable_attn


class TtTemporalSelfAttention:
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        batch_first=True,
        model_config=None,
    ):
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.device = device
        self.batch_first = batch_first
        self.params = params
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

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
        params = self.params
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = ttnn.stack([query, query], dim=1)
            value = ttnn.reshape(value, (bs * 2, len_bev, c))

        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue == 2

        query = ttnn.concat([value[:bs], query], dim=-1)

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )
        sampling_offsets = ttnn.reallocate(sampling_offsets)

        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        )

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

        if reference_points.shape[0] != bs * self.num_bev_queue:
            reference_points_list = [reference_points for _ in range(self.num_bev_queue)]
            reference_points = ttnn.concat(reference_points_list, dim=0)

        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            sampling_offsets_torch = ttnn.to_torch(sampling_offsets)
            offset_normalizer_xy_torch = ttnn.to_torch(offset_normalizer_xy)
            sampling_locations_torch = sampling_offsets_torch / offset_normalizer_xy_torch
            sampling_locations_torch = ttnn.to_torch(reference_xy) + sampling_locations_torch
            sampling_locations = ttnn.from_torch(
                sampling_locations_torch.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif reference_points.shape[-1] == 4:
            reference_points_reshape = ttnn.reshape(
                reference_points,
                [reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2],
            )
            sampling_locations = (
                reference_points_reshape + sampling_offsets / self.num_points * reference_points_reshape * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(sampling_offsets)
        ttnn.deallocate(value)
        output = ttnn.permute(output, (1, 2, 0))
        output_torch = ttnn.to_torch(output)
        ttnn.deallocate(output)
        output_torch = output_torch.reshape(num_query, embed_dims, bs, self.num_bev_queue).mean(-1)
        output = ttnn.from_torch(output_torch, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        output = ttnn.permute(output, (2, 0, 1))
        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        ttnn.deallocate(identity)
        return output
