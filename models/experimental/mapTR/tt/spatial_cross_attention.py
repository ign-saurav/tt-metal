# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import warnings
from models.experimental.mapTR.tt.utils import multi_scale_deformable_attn


class TtMSDeformableAttention3D:
    """TTNN implementation of MSDeformableAttention3D for BEVFormer."""

    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        batch_first=True,
    ):
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.device = device
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
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
        """Forward pass of TTNN MSDeformableAttention3D."""
        params = self.params

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        # Value projection
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)

        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))

        # Compute sampling offsets
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        # Compute attention weights
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # reference_points shape: (bs, num_query, num_Z_anchors, 2)
            # spatial_shapes is INT32, convert to float for division
            spatial_shapes_torch = ttnn.to_torch(spatial_shapes).float()
            offset_normalizer_torch = torch.stack([spatial_shapes_torch[..., 1], spatial_shapes_torch[..., 0]], dim=-1)

            # Get dimensions
            ref_shape = reference_points.shape
            bs_r, num_query_r, num_Z_anchors, xy = ref_shape

            # Normalize sampling offsets using torch (avoids dtype/broadcast issues)
            so_shape = sampling_offsets.shape
            sampling_offsets_torch = ttnn.to_torch(sampling_offsets)
            offset_normalizer_expanded = offset_normalizer_torch.view(1, 1, 1, self.num_levels, 1, 2)
            sampling_offsets_normalized_torch = sampling_offsets_torch / offset_normalizer_expanded
            sampling_offsets_normalized = ttnn.from_torch(
                sampling_offsets_normalized_torch, device=self.device, layout=ttnn.TILE_LAYOUT
            )

            # Reshape offsets for Z anchors
            num_all_points = self.num_points
            sampling_offsets_reshaped = ttnn.reshape(
                sampling_offsets_normalized,
                (bs, num_query_r, self.num_heads, self.num_levels, num_all_points // num_Z_anchors, num_Z_anchors, 2),
            )

            # Expand reference points: (bs, num_query, num_Z_anchors, 2) -> (bs, num_query, 1, 1, 1, num_Z_anchors, 2)
            reference_points_expanded = ttnn.reshape(reference_points, (bs_r, num_query_r, 1, 1, 1, num_Z_anchors, 2))

            # Add reference points to sampling offsets
            # Need to handle 7D tensors - convert to torch for this operation
            ref_pts_torch = ttnn.to_torch(reference_points_expanded)
            sampling_offsets_torch = ttnn.to_torch(sampling_offsets_reshaped)
            sampling_locations_torch = ref_pts_torch + sampling_offsets_torch

            # Reshape to final form
            sampling_locations_torch = sampling_locations_torch.view(
                bs, num_query_r, self.num_heads, self.num_levels, num_all_points, 2
            )
            sampling_locations = ttnn.from_torch(sampling_locations_torch, device=self.device, layout=ttnn.TILE_LAYOUT)
        else:
            raise ValueError(f"Last dim of reference_points must be 2, got {reference_points.shape[-1]}")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output


class TtSpatialCrossAttention:
    """TTNN implementation of SpatialCrossAttention for BEVFormer."""

    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_cams=6,
        batch_first=False,
        deformable_attention=None,
    ):
        self.device = device
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        self.params = params
        self.deformable_attention = deformable_attention

        dim_per_head = embed_dims // 8  # Assuming 8 heads
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
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward pass of TTNN SpatialCrossAttention."""
        params = self.params

        # Handle key/value defaults
        if key is None:
            key = query
        if value is None:
            value = key

        # Handle residual
        if residual is None:
            inp_residual = query
            slots = ttnn.zeros_like(query)

        # Add positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        bs, num_query, _ = query.shape
        D = reference_points_cam.shape[3]

        # Convert to torch for index processing
        # bev_mask may already be a torch tensor (no ttnn.bool dtype)
        bev_mask_torch = bev_mask if isinstance(bev_mask, torch.Tensor) else ttnn.to_torch(bev_mask)
        query_torch = ttnn.to_torch(query)
        reference_points_cam_torch = ttnn.to_torch(reference_points_cam)

        # Find valid queries per camera
        indexes = []
        for i, mask_per_img in enumerate(bev_mask_torch):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # Create rebatched queries and reference points
        queries_rebatch = query_torch.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam_torch.new_zeros([bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam_torch):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query_torch[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        # Convert back to TTNN
        queries_rebatch = ttnn.from_torch(queries_rebatch, device=self.device, layout=ttnn.TILE_LAYOUT)
        reference_points_rebatch = ttnn.from_torch(
            reference_points_rebatch, device=self.device, layout=ttnn.TILE_LAYOUT
        )

        # Reshape key and value for multi-camera processing
        num_cams, l, bs, embed_dims = key.shape
        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))
        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))

        # Apply deformable attention
        queries = self.deformable_attention(
            query=ttnn.reshape(queries_rebatch, (bs * self.num_cams, max_len, self.embed_dims)),
            key=key,
            value=value,
            reference_points=ttnn.reshape(reference_points_rebatch, (bs * self.num_cams, max_len, D, 2)),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        # Convert to torch for aggregation
        queries_torch = ttnn.to_torch(queries)
        slots_torch = ttnn.to_torch(slots)

        # Aggregate results per camera
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots_torch[j, index_query_per_img] += queries_torch[j, i, : len(index_query_per_img)]

        # Normalize by count
        bev_mask_torch = bev_mask_torch.sum(-1) > 0
        bev_mask_torch = bev_mask_torch.permute(1, 2, 0).sum(-1)
        bev_mask_torch = torch.clamp(bev_mask_torch, min=1.0)
        slots_torch = slots_torch / bev_mask_torch[..., None]

        # Convert back to TTNN
        slots = ttnn.from_torch(slots_torch, device=self.device, layout=ttnn.TILE_LAYOUT)

        # Apply output projection
        slots = ttnn.linear(slots, params.output_proj.weight, bias=params.output_proj.bias)

        # Add residual connection
        output = ttnn.add(slots, inp_residual)

        return output
