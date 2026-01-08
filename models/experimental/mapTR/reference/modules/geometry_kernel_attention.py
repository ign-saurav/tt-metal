# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
import torch
import torch.nn as nn
from typing import Optional


class GeometrySpatialCrossAttention(nn.Module):
    """Geometry-aware Spatial Cross Attention for BEVFormer (inference-only).

    Args:
        embed_dims (int): Embedding dimensions. Default: 256.
        num_cams (int): Number of cameras. Default: 6.
        pc_range (list): Point cloud range. Default: None.
        deformable_attention (nn.Module): Deformable attention module.
        batch_first (bool): Whether batch is first dimension. Default: False.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        pc_range: list = None,
        deformable_attention: nn.Module = None,
        batch_first: bool = False,
    ):
        super().__init__()
        self.pc_range = pc_range
        self.attention = deformable_attention
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        reference_points_cam: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            query: Query tensor with shape (bs, num_query, embed_dims).
            key: Key tensor with shape (num_cams, num_key, bs, embed_dims).
            value: Value tensor with shape (num_cams, num_key, bs, embed_dims).
            residual: Residual tensor. Default: None.
            query_pos: Query positional encoding. Default: None.
            reference_points_cam: Reference points in camera coordinates.
            bev_mask: BEV mask for valid queries per camera.
            spatial_shapes: Spatial shapes of feature maps.
            level_start_index: Start index of each level.

        Returns:
            Output tensor with shape (bs, num_query, embed_dims).
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

        # Find valid queries per camera
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # Rebatch queries for each camera
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
        key = key.permute(2, 0, 1, 3).reshape(bs_from_key * num_cams_from_key, l, embed_dims_from_key)
        value = value.permute(2, 0, 1, 3).reshape(bs_from_key * num_cams_from_key, l, embed_dims_from_key)

        # Apply deformable attention
        queries = self.attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        # Aggregate results
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return slots + inp_residual


class GeometryKernelAttention(nn.Module):
    """Geometry Kernel Attention for efficient spatial sampling (inference-only).

    Uses fixed kernel-based sampling offsets instead of learned deformable offsets.

    Args:
        embed_dims (int): Embedding dimensions. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of feature levels. Default: 4.
        kernel_size (tuple): Kernel size for sampling. Default: (3, 3).
        dilation (int): Dilation for kernel. Default: 1.
        batch_first (bool): Whether batch is first dimension. Default: True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        kernel_size: tuple = (3, 3),
        dilation: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        dim_per_head = embed_dims // num_heads
        if not (dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0:
            warnings.warn("For optimal performance, embed_dims should be set " "so that dim_per_head is a power of 2")

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]

        self.attention_weights = nn.Linear(embed_dims, num_levels * self.num_points * num_heads)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # Create fixed grid offsets
        grid_h, grid_w = kernel_size
        y = (torch.arange(grid_h) - grid_h // 2) * dilation
        x = (torch.arange(grid_w) - grid_w // 2) * dilation
        offsets = torch.stack(torch.meshgrid(x, y, indexing="xy")).permute(1, 2, 0).reshape(grid_h * grid_w, 2)
        self.register_buffer("grid_offsets", offsets.float(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            query: Query tensor with shape (bs, num_query, embed_dims).
            key: Key tensor (unused, defaults to query).
            value: Value tensor (defaults to query).
            identity: Identity tensor for residual. Default: None.
            query_pos: Query positional encoding. Default: None.
            reference_points: Reference points with shape (bs, num_query, num_Z_anchors, 2).
            spatial_shapes: Spatial shapes of feature maps.
            level_start_index: Start index of each level.

        Returns:
            Output tensor with shape (bs, num_query, embed_dims).
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
        _, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            with torch.no_grad():
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

                bs, num_query, num_Z_anchors, xy = reference_points.shape
                offsets = self.grid_offsets[None, None, None, None]
                reference_points_scaled = reference_points[:, :, :, None, :] * offset_normalizer

                # Compute sampling locations
                sampling_locations = (reference_points_scaled[:, :, :, :, None, :] + offsets).round().long()

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape

            # Use PyTorch implementation for portability
            output = self._forward_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        else:
            raise ValueError(f"Last dim of reference_points must be 2, got {reference_points.shape[-1]}")

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    def _forward_pytorch(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch implementation of geometry kernel attention.

        Args:
            value: Value tensor with shape (bs, num_keys, num_heads, dim).
            spatial_shapes: Spatial shapes of each level.
            sampling_locations: Sampling locations with shape
                (bs, num_queries, num_heads, num_levels, num_points, 2).
            attention_weights: Attention weights with shape
                (bs, num_queries, num_heads, num_levels, num_points).

        Returns:
            Output tensor with shape (bs, num_queries, embed_dims).
        """
        bs, num_keys, num_heads, dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

        # Flatten value
        value = value.transpose(1, 2).contiguous().view(bs * num_heads * num_keys, dim)

        # Compute sampling indices
        with torch.no_grad():
            sampling_index = sampling_locations.new_zeros((bs, num_queries, num_heads, num_levels, num_points)).to(
                value.device
            )

            start_index = 0
            for level, (H_, W_) in enumerate(spatial_shapes):
                H_, W_ = int(H_), int(W_)
                # Clamp to valid range
                sampling_locations[:, :, :, level, :, 0].clamp_(min=0, max=W_ - 1)
                sampling_locations[:, :, :, level, :, 1].clamp_(min=0, max=H_ - 1)
                sampling_index[:, :, :, level] = (
                    start_index
                    + sampling_locations[:, :, :, level, :, 0]
                    + sampling_locations[:, :, :, level, :, 1] * W_
                )
                start_index += H_ * W_

            # Add head and batch offsets
            sampling_index = sampling_index.transpose(1, 2).reshape(bs, num_heads, -1)
            sampling_index = sampling_index + (torch.arange(num_heads, device=sampling_index.device) * num_keys).view(
                1, num_heads, 1
            )
            sampling_index = sampling_index.reshape(bs, -1) + (
                torch.arange(bs, device=sampling_index.device) * num_keys * num_heads
            ).view(bs, 1)

        # Sample values
        sampling_value = value[sampling_index.long()].view(bs, num_heads, num_queries, num_levels * num_points, dim)

        # Apply attention weights
        attention_weights = (
            attention_weights.transpose(1, 2).contiguous().view(bs, num_heads, num_queries, num_levels * num_points, 1)
        )
        output = (sampling_value * attention_weights).sum(-2).transpose(1, 2).contiguous()

        return output.view(bs, num_queries, -1)
