# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TT implementation of GeometryKernelAttention and GeometrySpatialCrossAttention for MapTR."""

import torch
import ttnn


class TtGeometrySpatialCrossAttention:
    """TT implementation of Geometry-aware Spatial Cross Attention for BEVFormer.

    Args:
        device: TT device.
        params: Preprocessed model parameters.
        embed_dims (int): Embedding dimensions. Default: 256.
        num_cams (int): Number of cameras. Default: 6.
        pc_range (list): Point cloud range. Default: None.
        batch_first (bool): Whether batch is first dimension. Default: False.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_cams: int = 6,
        pc_range: list = None,
        batch_first: bool = False,
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

        # Create the geometry kernel attention
        deform_cfg = kwargs.get("attention", {})
        self.deformable_attention = TtGeometryKernelAttention(
            device=device,
            params=params,
            embed_dims=deform_cfg.get("embed_dims", embed_dims),
            num_heads=deform_cfg.get("num_heads", 4),
            num_levels=deform_cfg.get("num_levels", 1),
            kernel_size=deform_cfg.get("kernel_size", (3, 5)),
            dilation=deform_cfg.get("dilation", 1),
            batch_first=batch_first,
        )

    def __call__(
        self,
        query,
        key,
        value,
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
        """Forward function."""
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ttnn.zeros_like(query)
            slots = ttnn.to_torch(slots)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = ttnn.sum(mask_per_img[0], -1)
            index_query_per_img = ttnn.to_layout(index_query_per_img, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(3):
                index_query_per_img = ttnn.unsqueeze(index_query_per_img, 0)
            output_tensor = ttnn.nonzero(index_query_per_img, queue_id=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(index_query_per_img)

            no_of_non_zero_indices = output_tensor[0][..., 0].item()
            index_query_per_img = output_tensor[1][:, :, :, :no_of_non_zero_indices]

            for _ in range(3):
                index_query_per_img = ttnn.squeeze(index_query_per_img, 0)
            indexes.append(index_query_per_img)
            ttnn.deallocate(output_tensor[0])

        max_len = max([each.shape[0] for each in indexes])
        query = ttnn.to_torch(query)

        # Rebatch queries for each camera
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                index_query_per_img = ttnn.to_torch(index_query_per_img)
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        queries_rebatch = ttnn.from_torch(queries_rebatch, dtype=ttnn.bfloat16, device=self.device)
        reference_points_rebatch = ttnn.from_torch(reference_points_rebatch, dtype=ttnn.bfloat16, device=self.device)
        num_cams, l, bs_key, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs_key * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs_key * self.num_cams, l, self.embed_dims))

        # Apply geometry kernel attention
        queries = self.deformable_attention(
            query=ttnn.reshape(queries_rebatch, (bs * self.num_cams, max_len, self.embed_dims)),
            key=key,
            value=value,
            reference_points=ttnn.reshape(reference_points_rebatch, (bs * self.num_cams, max_len, D, 2)),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(reference_points_rebatch)

        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        queries = ttnn.to_torch(queries)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                index_query_per_img = ttnn.to_torch(index_query_per_img)
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]
        for i, index_query_per_img in enumerate(indexes):
            ttnn.deallocate(index_query_per_img)

        count = ttnn.sum(bev_mask, -1) > 0
        count = ttnn.permute(count, (1, 2, 0))
        count = ttnn.sum(count, -1)
        count = ttnn.clamp(count, min=1.0)
        count = ttnn.unsqueeze(count, -1)

        slots = ttnn.from_torch(slots, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        slots = ttnn.div(slots, count)
        slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(count)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        ttnn.deallocate(self.params.output_proj.weight)
        ttnn.deallocate(self.params.output_proj.bias)

        output = slots + inp_residual
        ttnn.deallocate(slots)
        ttnn.deallocate(inp_residual)

        return output


class TtGeometryKernelAttention:
    """TT implementation of Geometry Kernel Attention for MapTR.

    Uses fixed kernel-based sampling offsets instead of learned deformable offsets.

    Args:
        device: TT device.
        params: Preprocessed model parameters.
        embed_dims (int): Embedding dimensions. Default: 256.
        num_heads (int): Number of attention heads. Default: 4.
        num_levels (int): Number of feature levels. Default: 1.
        kernel_size (tuple): Kernel size for sampling. Default: (3, 5).
        dilation (int): Dilation for kernel. Default: 1.
        batch_first (bool): Whether batch is first dimension. Default: True.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_heads: int = 4,
        num_levels: int = 1,
        kernel_size: tuple = (3, 5),
        dilation: int = 1,
        batch_first: bool = True,
    ):
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.device = device
        self.params = params
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]

        # Create fixed grid offsets (same as PyTorch version)
        # Note: Use default indexing="ij" (not "xy") to match the checkpoint
        grid_h, grid_w = kernel_size
        y = (torch.arange(grid_h) - grid_h // 2) * dilation
        x = (torch.arange(grid_w) - grid_w // 2) * dilation
        # Default indexing="ij" gives (len(x), len(y)) shaped grids
        offsets = torch.stack(torch.meshgrid(x, y, indexing="ij")).permute(1, 2, 0).reshape(grid_h * grid_w, 2)
        self.grid_offsets = offsets.float()

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
        """Forward function."""
        params = self.params
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        # Project value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        ttnn.deallocate(params.value_proj.weight)
        ttnn.deallocate(params.value_proj.bias)

        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))

        # Compute attention weights
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(params.attention_weights.weight)
        ttnn.deallocate(params.attention_weights.bias)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        # Compute sampling locations using fixed grid offsets
        # Convert to torch for the sampling computation (complex indexing)
        value_torch = ttnn.to_torch(value)
        attention_weights_torch = ttnn.to_torch(attention_weights)
        reference_points_torch = ttnn.to_torch(reference_points)
        spatial_shapes_torch = ttnn.to_torch(spatial_shapes)

        # Use PyTorch for sampling (complex indexing operations)
        output = self._forward_pytorch(
            value_torch,
            spatial_shapes_torch,
            reference_points_torch,
            attention_weights_torch,
        )

        ttnn.deallocate(value)
        ttnn.deallocate(attention_weights)

        output = ttnn.from_torch(output, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output

    def _forward_pytorch(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch implementation of geometry kernel attention sampling.

        Args:
            value: Value tensor with shape (bs, num_keys, num_heads, dim).
            spatial_shapes: Spatial shapes of each level.
            reference_points: Reference points with shape (bs, num_queries, num_Z_anchors, 2).
            attention_weights: Attention weights with shape
                (bs, num_queries, num_heads, num_levels, num_points).

        Returns:
            Output tensor with shape (bs, num_queries, embed_dims).
        """
        bs, num_keys, num_heads, dim = value.shape
        bs, num_queries, num_Z_anchors, _ = reference_points.shape
        _, _, _, num_levels, num_points = attention_weights.shape

        # Flatten value
        value = value.transpose(1, 2).contiguous().view(bs * num_heads * num_keys, dim)

        # Compute sampling locations
        with torch.no_grad():
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            offsets = self.grid_offsets[None, None, None, None].to(reference_points.device)
            reference_points_scaled = reference_points[:, :, :, None, :] * offset_normalizer

            # Compute sampling locations: (bs, num_queries, num_Z_anchors, num_levels, num_points, 2)
            sampling_locations = (reference_points_scaled[:, :, :, :, None, :] + offsets).round().long()

            # Compute sampling indices
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
