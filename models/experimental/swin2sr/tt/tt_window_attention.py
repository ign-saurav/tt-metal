# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def _ttnn_normalize_l2(x, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Normalize tensor along specified dimension using L2 norm."""
    # Square the tensor
    x_squared = ttnn.multiply(x, x, memory_config=memory_config)

    # Sum along the specified dimension
    sum_squared = ttnn.sum(x_squared, dim=dim, keepdim=True, memory_config=memory_config)
    ttnn.deallocate(x_squared)

    # Add small epsilon and calculate square root
    # ttnn.add can accept a scalar directly
    sum_squared = ttnn.add(sum_squared, 1e-12, memory_config=memory_config)
    norm = ttnn.sqrt(sum_squared, memory_config=memory_config)
    ttnn.deallocate(sum_squared)

    # Divide input by norm
    normalized = ttnn.divide(x, norm, memory_config=memory_config)
    ttnn.deallocate(norm)

    return normalized


def to_2tuple(x):
    """Convert input to a tuple of 2 elements."""
    if isinstance(x, (int, float)):
        return (x, x)
    return x


class TtSwin2SRWindowAttention:
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    TT implementation of WindowAttention for Swin2SR. Supports both shifted and non-shifted windows.
    Uses SwinV2's cosine attention with log-spaced continuous position bias.

    Args:
        device: TT device.
        parameters: Model parameters dict containing qkv, proj, logit_scale, cpb_mlp weights.
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight (deprecated, not used). Default: 0.0
        proj_drop (float, optional): Dropout ratio of output (deprecated, not used). Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
        memory_config: Memory configuration for TT operations.
    """

    def __init__(
        self,
        device,
        parameters,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: tuple[int, int] = (0, 0),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.memory_config = memory_config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

        # Logit scale for cosine attention (SwinV2)
        self.logit_scale = parameters.get("logit_scale", None)
        if self.logit_scale is not None:
            self.logit_scale_max = ttnn.log(
                ttnn.full((1,), 1.0 / 0.01, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
            )

        # Relative position bias (pre-computed in preprocessor, like SwinV2)
        self.relative_position_bias = parameters.get("relative_position_bias", None)
        if self.relative_position_bias is None:
            # Fallback: compute zeros if not provided
            Wh, Ww = self.window_size
            num_heads = self.num_heads
            bias_shape = (num_heads, Wh * Ww, Wh * Ww)
            self.relative_position_bias = ttnn.zeros(
                bias_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )

        # QKV projection parameters
        self.qkv_weight = parameters["qkv"].get("weight", None)
        self.q_bias = parameters.get("q_bias", None)
        self.v_bias = parameters.get("v_bias", None)
        self.qkv_bias = qkv_bias

        # Output projection parameters
        self.proj_weight = parameters["proj"].get("weight", None)
        self.proj_bias = parameters["proj"].get("bias", None)

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor = None) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        Returns:
            Output tensor of shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # Prepare QKV bias
        qkv_bias_tt = None
        if self.qkv_bias and self.q_bias is not None:
            # Concatenate q_bias, zeros, v_bias
            zeros = ttnn.zeros_like(self.v_bias)
            qkv_bias_tt = ttnn.concat((self.q_bias, zeros, self.v_bias))

        # QKV projection with conditional memory config for large tensors
        qkv_memory_config = ttnn.L1_MEMORY_CONFIG if B_ * N * C < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=qkv_bias_tt,
            memory_config=qkv_memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)

        # Split into Q, K, V manually to support any head_dim (not just multiples of 32)
        # QKV shape: (B_, N, 3*C) where C = dim
        # We pad head_dim to nearest multiple of 32 for TTNN compatibility

        head_dim = self.dim // self.num_heads
        padded_head_dim = ((head_dim + 31) // 32) * 32  # Round up to nearest multiple of 32
        needs_padding = padded_head_dim != head_dim

        # Reshape: (B_, N, 3*C) -> (B_, N, 3, C)
        qkv_reshaped = ttnn.reshape(qkv, (B_, N, 3, self.dim))

        # Split into Q, K, V: each (B_, N, C)
        q, k, v = ttnn.chunk(qkv_reshaped, 3, dim=2)  # Split along dim=2

        # Squeeze dimension 2 from each
        q = ttnn.squeeze(q, dim=2)  # (B_, N, C)
        k = ttnn.squeeze(k, dim=2)  # (B_, N, C)
        v = ttnn.squeeze(v, dim=2)  # (B_, N, C)

        # Reshape and permute to (B_, num_heads, N, head_dim)
        q = ttnn.reshape(q, (B_, N, self.num_heads, head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))

        k = ttnn.reshape(k, (B_, N, self.num_heads, head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, (B_, N, self.num_heads, head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Pad head_dim dimension if needed for TTNN compatibility
        if needs_padding:
            padding_size = padded_head_dim - head_dim
            # Pad on the last dimension (head_dim)
            padding = ((0, 0), (0, 0), (0, 0), (0, padding_size))
            q = ttnn.pad(q, padding, value=0)
            k = ttnn.pad(k, padding, value=0)
            v = ttnn.pad(v, padding, value=0)
            # Store for later slicing
            self._head_dim = head_dim
            self._padded_head_dim = padded_head_dim
        else:
            self._head_dim = head_dim
            self._padded_head_dim = head_dim

            # Verify padding: last dimension should be multiple of 32
            assert q.shape[-1] % 32 == 0, f"q last dim {q.shape[-1]} must be multiple of 32"
            assert k.shape[-1] % 32 == 0, f"k last dim {k.shape[-1]} must be multiple of 32"
            assert v.shape[-1] % 32 == 0, f"v last dim {v.shape[-1]} must be multiple of 32"

        # Store padding info for later use in matmul operations
        self._head_dim = head_dim
        self._padded_head_dim = padded_head_dim

        # Cosine attention (SwinV2): normalize Q and K
        q = _ttnn_normalize_l2(q, dim=-1, memory_config=self.memory_config)
        k = _ttnn_normalize_l2(k, dim=-1, memory_config=self.memory_config)

        # Compute attention scores: Q @ K^T
        # Handle padding: if head_dim was padded, we need to slice k before transpose
        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            # Slice k to remove padding before transpose
            k = k[:, :, :, : self._head_dim]

        k_transposed = ttnn.permute(k, (0, 1, 3, 2), memory_config=self.memory_config)  # (B_, num_heads, head_dim, N)
        ttnn.deallocate(k)  # Deallocate k after transpose

        # For Q, we also need to slice if padded
        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            q = q[:, :, :, : self._head_dim]  # Slice to actual head_dim

        attn = ttnn.matmul(
            q, k_transposed, memory_config=self.memory_config, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(q)  # Deallocate q after matmul
        ttnn.deallocate(k_transposed)

        # Apply logit scale
        if self.logit_scale is not None:
            logit_scale = ttnn.clamp(self.logit_scale, max=self.logit_scale_max)
            logit_scale_tt = ttnn.exp(logit_scale)
            # Expand to match attn shape: (num_heads, 1, 1) -> (1, num_heads, 1, 1)
            # Reshape to (1, num_heads, 1, 1)
            logit_scale_tt = ttnn.reshape(logit_scale_tt, (1, self.num_heads, 1, 1), memory_config=self.memory_config)
            attn = ttnn.multiply(attn, logit_scale_tt, memory_config=self.memory_config)
            ttnn.deallocate(logit_scale_tt)

        # Add relative position bias (pre-computed during initialization)
        relative_position_bias_expanded = ttnn.unsqueeze(self.relative_position_bias, 0)
        attn = ttnn.add(attn, relative_position_bias_expanded, memory_config=self.memory_config)
        ttnn.deallocate(relative_position_bias_expanded)

        # Apply mask if provided
        if mask is not None:
            # mask shape: (nW, N, N), attn shape: (B_, num_heads, N, N)
            # Need to reshape attn to (B_//nW, nW, num_heads, N, N) and add mask
            nW = mask.shape[0]
            attn = ttnn.reshape(attn, (B_ // nW, nW, self.num_heads, N, N), memory_config=self.memory_config)
            # Expand mask: (nW, N, N) -> (1, nW, 1, N, N)
            mask_expanded = ttnn.unsqueeze(mask, 0)  # (1, nW, N, N)
            mask_expanded = ttnn.unsqueeze(mask_expanded, 2)  # (1, nW, 1, N, N)
            attn = ttnn.add(attn, mask_expanded, memory_config=self.memory_config)
            ttnn.deallocate(mask_expanded)
            attn = ttnn.reshape(attn, (B_, self.num_heads, N, N), memory_config=self.memory_config)

        # Softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=self.memory_config)

        # Attention output: attn @ V
        # Keep v padded for matmul (don't slice yet - matmul works with padded dimensions)
        attn_v = ttnn.matmul(
            attn, v, memory_config=self.memory_config, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # Reshape and concatenate heads: (B_, num_heads, N, head_dim) -> (B_, N, C)
        # Handle padding: if head_dim was padded, we need to slice before concatenating
        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            # Slice to remove padding, then concatenate heads - all on device
            # Slice to actual head_dim: (B_, num_heads, N, padded_head_dim) -> (B_, num_heads, N, head_dim)
            attn_v = attn_v[:, :, :, : self._head_dim]

            # Reshape and concatenate: (B_, num_heads, N, head_dim) -> (B_, N, num_heads, head_dim) -> (B_, N, C)
            attn_v = ttnn.permute(attn_v, (0, 2, 1, 3))  # (B_, N, num_heads, head_dim)
            x = ttnn.reshape(attn_v, (B_, N, self.dim))  # (B_, N, C)
        else:
            # Use ttnn concatenate_heads (head_dim is already multiple of 32)
            x = ttnn.transformer.concatenate_heads(attn_v, memory_config=self.memory_config)
            ttnn.deallocate(attn_v)

        # Output projection
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x
