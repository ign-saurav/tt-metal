# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
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
            # Convert to torch tensor if it's a TT tensor
            if isinstance(self.logit_scale, ttnn.Tensor):
                self.logit_scale = ttnn.to_torch(self.logit_scale)
            self.logit_scale_max = torch.log(torch.tensor(1.0 / 0.01))

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
            q_bias_torch = ttnn.to_torch(self.q_bias) if isinstance(self.q_bias, ttnn.Tensor) else self.q_bias
            v_bias_torch = ttnn.to_torch(self.v_bias) if isinstance(self.v_bias, ttnn.Tensor) else self.v_bias
            zeros = torch.zeros_like(v_bias_torch)
            qkv_bias_torch = torch.cat((q_bias_torch, zeros, v_bias_torch))
            qkv_bias_tt = ttnn.from_torch(
                qkv_bias_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )

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

        # Convert to torch for manual splitting
        qkv_torch = ttnn.to_torch(qkv)
        ttnn.deallocate(qkv)

        # Reshape: (B_, N, 3*C) -> (B_, N, 3, C)
        qkv_reshaped = qkv_torch.reshape(B_, N, 3, self.dim)

        # Split into Q, K, V: each (B_, N, C)
        q_torch, k_torch, v_torch = qkv_reshaped.chunk(3, dim=2)  # Split along dim=2
        q_torch = q_torch.squeeze(2)  # (B_, N, C)
        k_torch = k_torch.squeeze(2)  # (B_, N, C)
        v_torch = v_torch.squeeze(2)  # (B_, N, C)

        # Reshape and permute to (B_, num_heads, N, head_dim)
        q_torch = q_torch.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k_torch = k_torch.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v_torch = v_torch.reshape(B_, N, self.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        # Pad head_dim dimension if needed for TTNN compatibility
        if needs_padding:
            padding_size = padded_head_dim - head_dim
            # Pad on the last dimension (head_dim)
            q_torch = torch.nn.functional.pad(q_torch, (0, padding_size), mode="constant", value=0)
            k_torch = torch.nn.functional.pad(k_torch, (0, padding_size), mode="constant", value=0)
            v_torch = torch.nn.functional.pad(v_torch, (0, padding_size), mode="constant", value=0)
            # Store for later slicing
            self._head_dim = head_dim
            self._padded_head_dim = padded_head_dim
        else:
            self._head_dim = head_dim
            self._padded_head_dim = head_dim

        # Verify padding: last dimension should be multiple of 32
        assert q_torch.shape[-1] % 32 == 0, f"q_torch last dim {q_torch.shape[-1]} must be multiple of 32"
        assert k_torch.shape[-1] % 32 == 0, f"k_torch last dim {k_torch.shape[-1]} must be multiple of 32"
        assert v_torch.shape[-1] % 32 == 0, f"v_torch last dim {v_torch.shape[-1]} must be multiple of 32"

        # Convert back to TTNN tensors (now with padded_head_dim which is multiple of 32)
        q = ttnn.from_torch(
            q_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.memory_config,
        )
        k = ttnn.from_torch(
            k_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.memory_config,
        )
        v = ttnn.from_torch(
            v_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.memory_config,
        )

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
            k_torch = ttnn.to_torch(k)
            ttnn.deallocate(k)
            k_torch = k_torch[:, :, :, : self._head_dim]  # Slice to actual head_dim
            k = ttnn.from_torch(
                k_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )

        k_transposed = ttnn.permute(k, (0, 1, 3, 2), memory_config=self.memory_config)  # (B_, num_heads, head_dim, N)
        ttnn.deallocate(k)  # Deallocate k after transpose

        # For Q, we also need to slice if padded
        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            q_torch = ttnn.to_torch(q)
            ttnn.deallocate(q)
            q_torch = q_torch[:, :, :, : self._head_dim]  # Slice to actual head_dim
            q = ttnn.from_torch(
                q_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )

        attn = ttnn.matmul(
            q, k_transposed, memory_config=self.memory_config, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(q)  # Deallocate q after matmul
        ttnn.deallocate(k_transposed)

        # Apply logit scale
        if self.logit_scale is not None:
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            logit_scale_tt = ttnn.from_torch(
                logit_scale,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )
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
            # Slice to remove padding, then manually concatenate heads
            attn_v_torch = ttnn.to_torch(attn_v)
            ttnn.deallocate(attn_v)
            # Slice to actual head_dim: (B_, num_heads, N, padded_head_dim) -> (B_, num_heads, N, head_dim)
            attn_v_torch = attn_v_torch[:, :, :, : self._head_dim]
            # Reshape and concatenate: (B_, num_heads, N, head_dim) -> (B_, N, num_heads, head_dim) -> (B_, N, C)
            attn_v_torch = attn_v_torch.permute(0, 2, 1, 3).contiguous()  # (B_, N, num_heads, head_dim)
            attn_v_torch = attn_v_torch.reshape(B_, N, self.dim)  # (B_, N, C)
            x = ttnn.from_torch(
                attn_v_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )
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
