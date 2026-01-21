# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def _ttnn_normalize_l2(x, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Normalize tensor along specified dimension using L2 norm."""
    x_squared = ttnn.multiply(x, x, memory_config=memory_config)
    sum_squared = ttnn.sum(x_squared, dim=dim, keepdim=True, memory_config=memory_config)
    ttnn.deallocate(x_squared)
    sum_squared = ttnn.add(sum_squared, 1e-12, memory_config=memory_config)
    norm = ttnn.sqrt(sum_squared, memory_config=memory_config)
    ttnn.deallocate(sum_squared)
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        self.l1 = ttnn.L1_MEMORY_CONFIG
        self.dram = ttnn.DRAM_MEMORY_CONFIG
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

        self.logit_scale = parameters.get("logit_scale", None)
        if self.logit_scale is not None:
            self.logit_scale_max = ttnn.log(
                ttnn.full((1,), 1.0 / 0.01, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
            )

        self.relative_position_bias = parameters.get("relative_position_bias", None)
        if self.relative_position_bias is None:
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

        self.qkv_weight = parameters["qkv"].get("weight", None)
        self.q_bias = parameters.get("q_bias", None)
        self.v_bias = parameters.get("v_bias", None)
        self.qkv_bias = qkv_bias

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

        qkv_bias_tt = None
        if self.qkv_bias and self.q_bias is not None:
            zeros = ttnn.zeros_like(self.v_bias)
            qkv_bias_tt = ttnn.concat((self.q_bias, zeros, self.v_bias))

        qkv_size = B_ * N * C * 3
        qkv_memory_config = self.l1 if qkv_size < 1_000_000 else self.dram
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=qkv_bias_tt,
            memory_config=qkv_memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)

        head_dim = self.dim // self.num_heads
        padded_head_dim = ((head_dim + 31) // 32) * 32
        needs_padding = padded_head_dim != head_dim

        qkv_reshaped = ttnn.reshape(qkv, (B_, N, 3, self.dim))
        q, k, v = ttnn.chunk(qkv_reshaped, 3, dim=2)
        q = ttnn.squeeze(q, dim=2)
        k = ttnn.squeeze(k, dim=2)
        v = ttnn.squeeze(v, dim=2)
        q = ttnn.reshape(q, (B_, N, self.num_heads, head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))

        k = ttnn.reshape(k, (B_, N, self.num_heads, head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, (B_, N, self.num_heads, head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        if needs_padding:
            padding_size = padded_head_dim - head_dim
            padding = ((0, 0), (0, 0), (0, 0), (0, padding_size))
            q = ttnn.pad(q, padding, value=0)
            k = ttnn.pad(k, padding, value=0)
            v = ttnn.pad(v, padding, value=0)
            self._head_dim = head_dim
            self._padded_head_dim = padded_head_dim
        else:
            self._head_dim = head_dim
            self._padded_head_dim = head_dim
            assert q.shape[-1] % 32 == 0, f"q last dim {q.shape[-1]} must be multiple of 32"
            assert k.shape[-1] % 32 == 0, f"k last dim {k.shape[-1]} must be multiple of 32"
            assert v.shape[-1] % 32 == 0, f"v last dim {v.shape[-1]} must be multiple of 32"

        self._head_dim = head_dim
        self._padded_head_dim = padded_head_dim

        q_size = B_ * self.num_heads * N * head_dim
        qk_memory_config = self.l1 if q_size < 1_000_000 else self.dram

        q = _ttnn_normalize_l2(q, dim=-1, memory_config=qk_memory_config)
        k = _ttnn_normalize_l2(k, dim=-1, memory_config=qk_memory_config)

        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            k = k[:, :, :, : self._head_dim]

        k_transposed = ttnn.permute(k, (0, 1, 3, 2), memory_config=qk_memory_config)
        ttnn.deallocate(k)

        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            q = q[:, :, :, : self._head_dim]

        attn_size = B_ * self.num_heads * N * N
        attn_memory_config = self.l1 if attn_size < 1_000_000 else self.dram
        attn = ttnn.matmul(
            q, k_transposed, memory_config=attn_memory_config, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_transposed)

        if self.logit_scale is not None:
            logit_scale = ttnn.clamp(self.logit_scale, max=self.logit_scale_max)
            logit_scale_tt = ttnn.exp(logit_scale)
            logit_scale_tt = ttnn.reshape(logit_scale_tt, (1, self.num_heads, 1, 1), memory_config=attn_memory_config)
            attn = ttnn.multiply(attn, logit_scale_tt, memory_config=attn_memory_config)
            ttnn.deallocate(logit_scale_tt)

        relative_position_bias_expanded = ttnn.unsqueeze(self.relative_position_bias, 0)
        attn = ttnn.add(attn, relative_position_bias_expanded, memory_config=attn_memory_config)
        ttnn.deallocate(relative_position_bias_expanded)

        if mask is not None:
            nW = mask.shape[0]
            attn = ttnn.reshape(attn, (B_ // nW, nW, self.num_heads, N, N), memory_config=attn_memory_config)
            mask_expanded = ttnn.unsqueeze(mask, 0)
            mask_expanded = ttnn.unsqueeze(mask_expanded, 2)
            attn = ttnn.add(attn, mask_expanded, memory_config=attn_memory_config)
            ttnn.deallocate(mask_expanded)
            attn = ttnn.reshape(attn, (B_, self.num_heads, N, N), memory_config=attn_memory_config)

        attn = ttnn.softmax(attn, dim=-1, memory_config=attn_memory_config)

        v_size = B_ * self.num_heads * N * head_dim
        v_memory_config = self.l1 if v_size < 1_000_000 else self.dram
        attn_v = ttnn.matmul(
            attn, v, memory_config=attn_memory_config, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        if hasattr(self, "_padded_head_dim") and self._padded_head_dim != self._head_dim:
            attn_v = attn_v[:, :, :, : self._head_dim]
            attn_v = ttnn.permute(attn_v, (0, 2, 1, 3))
            x = ttnn.reshape(attn_v, (B_, N, self.dim))
        else:
            x = ttnn.transformer.concatenate_heads(attn_v, memory_config=self.memory_config)
            ttnn.deallocate(attn_v)

        proj_size = B_ * N * self.dim
        proj_memory_config = self.l1 if proj_size < 1_000_000 else self.dram
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=proj_memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x
