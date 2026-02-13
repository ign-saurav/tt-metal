# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TT implementation of SAM ImageEncoderViT (ViT-B).
Uses the same ttnn ops as tt_transformers multimodal image attention (no edits to that module):
  ttnn.linear, ttnn.experimental.nlp_create_qkv_heads, ttnn.transformer.scaled_dot_product_attention,
  ttnn.experimental.nlp_concat_heads. All layers are ttnn: conv2d, layer_norm, linear, gelu, SDPA.
"""

from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import ttnn
from models.common.lightweightmodule import LightweightModule


# --------------- Relative position bias (mirrors deepencoder.get_rel_pos / add_decomposed_rel_pos) ---------------


def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor, device: Optional[ttnn.Device] = None) -> torch.Tensor:
    """Relative position table indexed by (q_coord - k_coord). Returns (q_size, k_size, head_dim)."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolation only runs when grid != pretrained size (e.g. 640 gives grid 40 → max_rel_dist 79; rel_pos is 127 so we downsample). For 1024 (grid 64) rel_pos already has 127 entries so we skip interpolate.
        # Keep linear interpolation on host for PCC match (ttnn.upsample has nearest/bilinear only; nearest changes values)
        rel_pos = rel_pos.float()
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(rel_pos.dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    device: Optional[ttnn.Device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q: (B, n_heads, q_h*q_w, head_dim). Returns rel_h, rel_w for building attn_bias."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos(q_h, k_h, rel_pos_h, device=device)
    Rw = _get_rel_pos(q_w, k_w, rel_pos_w, device=device)
    B, n_heads, S, dim = q.shape
    r_q = q.reshape(B, n_heads, q_h, q_w, dim)
    rel_h = torch.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
    rel_w = torch.einsum("bnhwc,wkc->bnhwk", r_q, Rw)
    rel_h = rel_h.reshape(B, n_heads, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, n_heads, q_h * q_w, 1, k_w)
    return rel_h, rel_w


def compute_sam_attn_bias(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    spatial_size: Tuple[int, int],
    device: Optional[ttnn.Device] = None,
) -> torch.Tensor:
    """q: (1, n_heads, S, head_dim). Returns attn_bias (1, n_heads, S, S) for SDPA."""
    H, W = spatial_size
    S = H * W
    dtype = q.dtype
    q = q.float()
    rel_pos_h = rel_pos_h.float()
    rel_pos_w = rel_pos_w.float()
    rel_h, rel_w = _add_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, (H, W), (H, W), device=device)
    attn_bias = (rel_h + rel_w).reshape(1, q.shape[1], S, S).to(dtype)
    return attn_bias


# Same attention building blocks as tt_transformers (we call these ttnn APIs directly)
# Reference: models.tt_transformers.tt.multimodal.llama_image_attention.TtLlamaImageAttention.forward
# Uses: ttnn.linear, nlp_create_qkv_heads, scaled_dot_product_attention, nlp_concat_heads
#
# Layout requirements (cross-checked with ttnn APIs):
# - ttnn.conv2d: weight_tensor and bias_tensor must be ROW_MAJOR_LAYOUT (host conv).
# - ttnn.linear: weight and bias in TILE_LAYOUT; input in TILE_LAYOUT.
# - ttnn.layer_norm: weight and bias in TILE_LAYOUT.
# - run_tt_sam input: from_torch(..., TILE_LAYOUT) so first conv receives TILE; conv output stays device layout.


# --------------- Attention (same pattern as tt_transformers TtLlamaImageAttention) ---------------


class TtSamAttention(LightweightModule):
    """
    SAM self-attention using the same TT path as tt_transformers image attention:
    fused QKV linear -> nlp_create_qkv_heads -> scaled_dot_product_attention -> nlp_concat_heads -> output linear.
    Optional relative position bias: computed on host from q and rel_pos_h/w, then passed as attn_mask to SDPA.
    """

    def __init__(
        self,
        device: ttnn.Device,
        dim: int,
        num_heads: int,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        proj_weight: torch.Tensor,
        proj_bias: Optional[torch.Tensor],
        rel_pos_h: Optional[torch.Tensor] = None,
        rel_pos_w: Optional[torch.Tensor] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rel_pos = rel_pos_h is not None and rel_pos_w is not None and spatial_size is not None
        self._rel_pos_h = rel_pos_h
        self._rel_pos_w = rel_pos_w
        self._spatial_size = spatial_size
        if self.use_rel_pos:
            self._qkv_weight_torch = qkv_weight.detach().float()
            self._qkv_bias_torch = qkv_bias.detach().float() if qkv_bias is not None else None

        # SAM qkv_weight (dim*3, dim). .T -> (dim, dim*3); store (1, 1, 768, 2304) for matmul act (1,1,S,768) @ (768,2304)
        wqkv = qkv_weight.detach().float().T.unsqueeze(0).unsqueeze(0)  # (1, 1, 768, 2304)
        self.wqkv = ttnn.from_torch(
            wqkv,
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if qkv_bias is not None:
            b = qkv_bias.detach().float().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,dim*3)
            self.wqkv_bias = ttnn.from_torch(
                b,
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.wqkv_bias = None

        # proj (dim, dim); store (1, 1, dim, dim) as (1,1,768,768); matmul with transpose_b=True
        wproj = proj_weight.detach().float().T.unsqueeze(0).unsqueeze(0)  # (1, 1, 768, 768)
        self.wo = ttnn.from_torch(
            wproj,
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if proj_bias is not None:
            self.bo = ttnn.from_torch(
                proj_bias.detach().float().unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bo = None

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Match tt_transformers llama_image_attention SDPA: separate config with fp32_dest_acc_en=True
        # for better numerics in attention scores and weighted sum (model_config.py compute_kernel_config_sdpa)
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Same SDPAProgramConfig as tt_transformers (q_chunk_size=32*max(1,num_chunks); we use 32 for window 196)
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

    def _get_attn_bias(
        self,
        x_11SH: ttnn.Tensor,
        seq_len_orig: Optional[int] = None,
        seq_len_padded: Optional[int] = None,
    ) -> Optional[ttnn.Tensor]:
        """Compute rel_pos attn_bias; use TT slice + matmul + create_qkv_heads for q, then host for bias formula."""
        if not self.use_rel_pos:
            return None
        dim = x_11SH.shape[-1]
        if seq_len_orig is not None:
            x_for_bias = x_11SH[(slice(0, 1), slice(0, 1), slice(0, seq_len_orig), slice(0, dim))]
        else:
            x_for_bias = x_11SH
        qkv_tt = ttnn.matmul(
            x_for_bias,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        if seq_len_orig is not None:
            ttnn.deallocate(x_for_bias)
        if self.wqkv_bias is not None:
            qkv_tt = ttnn.add(qkv_tt, self.wqkv_bias)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            qkv_tt,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_tt)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)
        q_t = ttnn.to_torch(q_heads)
        ttnn.deallocate(q_heads)
        if q_t.device.type != "cpu":
            q_t = q_t.cpu()
        S = q_t.shape[2]
        attn_bias = compute_sam_attn_bias(
            q_t.float(), self._rel_pos_h, self._rel_pos_w, self._spatial_size, device=self.device
        )
        attn_bias = attn_bias.to(torch.bfloat16)
        if seq_len_padded is not None and seq_len_padded > S:
            full_bias = torch.full(
                (1, self.n_heads, seq_len_padded, seq_len_padded),
                float("-inf"),
                dtype=torch.bfloat16,
                device=attn_bias.device,
            )
            full_bias[:, :, :S, :S] = attn_bias
            attn_bias = full_bias
        return ttnn.from_torch(
            attn_bias,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x_11SH: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: (1, 1, seq_len, dim). Returns (1, 1, seq_len, dim).
        When seq_len is not divisible by 32, input is padded to a multiple of 32 for SDPA, then output is sliced back.
        """
        seq_len = x_11SH.shape[-2]
        pad_seq = ((seq_len + 31) // 32) * 32
        need_pad = pad_seq > seq_len

        if need_pad:
            # Pad sequence dim to multiple of 32 for SDPA (ttnn.pad: padding (before, after) per dim; value=0)
            padding = ((0, 0), (0, 0), (0, pad_seq - seq_len), (0, 0))
            x_work = ttnn.pad(
                x_11SH,
                padding=padding,
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            x_work = x_11SH

        attn_mask_tt = None
        if self.use_rel_pos:
            attn_mask_tt = self._get_attn_bias(
                x_work, seq_len_orig=seq_len if need_pad else None, seq_len_padded=pad_seq if need_pad else None
            )

        # input (1,1,S,768) @ weight (1,1,768,2304) -> (1,1,S,2304)
        xqkv_fused = ttnn.matmul(
            x_work,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        if need_pad:
            ttnn.deallocate(x_work)
        if self.wqkv_bias is not None:
            xqkv_fused = ttnn.add(xqkv_fused, self.wqkv_bias)

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # Manual SDPA: Q@K^T -> scale -> +mask -> softmax -> @V (scale_mask_softmax requires mask height 1 or 32, so we do add+softmax separately)
        qk = ttnn.matmul(
            q_heads,
            k_heads,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        qk = ttnn.multiply(qk, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if attn_mask_tt is not None:
            qk = ttnn.add(qk, attn_mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_mask_tt)
        attn_weights = ttnn.softmax(
            qk, dim=-1, compute_kernel_config=self.compute_kernel_config_sdpa, numeric_stable=True
        )
        ttnn.deallocate(qk)
        attn_out = ttnn.matmul(
            attn_weights,
            v_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )
        ttnn.deallocate(attn_weights)
        ttnn.deallocate(v_heads)

        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        # attn_concat (1,1,S,768) @ wo (1,1,768,768) -> (1,1,S,768); do slice after proj when padded
        output = ttnn.matmul(
            attn_concat,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        if self.bo is not None:
            output = ttnn.add(output, self.bo)
        if need_pad:
            output = output[(slice(0, 1), slice(0, 1), slice(0, seq_len), slice(0, output.shape[-1]))]
        ttnn.deallocate(attn_concat)
        return output


# --------------- LayerNorm (ttnn.layer_norm) ---------------


class TtSamLayerNorm(LightweightModule):
    def __init__(
        self,
        device: ttnn.Device,
        dim: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-6,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        w = weight.detach().float().unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        b = bias.detach().float().unsqueeze(0).unsqueeze(0)
        self.weight = ttnn.from_torch(
            w, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bias = ttnn.from_torch(
            b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x, weight=self.weight, bias=self.bias, epsilon=self.eps, compute_kernel_config=self.compute_config
        )


# --------------- MLP (linear -> gelu -> linear) ---------------


class TtSamMLP(LightweightModule):
    def __init__(
        self,
        device: ttnn.Device,
        dim: int,
        mlp_dim: int,
        lin1_weight: torch.Tensor,
        lin1_bias: Optional[torch.Tensor],
        lin2_weight: torch.Tensor,
        lin2_bias: Optional[torch.Tensor],
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # (in, out) for ttnn.linear: act (..., dim) @ weight (dim, mlp_dim); torch lin1 is (mlp_dim, dim) so use .T
        self.w1 = ttnn.from_torch(
            lin1_weight.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.b1 = (
            ttnn.from_torch(
                lin1_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if lin1_bias is not None
            else None
        )
        self.w2 = ttnn.from_torch(
            lin2_weight.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.b2 = (
            ttnn.from_torch(
                lin2_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if lin2_bias is not None
            else None
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation="gelu",
        )
        return ttnn.linear(
            x,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# --------------- Window partition / unpartition (mirrors deepencoder, torch for host) ---------------


def _window_partition_torch(x_bhwc: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """x_bhwc: (B, H, W, C). Returns windows (B*num_windows, window_size, window_size, C), (Hp, Wp)."""
    B, H, W, C = x_bhwc.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x_bhwc = F.pad(x_bhwc, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x_bhwc.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def _window_unpartition_torch(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """windows: (B*num_windows, window_size, window_size, C). Returns (B, H, W, C)."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def _window_partition_unpartition_tt(
    x_11SH: ttnn.Tensor,
    device: ttnn.Device,
    H: int,
    W: int,
    window_size: int,
    C: int,
    run_attn_per_window,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Partition (1,1,S,C) into windows on TT, run run_attn_per_window(win_11SW) per window, unpartition. Uses ROW_MAJOR for reshape/permute (non-32 dims)."""
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    Hp, Wp = H + pad_h, W + pad_w
    S_padded = Hp * Wp

    if pad_h > 0 or pad_w > 0:
        x_t = ttnn.to_torch(x_11SH)
        if x_t.device.type != "cpu":
            x_t = x_t.cpu()
        x_t = x_t.reshape(1, H, W, C)
        x_t = F.pad(x_t, (0, 0, 0, pad_w, 0, pad_h))
        x_t = x_t.reshape(1, S_padded, C)
        x_11SH = ttnn.from_torch(
            x_t.unsqueeze(1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

    # ROW_MAJOR so we can reshape to (1, Hp, Wp, C) with Hp, Wp not multiples of 32
    x_rm = ttnn.to_layout(x_11SH, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(x_11SH)
    x_rm = ttnn.reshape(x_rm, (1, Hp, Wp, C))

    nH, nW = Hp // window_size, Wp // window_size
    ws = window_size
    # (1, Hp, Wp, C) -> (1, nH, ws, nW, ws, C)
    x_rm = ttnn.reshape(x_rm, (1, nH, ws, nW, ws, C))
    # (1, nH, nW, ws, ws, C)
    x_rm = ttnn.permute(x_rm, (0, 1, 3, 2, 4, 5))
    # (1, num_windows, ws*ws, C)
    num_windows = nH * nW
    x_rm = ttnn.reshape(x_rm, (1, num_windows, ws * ws, C))

    out_list = []
    for i in range(num_windows):
        win = ttnn.slice(
            x_rm,
            (0, i, 0, 0),
            (1, i + 1, ws * ws, C),
            memory_config=memory_config,
        )
        win = ttnn.reshape(win, (1, 1, ws * ws, C))
        win = ttnn.to_layout(win, ttnn.TILE_LAYOUT, memory_config=memory_config)
        out_i = run_attn_per_window(win)
        ttnn.deallocate(win)
        out_i = ttnn.to_layout(out_i, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        out_list.append(out_i)
    ttnn.deallocate(x_rm)

    out_cat = ttnn.concat(out_list, dim=1, memory_config=memory_config)
    for t in out_list:
        ttnn.deallocate(t)
    # (1, nH, nW, ws, ws, C)
    out_cat = ttnn.reshape(out_cat, (1, nH, nW, ws, ws, C))
    out_cat = ttnn.permute(out_cat, (0, 1, 3, 2, 4, 5))
    out_cat = ttnn.reshape(out_cat, (1, Hp, Wp, C))
    if Hp > H or Wp > W:
        out_cat = ttnn.slice(
            out_cat,
            (0, 0, 0, 0),
            (1, H, W, C),
            memory_config=memory_config,
        )
    out_cat = ttnn.reshape(out_cat, (1, 1, H * W, C))
    out_tt = ttnn.to_layout(out_cat, ttnn.TILE_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(out_cat)
    return out_tt


# --------------- Block (norm -> attention -> add -> norm -> mlp -> add) ---------------


class TtSamBlock(LightweightModule):
    def __init__(
        self,
        device: ttnn.Device,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        attn_module: TtSamAttention,
        norm1: TtSamLayerNorm,
        norm2: TtSamLayerNorm,
        mlp: TtSamMLP,
        window_size: int = 0,
        grid_size: int = 40,
    ):
        super().__init__()
        self.device = device
        self.norm1 = norm1
        self.attn = attn_module
        self.norm2 = norm2
        self.mlp = mlp
        self.window_size = window_size
        self.grid_size = grid_size

    def forward(self, x_11SH: ttnn.Tensor) -> ttnn.Tensor:
        normed = self.norm1(x_11SH)
        if self.window_size > 0:
            attn_out = self._forward_window_attention(normed)
        else:
            attn_out = self.attn(normed)
        ttnn.deallocate(normed)
        res = ttnn.add(x_11SH, attn_out)
        ttnn.deallocate(attn_out)
        norm2_out = self.norm2(res)
        mlp_out = self.mlp(norm2_out)
        out = ttnn.add(res, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(res)
        return out

    def _forward_window_attention(self, x_11SH: ttnn.Tensor) -> ttnn.Tensor:
        """Partition x into windows on TT, run TT attention per window, unpartition on TT. x: (1, 1, S, C)."""
        H = W = self.grid_size
        window_size = self.window_size
        C = x_11SH.shape[-1]
        return _window_partition_unpartition_tt(
            x_11SH,
            device=self.device,
            H=H,
            W=W,
            window_size=window_size,
            C=C,
            run_attn_per_window=self.attn,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# --------------- PatchEmbed (ttnn.conv2d) ---------------


class TtPatchEmbed(LightweightModule):
    def __init__(
        self,
        device: ttnn.Device,
        in_chans: int,
        embed_dim: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        batch_size: int,
        input_height: int,
        input_width: int,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        # ttnn.conv2d requires host conv weights/bias in ROW_MAJOR; do not pass device so tensors stay on host
        self.weight = ttnn.from_torch(
            weight.detach().float(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.bias = (
            ttnn.from_torch(
                bias.detach().float().unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if bias is not None
            else None
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=None,
            deallocate_activation=False,
            reshard_if_not_optimal=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Support dynamic input size (e.g. neck uses 0,0 and gets from x)
        if len(x.shape) == 4:
            in_ch = x.shape[-1]
            in_h = self.input_height or x.shape[1]
            in_w = self.input_width or x.shape[2]
            batch = x.shape[0]
        else:
            in_ch = self.in_chans
            in_h = self.input_height
            in_w = self.input_width
            batch = self.batch_size
        out, (oh, ow), _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=in_ch,
            out_channels=self.embed_dim,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            batch_size=batch,
            input_height=in_h,
            input_width=in_w,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return out


# --------------- Generic TtConv2d (for neck and net_2/net_3) ---------------


class TtConv2d(LightweightModule):
    """Single TT conv2d layer; input dimensions taken from tensor in forward."""

    def __init__(
        self,
        device: ttnn.Device,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # ttnn.conv2d requires host conv weights/bias in ROW_MAJOR; do not pass device so tensors stay on host
        self.weight = ttnn.from_torch(
            weight.detach().float(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.bias = (
            ttnn.from_torch(
                bias.detach().float().unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if bias is not None
            else None
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=None,
            deallocate_activation=False,
            reshard_if_not_optimal=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # ttnn.conv2d expects input [N, H, W, C] (NHWC)
        if len(x.shape) == 4:
            batch, in_h, in_w, in_ch = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        else:
            batch, in_h, in_w, in_ch = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        out, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch,
            input_height=in_h,
            input_width=in_w,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return out


# --------------- LayerNorm2d (channel-wise: reshape B,C,H,W -> B*H*W,C, layer_norm, reshape back) ---------------


class TtLayerNorm2d(LightweightModule):
    def __init__(
        self,
        device: ttnn.Device,
        num_channels: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-6,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_channels = num_channels
        self.eps = eps
        w = weight.detach().float().unsqueeze(0).unsqueeze(0)
        b = bias.detach().float().unsqueeze(0).unsqueeze(0)
        self.weight = ttnn.from_torch(
            w, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bias = ttnn.from_torch(
            b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def forward(self, x_bchw: ttnn.Tensor) -> ttnn.Tensor:
        """x_bchw: (B, C, H, W). Normalize over C per spatial position."""
        shape = x_bchw.shape
        if len(shape) == 4:
            b, c, h, w = shape
            x_flat = ttnn.reshape(x_bchw, (b * h * w, c))
        else:
            x_flat = x_bchw
        out = ttnn.layer_norm(
            x_flat,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_config,
        )
        if len(shape) == 4:
            out = ttnn.reshape(out, (b, h, w, c))
            out = ttnn.permute(out, (0, 3, 1, 2))
        return out


# --------------- Top-level encoder ---------------


class TtImageEncoderViT(LightweightModule):
    """
    TT SAM ImageEncoderViT: patch_embed -> pos_embed -> blocks -> neck -> net_2 -> net_3.
    All layers are TT (ttnn) ops. Attention uses the same pattern as tt_transformers
    TtLlamaImageAttention (linear -> create_qkv_heads -> scaled_dot_product_attention -> concat_heads -> linear).
    """

    def __init__(
        self,
        device: ttnn.Device,
        sam_torch_module: torch.nn.Module,
        batch_size: int = 1,
        image_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.batch_size = batch_size

        # Patch embed
        proj = sam_torch_module.patch_embed.proj
        self.patch_embed = TtPatchEmbed(
            device=device,
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            weight=proj.weight,
            bias=proj.bias,
            batch_size=batch_size,
            input_height=image_size,
            input_width=image_size,
            dtype=dtype,
        )

        # Pos embed (optional add; keep on host and add after patch_embed to TT tensor when needed)
        if getattr(sam_torch_module, "pos_embed", None) is not None:
            self.pos_embed = sam_torch_module.pos_embed.detach()
            # Use same get_abs_pos_sam as reference (deepencoder) for identical interpolation
            deepencoder = importlib.import_module(type(sam_torch_module).__module__)
            self._get_abs_pos_sam = getattr(deepencoder, "get_abs_pos_sam", None)
        else:
            self.pos_embed = None
            self._get_abs_pos_sam = None

        # Blocks (SAM ViT-B: window_size=14 for most blocks, 0 for global-attn blocks 2,5,8,11)
        grid_size = self.grid_size
        mlp_dim = int(embed_dim * mlp_ratio)
        self.blocks = []
        self._blocks_torch = None
        for i in range(depth):
            blk = sam_torch_module.blocks[i]
            win_sz = getattr(blk, "window_size", 0)
            spatial_size = (win_sz, win_sz) if win_sz else (grid_size, grid_size)
            rel_pos_h = getattr(blk.attn, "rel_pos_h", None)
            rel_pos_w = getattr(blk.attn, "rel_pos_w", None)
            attn = TtSamAttention(
                device=device,
                dim=embed_dim,
                num_heads=num_heads,
                qkv_weight=blk.attn.qkv.weight,
                qkv_bias=blk.attn.qkv.bias,
                proj_weight=blk.attn.proj.weight,
                proj_bias=blk.attn.proj.bias,
                rel_pos_h=rel_pos_h,
                rel_pos_w=rel_pos_w,
                spatial_size=spatial_size if (rel_pos_h is not None and rel_pos_w is not None) else None,
                dtype=dtype,
            )
            norm1 = TtSamLayerNorm(device, embed_dim, blk.norm1.weight, blk.norm1.bias, eps=blk.norm1.eps, dtype=dtype)
            norm2 = TtSamLayerNorm(device, embed_dim, blk.norm2.weight, blk.norm2.bias, eps=blk.norm2.eps, dtype=dtype)
            mlp = TtSamMLP(
                device=device,
                dim=embed_dim,
                mlp_dim=mlp_dim,
                lin1_weight=blk.mlp.lin1.weight,
                lin1_bias=blk.mlp.lin1.bias,
                lin2_weight=blk.mlp.lin2.weight,
                lin2_bias=blk.mlp.lin2.bias,
                dtype=dtype,
            )
            self.blocks.append(
                TtSamBlock(
                    device,
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    attn,
                    norm1,
                    norm2,
                    mlp,
                    window_size=win_sz,
                    grid_size=grid_size,
                )
            )

        # Neck and head convs: run on PyTorch (CPU) to avoid device L1 OOM after 12 blocks
        self.neck_torch = sam_torch_module.neck
        self.net_2_torch = sam_torch_module.net_2
        self.net_3_torch = sam_torch_module.net_3

    @staticmethod
    def _conv_from_module(device: ttnn.Device, conv: torch.nn.Conv2d, dtype) -> TtConv2d:
        """Build TtConv2d from torch Conv2d."""
        k = conv.kernel_size
        s = conv.stride
        p = conv.padding
        if isinstance(p, int):
            p = (p, p)
        return TtConv2d(
            device=device,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=k if isinstance(k, tuple) else (k, k),
            stride=s if isinstance(s, tuple) else (s, s),
            padding=p,
            weight=conv.weight,
            bias=conv.bias,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: (B, C, H, W) in TT layout. Returns final feature map from net_3."""
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = ttnn.permute(x, (0, 2, 3, 1))
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # Pos: only interpolate on host (bicubic); add + reshape on TT.
            grid = self.grid_size
            b, h, w, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            if self._get_abs_pos_sam is not None:
                pos_torch = self._get_abs_pos_sam(self.pos_embed, grid)
            else:
                pos_torch = self.pos_embed.detach()
                if pos_torch.shape[1] != grid or pos_torch.shape[2] != grid:
                    pos_torch = pos_torch.permute(0, 3, 1, 2).to(torch.float32)
                    pos_torch = (
                        torch.nn.functional.interpolate(
                            pos_torch,
                            size=(grid, grid),
                            mode="bicubic",
                            antialias=True,
                            align_corners=False,
                        )
                        .to(torch.bfloat16)
                        .permute(0, 2, 3, 1)
                    )
            pos_torch = pos_torch.expand(b, grid, grid, c).clone().to(torch.bfloat16)
            pos_tt = ttnn.from_torch(
                pos_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # x on device: ensure (1, grid, grid, c) for add (use ROW_MAJOR for non-32 dims)
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            if (h, w) != (grid, grid) and h * w == grid * grid:
                x_rm = ttnn.reshape(x_rm, (1, grid, grid, c))
            x = ttnn.add(x_rm, pos_tt)
            ttnn.deallocate(x_rm)
            ttnn.deallocate(pos_tt)
            x = ttnn.reshape(x, (1, 1, grid * grid, c))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            b, h, w, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            x = ttnn.reshape(x, (1, 1, b * h * w, c))
        for blk in self.blocks:
            x = blk(x)
        # fallback: neck + net_2 + net_3 in PyTorch to avoid device L1 OOM
        x_t = ttnn.to_torch(x)
        ttnn.deallocate(x)
        grid = self.grid_size
        x_t = x_t.reshape(1, grid, grid, c).permute(0, 3, 1, 2)
        with torch.no_grad():
            for layer in self.neck_torch:
                x_t = layer(x_t)
            x_t = self.net_2_torch(x_t)
            x_t = self.net_3_torch(x_t)
        x_t = x_t.contiguous().to(torch.bfloat16)
        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16)
        return x


# --------------- TT SAM run ---------------


def run_tt_sam(
    device: ttnn.Device,
    sam_torch_module: torch.nn.Module,
    input_tensor: torch.Tensor,
    batch_size: int = 1,
    image_size: int = 1024,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """Run TT SAM image encoder. input_tensor: (B, 3, H, W). Returns TT tensor (same shape as torch SAM)."""
    tt_model = TtImageEncoderViT(
        device=device,
        sam_torch_module=sam_torch_module,
        batch_size=batch_size,
        image_size=image_size,
        dtype=dtype,
    )
    tt_input = ttnn.from_torch(
        input_tensor.detach().to(torch.bfloat16),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_model(tt_input)
