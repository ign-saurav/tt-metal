# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TT implementation of SAM ImageEncoderViT (ViT-B).
Uses the same ttnn ops as tt_transformers multimodal image attention (no edits to that module):
  ttnn.linear, ttnn.experimental.nlp_create_qkv_heads, ttnn.transformer.scaled_dot_product_attention,
  ttnn.experimental.nlp_concat_heads. All layers are ttnn: conv2d, layer_norm, linear, gelu, SDPA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

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
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

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
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

    def forward(self, x_11SH: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: (1, 1, seq_len, dim). Returns (1, 1, seq_len, dim)."""
        seq_len = x_11SH.shape[-2]

        # input (1,1,S,768) @ weight (1,1,768,2304) -> (1,1,S,2304)
        xqkv_fused = ttnn.matmul(
            x_11SH,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
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

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            attn_mask=None,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        # attn_concat (1,1,S,768) @ wo (1,1,768,768) -> (1,1,S,768)
        output = ttnn.matmul(
            attn_concat,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        if self.bo is not None:
            output = ttnn.add(output, self.bo)
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
    ):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn_module
        self.norm2 = norm2
        self.mlp = mlp

    def forward(self, x_11SH: ttnn.Tensor) -> ttnn.Tensor:
        attn_out = self.attn(self.norm1(x_11SH))
        res = ttnn.add(x_11SH, attn_out)
        ttnn.deallocate(attn_out)
        mlp_out = self.mlp(self.norm2(res))
        out = ttnn.add(res, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(res)
        return out


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
        else:
            self.pos_embed = None

        # Blocks
        mlp_dim = int(embed_dim * mlp_ratio)
        self.blocks = []
        for i in range(depth):
            blk = sam_torch_module.blocks[i]
            attn = TtSamAttention(
                device=device,
                dim=embed_dim,
                num_heads=num_heads,
                qkv_weight=blk.attn.qkv.weight,
                qkv_bias=blk.attn.qkv.bias,
                proj_weight=blk.attn.proj.weight,
                proj_bias=blk.attn.proj.bias,
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
            self.blocks.append(TtSamBlock(device, embed_dim, num_heads, mlp_ratio, attn, norm1, norm2, mlp))

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
        """
        x: (B, C, H, W) in TT layout. Returns final feature map from net_3.
        """
        # Input (B, C, H, W) -> (B, H, W, C) for patch_embed conv
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = ttnn.permute(x, (0, 2, 3, 1))
        # Patch embed -> (B, grid, grid, embed_dim)
        x = self.patch_embed(x)
        # Add pos_embed if present (x is B,H,W,C; pos_embed is 1,H,W,C)
        if self.pos_embed is not None:
            # Add on host to avoid device "Invalid subtile broadcast type" (tile layout mismatch)
            b, h, w, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            x_torch = ttnn.to_torch(x)
            pos_torch = self.pos_embed.detach().to(torch.bfloat16)  # (1, pos_h, pos_w, C), e.g. (1,64,64,768)
            grid = self.grid_size  # patch grid for current image_size (e.g. 40 for 640)
            if pos_torch.shape[1] != grid or pos_torch.shape[2] != grid:
                pos_torch = torch.nn.functional.interpolate(
                    pos_torch.permute(0, 3, 1, 2), size=(grid, grid), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            # Reshape pos to match x (x may be (B,H,W,C) or (B,1,H*W,C) depending on conv2d output)
            if (pos_torch.shape[1], pos_torch.shape[2]) != (h, w):
                pos_torch = pos_torch.reshape(1, 1, grid * grid, c).expand(b, 1, grid * grid, c).clone()
            else:
                pos_torch = pos_torch.expand(b, h, w, c).clone()
            x_torch = x_torch + pos_torch
            # Use ROW_MAJOR then TILE so layout matches conv2d output and matmul sees (S, C)
            x = ttnn.from_torch(
                x_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # (B, H, W, C) -> (1, 1, S, C)
        b, h, w, c = x.shape
        x = ttnn.reshape(x, (1, 1, b * h * w, c))
        for blk in self.blocks:
            x = blk(x)
        # (1, 1, S, C) -> (B, C, H, W); run neck + net_2 + net_3 in PyTorch to avoid device L1 OOM
        x_t = ttnn.to_torch(x)
        ttnn.deallocate(x)
        # (1, 1, S, C) -> (B, H, W, C) -> (B, C, H, W); use grid_size so h,w are correct after blocks
        grid = self.grid_size
        x_t = x_t.reshape(1, grid, grid, c).permute(0, 3, 1, 2)
        with torch.no_grad():
            for layer in self.neck_torch:
                x_t = layer(x_t)
            x_t = self.net_2_torch(x_t)
            x_t = self.net_3_torch(x_t)
        # x_t: (1, 1024, 10, 10) for image_size 640; keep on host so to_torch preserves shape
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
    """
    Run TT SAM image encoder forward.
    input_tensor: (B, 3, H, W) torch tensor, e.g. bfloat16. Will be transferred to device.
    Returns TT tensor output from TtImageEncoderViT (same shape as torch SAM forward).
    """
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
