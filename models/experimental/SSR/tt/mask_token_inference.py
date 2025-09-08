# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class TTMaskTokenInference(LightweightModule):
    def __init__(
        self, device, parameters, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        self.device = device
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim**-0.5)

        # Layer norm parameters (would need to be loaded from state dict)
        self.norm_weight = parameters["norm"]["weight"]  # ttnn tensor for layer norm weight
        self.norm_bias = parameters["norm"]["bias"]  # ttnn tensor for layer norm bias

        # Linear layer weights (would need to be preprocessed and loaded)
        self.q_weight = parameters["q"]["weight"]  # ttnn tensor for query projection
        self.k_weight = parameters["k"]["weight"]  # ttnn tensor for key projection
        self.v_weight = parameters["v"]["weight"]  # ttnn tensor for value projection
        self.proj_weight = parameters["proj"]["weight"]  # ttnn tensor for output projection

        self.q_bias = parameters["q"]["bias"] if qkv_bias else None
        self.k_bias = parameters["k"]["bias"] if qkv_bias else None
        self.v_bias = parameters["v"]["bias"] if qkv_bias else None
        self.proj_bias = parameters["proj"]["bias"]

        # Scale tensor
        scale_tensor = torch.tensor(self.scale).view(1, 1, 1, 1)
        self.tt_scale = ttnn.from_torch(scale_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    def __call__(self, fea):
        B, N, C = fea.shape

        # Layer normalization
        x = ttnn.layer_norm(fea, weight=self.norm_weight, bias=self.norm_bias)

        # Split into classification token and feature tokens
        # T_s: classification token [B, 1, C]
        # F_s: feature tokens [B, N-1, C]
        T_s = ttnn.slice(x, [0, 0, 0], [B, 1, C])
        F_s = ttnn.slice(x, [0, 1, 0], [B, N, C])

        # Query from feature tokens
        q = ttnn.linear(F_s, self.q_weight, bias=self.q_bias)
        q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.reshape(q, (B, N - 1, self.num_heads, self.head_dim))
        q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
        q = ttnn.permute(q, (0, 2, 1, 3))

        # Key from classification token
        k = ttnn.linear(T_s, self.k_weight, bias=self.k_bias)
        k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.reshape(k, (B, 1, self.num_heads, self.head_dim))
        k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
        k = ttnn.permute(k, (0, 2, 1, 3))

        # Value from classification token
        v = ttnn.linear(T_s, self.v_weight, bias=self.v_bias)
        v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
        v = ttnn.reshape(v, (B, 1, self.num_heads, self.head_dim))
        v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Attention computation: q @ k.T
        k_transposed = ttnn.transpose(k, -2, -1)
        attn = ttnn.matmul(q, k_transposed)

        # Scale attention scores
        attn = ttnn.multiply(attn, self.tt_scale)

        # Apply sigmoid instead of softmax
        attn = ttnn.sigmoid(attn)

        # Apply attention dropout (if needed, would require custom implementation)
        # attn = apply_dropout(attn, attn_drop)

        # Compute attention output
        infer_fea = ttnn.matmul(attn, v)

        # Reshape back to [B, N-1, C]
        infer_fea = ttnn.permute(infer_fea, (0, 2, 1, 3))
        infer_fea = ttnn.to_layout(infer_fea, layout=ttnn.ROW_MAJOR_LAYOUT)
        infer_fea = ttnn.reshape(infer_fea, (B, N - 1, C))
        infer_fea = ttnn.to_layout(infer_fea, layout=ttnn.TILE_LAYOUT)

        # Output projection
        infer_fea = ttnn.linear(infer_fea, self.proj_weight, bias=self.proj_bias)

        # Apply projection dropout (if needed)
        # infer_fea = apply_dropout(infer_fea, proj_drop)

        # Residual connection with original feature tokens
        original_features = ttnn.slice(fea, [0, 1, 0], [B, N, C])
        infer_fea = ttnn.add(infer_fea, original_features)

        return infer_fea
