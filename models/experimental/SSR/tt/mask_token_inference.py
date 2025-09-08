# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.utils.config_helpers import matmul_config


class TTMaskTokenInference(LightweightModule):
    # def __init__(
    #     self, device, parameters, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    # ):
    #     self.device = device
    #     self.dim = dim
    #     self.num_heads = num_heads
    #     self.head_dim = dim // num_heads
    #     self.scale = qk_scale or (self.head_dim**-0.5)

    #     # Layer norm parameters (would need to be loaded from state dict)
    #     self.norm_weight = parameters["norm"]["weight"]  # ttnn tensor for layer norm weight
    #     self.norm_bias = parameters["norm"]["bias"]  # ttnn tensor for layer norm bias

    #     # Linear layer weights (would need to be preprocessed and loaded)
    #     self.q_weight = parameters["q"]["weight"]  # ttnn tensor for query projection
    #     self.k_weight = parameters["k"]["weight"]  # ttnn tensor for key projection
    #     self.v_weight = parameters["v"]["weight"]  # ttnn tensor for value projection
    #     self.proj_weight = parameters["proj"]["weight"]  # ttnn tensor for output projection

    #     self.q_bias = parameters["q"]["bias"] if qkv_bias else None
    #     self.k_bias = parameters["k"]["bias"] if qkv_bias else None
    #     self.v_bias = parameters["v"]["bias"] if qkv_bias else None
    #     self.proj_bias = parameters["proj"]["bias"]

    #     # Scale tensor
    #     scale_tensor = torch.tensor(self.scale).view(1, 1, 1, 1)
    #     self.tt_scale = ttnn.from_torch(scale_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

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
        self.proj_weight = parameters["proj"]["weight"]  # ttnn tensor for output projection

        self.proj_bias = parameters["proj"]["bias"]

        self.qkv_weight = parameters["qkv"]["weight"]  # Pre-fused QKV weight tensor
        self.qkv_bias = parameters["qkv"]["bias"] if qkv_bias else None

        # Scale tensor
        scale_tensor = torch.tensor(self.scale).view(1, 1, 1, 1)
        self.tt_scale = ttnn.from_torch(scale_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # def __call__(self, fea):
    #     B, N, C = fea.shape

    #     # Layer normalization
    #     x = ttnn.layer_norm(fea, weight=self.norm_weight, bias=self.norm_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    #     fea_skip = fea
    #     fea_skip = ttnn.reallocate(fea_skip, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    #     ttnn.deallocate(fea)

    #     # Split into classification token and feature tokens
    #     # T_s: classification token [B, 1, C]
    #     # F_s: feature tokens [B, N-1, C]
    #     T_s = ttnn.slice(x, [0, 0, 0], [B, 1, C])
    #     F_s = ttnn.slice(x, [0, 1, 0], [B, N, C])
    #     ttnn.deallocate(x)

    #     # Query from feature tokens
    #     q = ttnn.linear(F_s, self.q_weight, bias=self.q_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    #     # q = ttnn.reshape(q, (B, N - 1, self.num_heads, self.head_dim))
    #     # q = ttnn.permute(q, (0, 2, 1, 3))

    #     # Key from classification token
    #     k = ttnn.linear(T_s, self.k_weight, bias=self.k_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    #     # k = ttnn.reshape(k, (B, 1, self.num_heads, self.head_dim))
    #     # k = ttnn.permute(k, (0, 2, 1, 3))

    #     # Value from classification token
    #     v = ttnn.linear(T_s, self.v_weight, bias=self.v_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    #     # v = ttnn.reshape(v, (B, 1, self.num_heads, self.head_dim))
    #     # v = ttnn.permute(v, (0, 2, 1, 3))

    #     # Attention computation: q @ k.T
    #     k_transposed = ttnn.transpose(k, -2, -1)
    #     attn = ttnn.matmul(q, k_transposed)

    #     # Scale attention scores
    #     attn = ttnn.multiply(attn, self.tt_scale)

    #     # Apply sigmoid instead of softmax
    #     attn = ttnn.sigmoid(attn)

    #     # Apply attention dropout (if needed, would require custom implementation)
    #     # attn = apply_dropout(attn, attn_drop)

    #     # Compute attention output
    #     infer_fea = ttnn.matmul(attn, v)

    #     # Reshape back to [B, N-1, C]
    #     infer_fea = ttnn.permute(infer_fea, (0, 2, 1, 3))
    #     infer_fea = ttnn.to_layout(infer_fea, layout=ttnn.ROW_MAJOR_LAYOUT)
    #     infer_fea = ttnn.reshape(infer_fea, (B, N - 1, C))
    #     infer_fea = ttnn.to_layout(infer_fea, layout=ttnn.TILE_LAYOUT)

    #     # Output projection
    #     infer_fea = ttnn.linear(infer_fea, self.proj_weight, bias=self.proj_bias)

    #     # Apply projection dropout (if needed)
    #     # infer_fea = apply_dropout(infer_fea, proj_drop)

    #     # Residual connection with original feature tokens
    #     original_features = ttnn.slice(fea_skip, [0, 1, 0], [B, N, C])
    #     infer_fea = ttnn.add(infer_fea, original_features)

    #     return infer_fea

    def __call__(self, fea):
        B, N, C = fea.shape

        # Layer normalization
        x = ttnn.layer_norm(fea, weight=self.norm_weight, bias=self.norm_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        fea_skip = fea
        fea_skip = ttnn.reallocate(fea_skip, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(fea)

        # Split into classification token and feature tokens
        # T_s: classification token [B, 1, C]
        # F_s: feature tokens [B, N-1, C]
        T_s = ttnn.slice(x, [0, 0, 0], [B, 1, C])
        F_s = ttnn.slice(x, [0, 1, 0], [B, N, C])
        ttnn.deallocate(x)

        # Query from feature tokens
        F_s_prg_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=36,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        F_s_qkv = ttnn.linear(
            F_s, self.qkv_weight, bias=self.qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=F_s_prg_config
        )
        ttnn.deallocate(F_s)

        # Key from classification token
        # For classification token (keys and values from T_s)
        T_S_PRG_CONFIG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=36,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        T_s_qkv = ttnn.linear(
            T_s, self.qkv_weight, bias=self.qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=T_S_PRG_CONFIG
        )
        ttnn.deallocate(T_s)

        # Split F_s QKV (for queries)
        (q_from_F_s, k_from_FS, v_from_FS) = ttnn.transformer.split_query_key_value_and_split_heads(
            F_s_qkv,
            num_heads=self.num_heads,
            transpose_key=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(F_s_qkv)
        ttnn.deallocate(k_from_FS)
        ttnn.deallocate(v_from_FS)

        (q_from_T_s, k_from_T_s, v_from_T_s) = ttnn.transformer.split_query_key_value_and_split_heads(
            T_s_qkv,
            num_heads=self.num_heads,
            transpose_key=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(T_s_qkv)
        ttnn.deallocate(q_from_T_s)

        # Attention computation: q @ k.T
        prg_config = matmul_config(q_from_F_s.shape[-2], q_from_F_s.shape[-1], k_from_T_s.shape[-1], (8, 8))
        attn = ttnn.matmul(q_from_F_s, k_from_T_s, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=prg_config)
        ttnn.deallocate(q_from_F_s)
        ttnn.deallocate(k_from_T_s)

        # Scale attention scores
        attn = ttnn.multiply(attn, self.tt_scale, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Apply sigmoid instead of softmax
        attn = ttnn.sigmoid(attn, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Compute attention output
        infer_fea = ttnn.matmul(attn, v_from_T_s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(v_from_T_s)
        ttnn.reallocate(infer_fea)

        # Reshape back to [B, N-1, C]
        infer_fea = ttnn.permute(infer_fea, (0, 2, 1, 3))
        infer_fea = ttnn.reshape(infer_fea, (B, N - 1, C))

        # Output projection
        infer_fea = ttnn.linear(infer_fea, self.proj_weight, bias=self.proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Apply projection dropout (if needed)

        # Residual connection with original feature tokens
        original_features = ttnn.slice(fea_skip, [0, 1, 0], [B, N, C])
        infer_fea = ttnn.add(infer_fea, original_features, memory_config=ttnn.L1_MEMORY_CONFIG)

        return infer_fea
