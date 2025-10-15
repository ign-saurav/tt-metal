# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTSelfAttention(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        n_embed,
        n_head,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None,
    ):
        self.parameters = parameters
        self.device = device
        self.n_embed = n_embed
        self.n_head = n_head
        self.dtype = dtype
        self.memory_config = memory_config
        self.compute_kernel_config = compute_kernel_config
        # Add program configs
        self.qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 6),
            in0_block_w=2,  # K=96/32=3 tiles, use 2
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,  # 192/32/6 = 1
            per_core_N=2,  # 384/32/6 = 2
            transpose_mcast=False,
            fused_activation=None,
        )

        self.proj_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(3, 3),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,  # 192/32/3 = 2
            per_core_N=1,  # 96/32/3 = 1
            transpose_mcast=False,
            fused_activation=None,
        )

    def forward(self, x):
        B, T, C = x.shape

        # Ensure input is in tile layout
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Fused QKV projection
        # query_key_value = ttnn.linear(
        #    x,
        #    self.parameters["query_key_value"]["weight"],
        #    bias=self.parameters["query_key_value"]["bias"],
        #    memory_config=self.memory_config,
        #    dtype=self.dtype,
        #    core_grid=ttnn.CoreGrid(y=B, x=8),
        # )

        # Fused QKV projection
        query_key_value = ttnn.linear(
            x,
            self.parameters["query_key_value"]["weight"],
            bias=self.parameters["query_key_value"]["bias"],
            program_config=self.qkv_program_config,  # ADD THIS
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,  # CHANGE THIS
            dtype=self.dtype,
        )
        # Split QKV and split heads
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_heads=self.n_head, transpose_key=False
        )
        ttnn.deallocate(query_key_value)

        # Scaled dot product attention - single fused operation
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            scale=None,  # Will use default 1/sqrt(head_dim)
            program_config=None,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
        )

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        # Concatenate heads
        y = ttnn.transformer.concatenate_heads(
            attn_output,
            memory_config=self.memory_config,
        )
        ttnn.deallocate(attn_output)

        if y.shape[-1] != self.n_embed:
            # slice and remove padding
            y = ttnn.to_torch(y)[..., : self.n_embed]
            y = ttnn.from_torch(
                y,
                device=self.device,
                dtype=self.dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )

        # Output projection
        y = ttnn.linear(
            y,
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=B, x=8),
            dtype=self.dtype,
        )

        return y
