# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTOCAB(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        input_resolution,
        window_size,
        overlap_ratio,
        num_heads,
        parameters,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        self.overlap_ratio = overlap_ratio
        self.dtype = dtype

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        # Extract preprocessed parameters
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]

        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"]

        self.relative_position_bias_table = parameters["relative_position_bias_table"]

        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"]

        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]

        self.mlp_fc1_weight = parameters["mlp"]["fc1"]["weight"]
        self.mlp_fc1_bias = parameters["mlp"]["fc1"]["bias"]
        self.mlp_fc2_weight = parameters["mlp"]["fc2"]["weight"]
        self.mlp_fc2_bias = parameters["mlp"]["fc2"]["bias"]

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x

        # Layer normalization
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Fused QKV projection - use single linear operation
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        # Use transformer function for QKV splitting
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_heads=self.num_heads, transpose_key=False
        )
        ttnn.deallocate(qkv)

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 7],
            q_chunk_size=512,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Use optimized scaled dot product attention
        attention_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Deallocate intermediate tensors
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        # Use transformer function for head concatenation
        context_layer = ttnn.transformer.concatenate_heads(
            attention_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention_output)

        if context_layer.shape[-1] != self.dim:
            # remove padding
            context_layer = ttnn.to_torch(context_layer)[..., : self.dim]  # slice to 180 and remove padding
            context_layer = ttnn.from_torch(
                context_layer,
                device=self.device,
                dtype=self.dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
        x = ttnn.reshape(context_layer, (b, h * w, self.dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Output projection and residual
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )
        x = ttnn.add(x, shortcut, dtype=self.dtype)

        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        mlp_out = ttnn.linear(
            x,
            self.mlp_fc1_weight,
            bias=self.mlp_fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="gelu",
        )
        mlp_out = ttnn.linear(
            mlp_out,
            self.mlp_fc2_weight,
            bias=self.mlp_fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        x = ttnn.add(x, mlp_out, dtype=self.dtype)
        return x
