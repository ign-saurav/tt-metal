# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class TTWindowAttentionTR(LightweightModule):
    def __init__(self, device, parameters, dim, window_size, num_heads, memory_config=None, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dtype = dtype

        # Extract preprocessed parameters
        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"] if "bias" in parameters["qkv"] else None
        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"] if "bias" in parameters["proj"] else None
        self.relative_position_bias = parameters["relative_position_bias"]

        # Scale factor
        self.scale = self.head_dim**-0.5

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        self.memory_config = ttnn.L1_MEMORY_CONFIG if b_ * n * c < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=self.memory_config,
            dtype=self.dtype,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(x)
        (
            q,
            k,
            v,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads
        )

        ttnn.deallocate(qkv)

        # Remove the first dimension
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # Scale Q
        q = ttnn.multiply(q, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.dtype)

        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.add(attn, self.relative_position_bias, memory_config=self.memory_config, dtype=self.dtype)

        # Apply mask if provided
        if mask is not None:
            nw = mask.shape[0]
            attn = ttnn.reshape(attn, [b_ // nw, nw, self.num_heads, n, n])
            mask_expanded = ttnn.unsqueeze(ttnn.unsqueeze(mask, 1), 0)
            attn = ttnn.add(attn, mask_expanded, dtype=self.dtype)
            attn = ttnn.reshape(attn, [-1, self.num_heads, n, n])

        # Softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=self.memory_config)

        # Apply attention to values
        x = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )  # [b_, num_heads, n, head_dim]

        output_tensor = ttnn.transformer.concatenate_heads(x)

        if self.proj_weight.shape[-1] != output_tensor.shape[-1]:
            head_size = self.proj_weight.shape[-1] // self.num_heads
            padded_head_size = output_tensor.shape[-1] // self.num_heads
            output_tensor = ttnn.to_torch(output_tensor)

            # Remove the padding
            output_tensor = torch.cat(
                [chunk[..., :head_size] for chunk in torch.split(output_tensor, padded_head_size, dim=-1)], dim=-1
            )
            x = ttnn.from_torch(
                output_tensor,
                device=self.device,
                dtype=self.dtype,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )

        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        return x
