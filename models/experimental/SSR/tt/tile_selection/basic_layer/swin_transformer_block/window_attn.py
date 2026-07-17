# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


class TTWindowAttention(nn.Module):
    def __init__(
        self,
        parameters,
        device,
        dim,
        window_size,
        num_heads,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dtype = dtype

    def forward(self, input_tensor, mask=None):
        """
        Args:
            input_tensor: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        relative_position_bias = self.parameters["relative_position_bias"]

        B_, N, C = input_tensor.shape

        # QKV projection
        qkv_weight = self.parameters["qkv"]["weight"]
        qkv_bias = self.parameters["qkv"]["bias"]

        qkv = ttnn.linear(
            input_tensor,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG if B_ * N * C < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.dtype,
        )
        ttnn.deallocate(input_tensor)

        # Split QKV using built-in function
        (
            q,
            k,
            v,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_heads=self.num_heads, transpose_key=True
        )
        ttnn.deallocate(qkv)

        # Apply scaling
        q = q * self.scale

        # Compute attention scores
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )

        # Clean up intermediate tensors
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # Add relative position bias
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.dtype)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = ttnn.reshape(attn, (B_ // nW, nW, self.num_heads, N, N), memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = attn + ttnn.unsqueeze(ttnn.unsqueeze(mask, 1), 0)
            attn = ttnn.reshape(attn, (-1, self.num_heads, N, N), memory_config=ttnn.L1_MEMORY_CONFIG)

            ttnn.deallocate(mask)

        # Apply softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Compute final output
        output_tensor = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
        )

        # Clean up attention and value tensors
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        output_tensor = ttnn.transformer.concatenate_heads(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Apply projection
        proj_weight = self.parameters["proj"]["weight"]
        proj_bias = self.parameters["proj"]["bias"]

        output_tensor = ttnn.linear(
            output_tensor,
            proj_weight,
            bias=proj_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.dtype,
        )

        return output_tensor
