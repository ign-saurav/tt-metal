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
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

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

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.linear(
            input_tensor,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(input_tensor)

        # Reshape and permute QKV
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Extract Q, K, V
        q = qkv[0:1, :, :, :, :]
        k = qkv[1:2, :, :, :, :]
        v = qkv[2:3, :, :, :, :]
        ttnn.deallocate(qkv)

        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # Convert to tile layout for computation
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply scaling
        q = q * self.scale

        # Compute attention scores
        k = ttnn.permute(k, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Clean up intermediate tensors
        # ttnn.deallocate(qkv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # Add relative position bias
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (B_ // nW, nW, self.num_heads, N, N))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = attn + ttnn.unsqueeze(ttnn.unsqueeze(mask, 1), 0)
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (-1, self.num_heads, N, N))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Compute final output
        output_tensor = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Clean up attention and value tensors
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        # Reshape output
        output_tensor = ttnn.permute(output_tensor, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(
            output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        output_tensor = ttnn.reshape(output_tensor, (B_, N, C))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return output_tensor
