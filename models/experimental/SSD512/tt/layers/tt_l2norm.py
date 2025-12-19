# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.utility_functions import torch_to_tt_tensor_rm


class TtL2Norm:
    def __init__(self, n_channels, scale=20, eps=1e-10, device=None):
        self.n_channels = n_channels
        self.eps = eps
        self.device = device
        self.dtype = ttnn.bfloat16

        # Create weight tensor in DRAM to avoid L1 overflow
        weight_torch = torch.full([1, 1, 1, n_channels], scale, dtype=torch.float32)
        self.weight = ttnn.from_torch(
            weight_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x, memory_config=None):
        x_shape = x.shape
        batch_size = x_shape[0]

        # Determine input format and convert to NHWC if needed
        if len(x_shape) == 4:
            dim1_val = x_shape[1]
            dim3_val = x_shape[3]

            # Check if input is in NCHW format (channels at index 1)
            if dim1_val == self.n_channels and dim3_val != self.n_channels:
                # NCHW format - convert to NHWC
                if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = ttnn.permute(x, (0, 2, 3, 1))
                if x.layout != ttnn.TILE_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Now x is in NHWC format [batch, height, width, channels]
        x_shape = x.shape
        height, width, channels = x_shape[1], x_shape[2], x_shape[3]

        # Use DRAM for large tensors to avoid L1 overflow
        tensor_size_estimate = batch_size * height * width * channels * 2  # bfloat16 = 2 bytes
        use_l1 = height <= 32 and width <= 32 and tensor_size_estimate <= 1 * 1024 * 1024
        layer_memory_config = ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG

        # Ensure input is in correct memory location
        if x.memory_config().buffer_type != layer_memory_config.buffer_type:
            x = ttnn.to_memory_config(x, layer_memory_config)

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Compute L2 norm: sqrt(sum(x^2, dim=channels) + eps)
        squared = ttnn.mul(x, x, memory_config=layer_memory_config)

        # Sum over channel dimension (dim=3 in NHWC)
        sum_result = ttnn.sum(squared, dim=3, keepdim=True, memory_config=layer_memory_config)

        # Add epsilon
        eps_tensor = ttnn.full_like(sum_result, self.eps, memory_config=layer_memory_config)
        sum_with_eps = ttnn.add(sum_result, eps_tensor, memory_config=layer_memory_config)

        # Take sqrt
        norm = ttnn.sqrt(sum_with_eps, memory_config=layer_memory_config)

        # Normalize: x / norm
        x_norm = ttnn.div(x, norm, memory_config=layer_memory_config)

        # Ensure weight is in same memory as x_norm
        weight = self.weight
        if weight.memory_config().buffer_type != layer_memory_config.buffer_type:
            weight = ttnn.to_memory_config(weight, layer_memory_config)

        # Scale by learned weight
        out = ttnn.mul(x_norm, weight, memory_config=layer_memory_config)

        return out


def l2norm(input_tensor, num_channels=512, scale=20.0, device=None):
    l2norm_module = TtL2Norm(n_channels=num_channels, scale=scale, device=device)

    if isinstance(input_tensor, torch.Tensor):
        input_ttnn = torch_to_tt_tensor_rm(input_tensor, device=device)
        output = l2norm_module(input_ttnn)
        return output
    else:
        output = l2norm_module(input_tensor)
        return output
