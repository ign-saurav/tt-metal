# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from models.experimental.tt_symbiote.core.module import TTNNModule


class QuickGELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TTNNQuickGelu(TTNNModule):
    """TTNN-accelerated Quick Gelu activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = QuickGELU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Quick Gelu activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scaled = ttnn.multiply(input_tensor, 1.702)
        sigmoid_output = ttnn.sigmoid(scaled)
        tt_output = ttnn.multiply(input_tensor, sigmoid_output)
        ttnn.deallocate(scaled)
        ttnn.deallocate(sigmoid_output)
        return tt_output
