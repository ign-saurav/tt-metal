# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Bottleneck block for ResNet using the tt_cnn format.
"""

from typing import Optional

import ttnn

from models.experimental.mapTR.tt.common import Conv2dConfig, TtConv2d


class TtBottleneck:
    """ResNet Bottleneck block using tt_cnn format Conv2D layers."""

    def __init__(
        self,
        conv1_config: Conv2dConfig,
        conv2_config: Conv2dConfig,
        conv3_config: Conv2dConfig,
        device: ttnn.Device,
        downsample_config: Optional[Conv2dConfig] = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        """Initialize the Bottleneck block.

        Args:
            conv1_config: Configuration for the first 1x1 conv
            conv2_config: Configuration for the 3x3 conv
            conv3_config: Configuration for the final 1x1 conv
            device: TTNN device
            downsample_config: Configuration for downsample conv (optional)
            activation_dtype: Data type for activations (bfloat16 or bfloat8_b)
        """
        self.device = device
        self.activation_dtype = activation_dtype
        self.has_downsample = downsample_config is not None

        # Create conv layers
        self.conv1 = TtConv2d(conv1_config, device)
        self.conv2 = TtConv2d(conv2_config, device)
        self.conv3 = TtConv2d(conv3_config, device)

        if self.has_downsample:
            self.downsample = TtConv2d(downsample_config, device)

    def __call__(self, x_identity: ttnn.Tensor) -> ttnn.Tensor:
        """Execute the bottleneck block.

        Args:
            x_identity: Input tensor

        Returns:
            Output tensor after bottleneck processing
        """
        # First conv (1x1) with ReLU
        x, _, _ = self.conv1(x_identity)

        # Convert identity to bfloat8_b if needed
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        # Move to DRAM between convs
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Second conv (3x3) with ReLU
        x, _, _ = self.conv2(x)

        # Third conv (1x1) without ReLU
        x, _, _ = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Downsample identity if needed
        if self.has_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        # Residual connection + ReLU
        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
