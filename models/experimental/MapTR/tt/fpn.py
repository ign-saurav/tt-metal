# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import ttnn

from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration


class TtConvModule:
    """TTNN ConvModule wrapper for FPN."""

    def __init__(self, config: Conv2dConfiguration, device: ttnn.Device):
        """Initialize the ConvModule.

        Args:
            config: Conv2dConfiguration for the convolution layer
            device: TTNN device
        """
        self.device = device
        self.conv = TtConv2d(config, device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Execute the convolution.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.conv(x)


class TtFPN:
    """TTNN FPN (Feature Pyramid Network)."""

    def __init__(
        self,
        lateral_conv_config: Conv2dConfiguration,
        fpn_conv_config: Conv2dConfiguration,
        device: ttnn.Device,
    ):
        """Initialize the FPN.

        Args:
            lateral_conv_config: Configuration for lateral convolution (1x1)
            fpn_conv_config: Configuration for FPN convolution (3x3)
            device: TTNN device
        """
        self.device = device
        self.lateral_conv_config = lateral_conv_config
        self.fpn_conv_config = fpn_conv_config
        self.lateral_convs = TtConvModule(lateral_conv_config, device=device)
        self.fpn_convs = TtConvModule(fpn_conv_config, device=device)

    def __call__(self, inputs: list) -> Tuple[ttnn.Tensor, ...]:
        """Execute the FPN forward pass.

        Args:
            inputs: List of input tensors (typically one tensor from backbone)

        Returns:
            Tuple of output tensors
        """
        # Build laterals
        laterals = self.lateral_convs(inputs[0])
        # Move to DRAM between convs
        laterals = ttnn.to_memory_config(laterals, ttnn.DRAM_MEMORY_CONFIG)

        # Apply FPN convs
        outs = self.fpn_convs(laterals)
        ttnn.deallocate(laterals)

        # Reshape output to ensure correct spatial dimensions
        # Output from conv2d is in NHWC format: [batch, height, width, channels]
        # Get output dimensions from config
        batch_size = self.fpn_conv_config.batch_size
        out_channels = self.fpn_conv_config.out_channels

        # Compute output dimensions: 3x3 conv with padding=1 preserves spatial size
        # Input to FPN conv is the output of lateral conv
        fpn_input_h = self.fpn_conv_config.input_height
        fpn_input_w = self.fpn_conv_config.input_width
        fpn_out_h = fpn_input_h  # 3x3 with padding=1 preserves size
        fpn_out_w = fpn_input_w

        # Reshape to ensure dimensions match expected output
        outs = ttnn.reshape(outs, (batch_size, fpn_out_h, fpn_out_w, out_channels))

        return (outs,)
