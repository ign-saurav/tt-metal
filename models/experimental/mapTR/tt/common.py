# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common TTNN building blocks for MapTR using the tt_cnn format.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import ttnn


@dataclass(frozen=True)
class Conv2dConfig:
    """Configuration for a 2D convolution layer."""

    input_height: int
    input_width: int
    in_channels: int
    out_channels: int
    batch_size: int
    kernel_size: Tuple[int, int]
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor] = None
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    groups: int = 1
    dilation: Tuple[int, int] = (1, 1)

    # Activation function (e.g., ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))
    activation: Optional[ttnn.UnaryWithParam] = None

    # Data types
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    output_dtype: ttnn.DataType = ttnn.bfloat16
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT

    # Sharding configuration
    shard_layout: Optional[ttnn.TensorMemoryLayout] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    act_block_h_override: int = 0

    # Memory management
    deallocate_activation: bool = False
    reshard_if_not_optimal: bool = True

    # Compute configuration
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    @classmethod
    def from_torch_with_folded_bn(
        cls,
        conv: torch.nn.Conv2d,
        bn: torch.nn.BatchNorm2d,
        input_height: int,
        input_width: int,
        batch_size: int,
        **kwargs,
    ) -> "Conv2dConfig":
        """Create Conv2dConfig from PyTorch conv and batch norm layers with folded weights."""
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
        weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias_ttnn = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32)

        return cls(
            input_height=input_height,
            input_width=input_width,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            batch_size=batch_size,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            dilation=conv.dilation,
            weight=weight_ttnn,
            bias=bias_ttnn,
            **kwargs,
        )


class TtConv2d:
    """TTNN Conv2D layer using the tt_cnn format."""

    def __init__(self, config: Conv2dConfig, device: ttnn.Device):
        self.config = config
        self.device = device

        # Store weights
        self.weight = config.weight
        self.bias = config.bias

        # Build compute config
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=config.math_fidelity,
            fp32_dest_acc_en=config.fp32_dest_acc_en,
            packer_l1_acc=config.packer_l1_acc,
            math_approx_mode=True,
        )

        # Build conv config
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=config.weights_dtype,
            activation=config.activation,
            deallocate_activation=config.deallocate_activation,
            reshard_if_not_optimal=config.reshard_if_not_optimal,
            shard_layout=config.shard_layout,
            output_layout=config.output_layout,
            enable_act_double_buffer=False,
        )

        if config.act_block_h_override > 0:
            self.conv_config.act_block_h_override = config.act_block_h_override

    def __call__(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """Execute the conv2d operation.

        Returns:
            Tuple of (output_tensor, output_height, output_width)
        """
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            input_height=self.config.input_height,
            input_width=self.config.input_width,
            batch_size=self.config.batch_size,
            kernel_size=self.config.kernel_size,
            stride=self.config.stride,
            padding=self.config.padding,
            groups=self.config.groups,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.config.activation_dtype,
        )
        return x, output_height, output_width


@dataclass(frozen=True)
class MaxPool2dConfig:
    """Configuration for a 2D max pooling layer."""

    input_height: int
    input_width: int
    channels: int
    batch_size: int
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (2, 2)
    padding: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    ceil_mode: bool = False


class TtMaxPool2d:
    """TTNN MaxPool2D layer using the tt_cnn format."""

    def __init__(self, config: MaxPool2dConfig, device: ttnn.Device):
        self.config = config
        self.device = device

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Execute the maxpool2d operation."""
        return ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.config.batch_size,
            input_h=self.config.input_height,
            input_w=self.config.input_width,
            channels=self.config.channels,
            kernel_size=list(self.config.kernel_size),
            stride=list(self.config.stride),
            padding=list(self.config.padding),
            dilation=list(self.config.dilation),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=self.config.ceil_mode,
        )


def fold_batch_norm2d_into_conv2d(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold batch normalization weights into convolution weights.

    Args:
        conv: PyTorch Conv2d layer
        bn: PyTorch BatchNorm2d layer

    Returns:
        Tuple of (folded_weight, folded_bias) tensors
    """
    # Get batch norm parameters
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    # Compute scale factor
    std = torch.sqrt(var + eps)
    scale = gamma / std

    # Fold into conv weights
    weight = conv.weight * scale.view(-1, 1, 1, 1)

    # Compute folded bias
    if conv.bias is not None:
        bias = (conv.bias - mean) * scale + beta
    else:
        bias = -mean * scale + beta

    return weight, bias
