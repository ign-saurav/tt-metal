# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# This is a standalone implementation that works without MMCV's compiled extensions

from typing import Tuple, Union
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair


def _deform_conv2d_torchvision(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    bias: Tensor = None,
) -> Tensor:
    """Use torchvision's deform_conv2d as a fallback when MMCV extensions aren't available."""
    try:
        from torchvision.ops import deform_conv2d as tv_deform_conv2d

        return tv_deform_conv2d(input, offset, weight, bias, stride, padding, dilation)
    except ImportError:
        raise ImportError("torchvision.ops.deform_conv2d is required for DCN fallback")


class DeformConv2d(nn.Module):
    """Deformable Convolution 2D layer.

    This is a standalone implementation that uses torchvision's deform_conv2d
    when MMCV's compiled extensions are not available.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dilation (int or tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input to output channels.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output. Default: False.
        im2col_step (int): Number of samples processed by im2col_cuda_kernel per call.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
        im2col_step: int = 32,
    ):
        super(DeformConv2d, self).__init__()

        assert not bias, "DeformConv2d does not support bias"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.im2col_step = im2col_step

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, offset: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map, shape (B, C_in, H_in, W_in)
            offset (Tensor): Offset for deformable convolution, shape
                (B, deform_groups*kernel_size[0]*kernel_size[1]*2, H_out, W_out)

        Returns:
            Tensor: Output feature map.
        """
        # Use torchvision's implementation
        return _deform_conv2d_torchvision(
            x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, None
        )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"dilation={self.dilation}, "
        s += f"groups={self.groups}, "
        s += f"deform_groups={self.deform_groups}, "
        s += "bias=False)"
        return s


class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    This automatically generates the offset from the input, so it can be used
    as a drop-in replacement for regular Conv2d.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    """

    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)

        # Create offset generation layer
        offset_channels = self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        """Initialize offset to zero."""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map, shape (B, C_in, H_in, W_in)

        Returns:
            Tensor: Output feature map.
        """
        offset = self.conv_offset(x)
        return super().forward(x, offset)


# Alias for compatibility
DCN = DeformConv2dPack
