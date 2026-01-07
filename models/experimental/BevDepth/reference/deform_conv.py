# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

########################################################
# Adapted from: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/deform_conv.py
# Copyright (c) OpenMMLab. All rights reserved.
########################################################

from typing import Tuple, Union
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


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
    """Torchvision's deform_conv2d, as mmcv's deform_conv2d needs gpu support."""

    return deform_conv2d(input, offset, weight, bias, stride, padding, dilation)


class DeformConv2d(nn.Module):
    """Deformable Convolution 2D layer (DCNv1) with torchvision's deform_conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
        im2col_step: int = 128,
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
        # Using torchvision's deform_conv2d
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
