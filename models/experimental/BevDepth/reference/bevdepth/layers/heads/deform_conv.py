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
    """Use torchvision's deform_conv2d when MMCV extensions aren't available.

    BEVDepth DepthNet uses DCN with:
        - kernel_size=3
        - groups=4
        - im2col_step=128 (MMCV-specific, not used in torchvision)
        - deform_groups=1 (default)

    Note on torchvision.ops.deform_conv2d (torchvision >= 0.9.0):
        Signature: deform_conv2d(input, offset, weight, bias=None, stride=(1, 1),
                                 padding=(0, 0), dilation=(1, 1), mask=None)

        - Does NOT have explicit 'groups' parameter
        - Groups are inferred from weight shape: (out_channels, in_channels // groups, kH, kW)
        - 'mask' parameter is for DCNv2 (modulated), not used in DCNv1 (BEVDepth uses DCNv1)
        - 'im2col_step' is MMCV-specific and not applicable to torchvision

    Note on MMCV vs torchvision:
        - MMCV's deform_conv2d: supports explicit 'groups' and 'deform_groups' parameters
        - torchvision's deform_conv2d: groups inferred from weight shape
        - Both should produce equivalent results when weight shapes match
    """
    try:
        from torchvision.ops import deform_conv2d as tv_deform_conv2d

        # torchvision's deform_conv2d infers groups from weight shape
        # Weight shape should be: (out_channels, in_channels // groups, kH, kW)
        # When groups > 1, the weight's second dimension reflects this division
        return tv_deform_conv2d(input, offset, weight, bias, stride, padding, dilation)
    except ImportError:
        raise ImportError("torchvision.ops.deform_conv2d is required for DCN fallback")


class DeformConv2d(nn.Module):
    """Deformable Convolution 2D layer (DCNv1).

    This is a standalone implementation that uses torchvision's deform_conv2d
    when MMCV's compiled extensions are not available.

    BEVDepth DepthNet configuration:
        - in_channels=512 (mid_channels)
        - out_channels=512 (mid_channels)
        - kernel_size=3
        - padding=1
        - groups=4
        - im2col_step=128 (MMCV-specific, stored but not used with torchvision)
        - deform_groups=1 (default)

    Args:
        in_channels (int): Number of input channels. BEVDepth uses 512.
        out_channels (int): Number of output channels. BEVDepth uses 512.
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3.
        stride (int or tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides. Default: 0 (set to 1 for BEVDepth's 3x3 kernel).
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input to output. Default: 1 (BEVDepth uses 4).
        deform_groups (int): Number of deformable group partitions. Default: 1.
        bias (bool): If True, adds a learnable bias. Default: False (DCN doesn't support bias).
        im2col_step (int): MMCV-specific batch size for im2col. Default: 128 (BEVDepth uses 128).
            Note: This parameter is only used by MMCV's implementation, not torchvision.
    """

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
