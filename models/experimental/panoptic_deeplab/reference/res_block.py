# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from torch import Tensor
from models.experimental.panoptic_deeplab.reference.utils import Conv2d, DepthwiseSeparableConv2d


class ResModel(torch.nn.Module):
    """
    Decoder Res Module.
    The input and output of `forward()` method must be NCHW tensors.

    Args:
        in_channels (int):              input channel length
        intermediate_channels (int):    intermediate channel length
        out_channels (int):             output channel length
    """

    def __init__(self, in_channels, intermediate_channels, out_channels) -> None:
        super().__init__()
        self.project_conv = Conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=nn.BatchNorm2d(in_channels // 8),
            activation=nn.ReLU(),
        )
        self.fuse_conv = DepthwiseSeparableConv2d(
            intermediate_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            dilation=1,
            norm1=nn.BatchNorm2d(intermediate_channels),
            activation1=nn.ReLU(),
            norm2=nn.BatchNorm2d(out_channels),
            activation2=nn.ReLU(),
        )

    def forward(self, x: Tensor, res2: Tensor) -> Tensor:
        """
        Forward pass of Res Module.

        Args:
            x:      Input tensor of shape [N, C, H, W]
            res2:   Residual Input tensor of shape [N, C, H, W]

        Returns:
            out: res output
        """
        out = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        out_ = self.project_conv(res2)
        out = torch.cat((out_, out), dim=1)
        out = self.fuse_conv(out)
        return out
