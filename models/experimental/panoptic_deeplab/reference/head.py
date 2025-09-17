# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from torch import Tensor
from models.experimental.panoptic_deeplab.reference.utils import Conv2d


class HeadModel(torch.nn.Module):
    """
    Decoder Head Module.
    The input and output of `forward()` method must be NCHW tensors.

    Args:
        in_channels (int): input channel length
        intermediate_channels (int): intermediate channel length
        out_channels (int): output channel length
    """

    def __init__(self, in_channels, intermediate_channels, out_channels) -> None:
        super().__init__()

        if out_channels == 1:  # instance center head
            self.conv1 = Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(in_channels),
                activation=nn.ReLU(),
            )
            self.conv2 = Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(intermediate_channels),
                activation=nn.ReLU(),
            )
        else:  # instance offset head and semantics head
            self.conv1 = Conv2d(
                in_channels,
                in_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                groups=in_channels,
                bias=False,
                norm=nn.BatchNorm2d(in_channels),
                activation=nn.ReLU(),
            )
            self.conv2 = Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=nn.BatchNorm2d(intermediate_channels),
                activation=nn.ReLU(),
            )
        self.conv3 = Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Head Module.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            out: Segmentation Head output
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = nn.functional.interpolate(out, scale_factor=4, mode="bilinear")
        return out
