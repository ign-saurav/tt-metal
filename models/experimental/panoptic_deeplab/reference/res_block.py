# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from torch import Tensor


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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, 1, bias=False), nn.BatchNorm2d(in_channels // 8), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, 5, 1, 2, 1, intermediate_channels, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_channels, out_channels, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()
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
        out_ = self.conv1(res2)
        out = torch.cat((out_, out), dim=1)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
