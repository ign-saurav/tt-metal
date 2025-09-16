# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn
from torch import Tensor
from models.experimental.panoptic_deeplab.tt.common import Conv2d


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d(
            inplanes, width, kernel_size=1, stride=1, bias=False, norm=norm_layer(width), activation=self.relu
        )
        self.conv2 = Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
            norm=norm_layer(width),
            activation=self.relu,
        )
        self.conv3 = Conv2d(
            width,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=norm_layer(planes * self.expansion),
        )
        self.shortcut = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out
