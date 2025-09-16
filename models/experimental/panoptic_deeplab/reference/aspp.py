# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from torch import Tensor


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ASPPModel(torch.nn.Module):
    """
    ASPP MODULE
    The input and output of `forward()` method must be NCHW tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ASPP_0_Conv = Conv2d(2048, 256, 1, 1, bias=False, norm=nn.BatchNorm2d(256), activation=nn.ReLU())
        self.ASPP_1_Depthwise = Conv2d(
            2048, 2048, 3, 1, 6, 6, 2048, bias=False, norm=nn.BatchNorm2d(2048), activation=nn.ReLU()
        )
        self.ASPP_1_pointwise = Conv2d(2048, 256, 1, 1, bias=False, norm=nn.BatchNorm2d(256), activation=nn.ReLU())

        self.ASPP_2_Depthwise = Conv2d(
            2048, 2048, 3, 1, 12, 12, 2048, bias=False, norm=nn.BatchNorm2d(2048), activation=nn.ReLU()
        )
        self.ASPP_2_pointwise = Conv2d(2048, 256, 1, 1, bias=False, norm=nn.BatchNorm2d(256), activation=nn.ReLU())

        self.ASPP_3_Depthwise = Conv2d(
            2048, 2048, 3, 1, 18, 18, 2048, bias=False, norm=nn.BatchNorm2d(2048), activation=nn.ReLU()
        )
        self.ASPP_3_pointwise = Conv2d(2048, 256, 1, 1, bias=False, norm=nn.BatchNorm2d(256), activation=nn.ReLU())

        self.ASPP_4_avg_pool = torch.nn.AvgPool2d((32, 64), stride=1, count_include_pad=True)
        self.ASPP_4_Conv_1 = Conv2d(2048, 256, 1, 1, activation=nn.ReLU())

        self.ASPP_project = Conv2d(1280, 256, 1, 1, bias=False, norm=nn.BatchNorm2d(256), activation=nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of ASPP Module.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            out: ASPP output
        """
        t0 = self.ASPP_0_Conv(x)
        t1 = self.ASPP_1_Depthwise(x)
        t2 = self.ASPP_2_Depthwise(x)
        t3 = self.ASPP_3_Depthwise(x)
        t4 = self.ASPP_4_avg_pool(x)

        t4 = self.ASPP_4_Conv_1(t4)
        t4 = nn.functional.interpolate(t4, (32, 64), mode="bilinear")

        t1 = self.ASPP_1_pointwise(t1)
        t2 = self.ASPP_2_pointwise(t2)
        t3 = self.ASPP_3_pointwise(t3)

        out = torch.cat((t0, t1, t2, t3, t4), dim=1)
        out = self.ASPP_project(out)

        return out
