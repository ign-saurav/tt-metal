# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from models.experimental.panoptic_deeplab.reference.utils import Conv2d, DepthwiseSeparableConv2d


class ASPPModel(torch.nn.Module):
    """
    ASPP MODULE
    The input and output of `forward()` method must be NCHW tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        dilations = [6, 12, 18]
        in_channels = 2048
        out_channels = 256
        pool_kernel_size = (32, 64)
        norm = nn.BatchNorm2d
        activation = nn.ReLU()
        use_bias = norm == ""
        self.convs = nn.ModuleList()

        # conv 1x1
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=nn.BatchNorm2d(out_channels),
                activation=deepcopy(activation),
            )
        )
        # atrous convs
        for dilation in dilations:
            self.convs.append(
                DepthwiseSeparableConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    norm1=nn.BatchNorm2d(in_channels),
                    activation1=deepcopy(activation),
                    norm2=nn.BatchNorm2d(out_channels),
                    activation2=deepcopy(activation),
                )
            )

        # image pooling
        # We do not add BatchNorm because the spatial resolution is 1x1,
        # the original TF implementation has BatchNorm.
        image_pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
            Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
        )
        self.convs.append(image_pooling)

        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=nn.BatchNorm2d(out_channels),
            activation=deepcopy(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of ASPP Module.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            out: ASPP output
        """
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = nn.functional.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res
