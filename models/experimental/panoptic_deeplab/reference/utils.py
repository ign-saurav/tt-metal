# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


# ---------------------------
# Torch utility modules
# ---------------------------


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
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseSeparableConv2d(torch.nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        *,
        norm1=None,
        activation1=None,
        norm2=None,
        activation2=None,
    ):
        """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
        """
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=not norm1,
            norm=norm1,
            activation=activation1,
        )
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=not norm2,
            norm=norm2,
            activation=activation2,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
