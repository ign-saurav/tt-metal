# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from typing import Tuple

from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel


class DecoderModel(torch.nn.Module):
    """
    Modular decoder architecture.
    The input and output of `forward()` method must be NCHW tensors.

    Args:
            name (string): name of segmentation head
    """

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.aspp = ASPPModel()
        if name == "semantic_decoder":
            self.res3 = ResModel(512, 320, 256)
            self.res2 = ResModel(256, 288, 256)
            self.head_1 = HeadModel(256, 256, 19)
        else:
            self.res3 = ResModel(512, 320, 128)
            self.res2 = ResModel(256, 160, 128)
            self.head_1 = HeadModel(128, 32, 2)
            self.head_2 = HeadModel(128, 32, 1)

    def forward(self, x: Tensor, res3: Tensor, res2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of Decoder Module.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            out:
            out_2:
        """
        out = self.aspp(x)
        out = self.res3(out, res3)
        out_ = self.res2(out, res2)
        out = self.head_1(out_)

        if self.name == "instance_decoder":
            out_2 = self.head_2(out_)
        else:
            out_2 = None
        return out, out_2
