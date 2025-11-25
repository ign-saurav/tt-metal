# SPDX-FileCopyrightText: © 2025
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Exact MMCV FPN ConvModule when norm_cfg=None and act_cfg=None.

    FPN in MMCV uses:
      conv -> (no norm) -> (no activation)
      bias=True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,  # <-- IMPORTANT
        )

    def forward(self, x):
        return self.conv(x)  # <-- ONLY conv, no BN, no ReLU


class FPN(nn.Module):
    """
    Standalone FPN identical in behavior to the MMDetection FPN.
    Supports:
        - lateral convs
        - top-down pathway
        - upsampling
        - extra convs (`on_input`, `on_lateral`, `on_output`)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.add_extra_convs = add_extra_convs

        # ---------------------------
        # Determine backbone end level
        # ---------------------------
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            assert end_level <= self.num_ins
            self.backbone_end_level = end_level
            assert num_outs == end_level - start_level

        # ---------------------------
        # Build lateral and fpn convs
        # ---------------------------
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(start_level, self.backbone_end_level):
            self.lateral_convs.append(ConvModule(in_channels[i], out_channels, kernel_size=1))
            self.fpn_convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1))

        # ---------------------------
        # Extra FPN levels (P6, P7…)
        # ---------------------------
        extra_levels = num_outs - (self.backbone_end_level - start_level)

        if extra_levels > 0:
            for i in range(extra_levels):
                if i == 0:
                    if add_extra_convs == "on_input":
                        in_ch = in_channels[self.backbone_end_level - 1]
                    elif add_extra_convs == "on_lateral":
                        in_ch = out_channels
                    else:  # on_output
                        in_ch = out_channels
                else:
                    in_ch = out_channels

                self.fpn_convs.append(ConvModule(in_ch, out_channels, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        """Build FPN feature pyramid."""

        assert len(inputs) == len(self.in_channels)

        # --------------------------------
        # Step 1: Build lateral connections
        # --------------------------------
        laterals = [l_conv(inputs[i + self.start_level]) for i, l_conv in enumerate(self.lateral_convs)]

        # --------------------------------
        # Step 2: Top-down pathway
        # --------------------------------
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")

        # --------------------------------
        # Step 3: Final FPN convs
        # --------------------------------
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # --------------------------------
        # Step 4: Extra levels
        # --------------------------------
        if self.num_outs > len(outs):
            # Case A: no extra convs → max-pooling
            if not self.add_extra_convs:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))

            # Case B: RetinaNet-style extra convs
            else:
                # first extra level
                if self.add_extra_convs == "on_input":
                    x = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    x = laterals[-1]
                else:  # "on_output"
                    x = outs[-1]

                outs.append(self.fpn_convs[len(laterals)](x))

                # remaining extra levels
                for i in range(len(laterals) + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        x = F.relu(outs[-1])
                    else:
                        x = outs[-1]
                    outs.append(self.fpn_convs[i](x))

        return tuple(outs)
