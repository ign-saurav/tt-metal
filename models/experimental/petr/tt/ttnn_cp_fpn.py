# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration


class ttnn_ConvModule:
    def __init__(
        self,
        conv_args=None,
        model_config=None,
        parameters=None,
        device=None,
    ):
        print(f"parameters: {parameters.keys()}")
        print(f"conv_args: {conv_args}")
        self.conv_config = Conv2dConfiguration.from_model_args(
            conv2d_args=conv_args,
            weights=parameters["weight"],
            bias=parameters["bias"],
            # **layer_optimisations.conv3,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv = TtConv2d(self.conv_config, device)

    def __call__(
        self,
        device,
        x,
    ):
        print(f"x shape: {x.shape}")
        x, [output_height, output_width] = self.conv(x, return_output_dim=True)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (self.batch_size, output_height, output_width, x.shape[3]))
        return x


class ttnn_CPFPN:
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        batch_size,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        upsample_cfg=dict(mode="nearest"),
        model_config=None,
        model_args=None,
        parameters=None,
        device=None,
    ):
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.device = device

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = []
        self.fpn_convs = []

        print(f"self.start_level: {self.start_level}")
        print(f"self.backbone_end_level: {self.backbone_end_level}")
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ttnn_ConvModule(
                conv_args=model_args["lateral_convs"][i]["conv"],
                model_config=model_config,
                device=device,
                parameters=parameters["lateral_convs"][i]["conv"],
            )
            self.lateral_convs.append(l_conv)
            if i == 0:
                fpn_conv = ttnn_ConvModule(
                    conv_args=model_args["fpn_convs"][i]["conv"],
                    model_config=model_config,
                    device=device,
                    parameters=parameters["fpn_convs"][i]["conv"],
                )
                self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ttnn_ConvModule(
                    conv_args=model_args["fpn_convs"][i]["conv"],
                    model_config=model_config,
                    device=device,
                    parameters=parameters["fpn_convs"][i]["conv"],
                )
                self.fpn_convs.append(extra_fpn_conv)

    def __call__(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(self.device, inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += ttnn.upsample(laterals[i], **self.upsample_cfg)
            else:
                tmp = ttnn.to_layout(
                    ttnn.upsample(
                        ttnn.to_layout(laterals[i], layout=ttnn.ROW_MAJOR_LAYOUT),
                        scale_factor=(2, 2),
                        **self.upsample_cfg,
                    ),
                    layout=ttnn.TILE_LAYOUT,
                )
                print(f"laterals[i - 1] shape: {laterals[i - 1].shape}")
                print(f"tmp shape: {tmp.shape}")
                laterals[i - 1] += tmp

        outs = [
            self.fpn_convs[i](self.device, laterals[i]) if i == 0 else laterals[i] for i in range(used_backbone_levels)
        ]

        return tuple(outs)
