# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
    MaxPool2dConfiguration,
)
from models.experimental.centernet.tt.tree import TtTree


class TtDLA(LightweightModule):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=None,
        residual_root=False,
        return_levels=False,
        pool_size=7,
        linear_root=False,
        parameters=None,
        device=None,
        layer_args=None,
    ):
        super(TtDLA, self).__init__()
        self.device = device
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

        self.layer_args = layer_args
        self.base_layer = self._create_base_layer(parameters.base_layer, layer_args.base_layer["0"])

        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], parameters=parameters.level0, layer_args=layer_args.level0
        )
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, parameters=parameters.level1, layer_args=layer_args.level1
        )

        self.level2 = TtTree(
            levels[2],
            block,
            channels[1],
            channels[2],
            stride=2,
            level_root=False,
            root_residual=residual_root,
            parameters=parameters.level2,
            device=device,
            layer_args=layer_args.level2,
        )
        self.level3 = TtTree(
            levels[3],
            block,
            channels[2],
            channels[3],
            stride=2,
            level_root=True,
            root_residual=residual_root,
            parameters=parameters.level3,
            device=device,
            layer_args=layer_args.level3,
        )
        self.level4 = TtTree(
            levels[4],
            block,
            channels[3],
            channels[4],
            stride=2,
            level_root=True,
            root_residual=residual_root,
            parameters=parameters.level4,
            device=device,
            layer_args=layer_args.level4,
        )
        self.level5 = TtTree(
            levels[5],
            block,
            channels[4],
            channels[5],
            stride=2,
            level_root=True,
            root_residual=residual_root,
            parameters=parameters.level5,
            device=device,
            layer_args=layer_args.level5,
        )

    def _create_base_layer(self, parameters, layer_args):
        """Create base layer with folded BatchNorm"""
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

        return TtConv2d(
            Conv2dConfiguration(
                input_height=layer_args.input_height,
                input_width=layer_args.input_width,
                in_channels=3,
                out_channels=self.channels[0],
                batch_size=layer_args.batch_size,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                weight=parameters.conv.weight,
                bias=parameters.conv.bias,
                activation=conv_config.activation,
                weights_dtype=conv_config.weights_dtype,
                output_dtype=ttnn.bfloat16,
                sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
                deallocate_activation=conv_config.deallocate_activation,
            ),
            self.device,
        )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, parameters=None, layer_args=None):
        """Create a sequence of conv layers"""
        layers = []

        for i in range(convs):
            weight, bias = parameters[f"conv{i}"].weight, parameters[f"conv{i}"].bias

            conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                deallocate_activation=True,
                enable_act_double_buffer=True,
            )

            layer_stride = stride if i == 0 else 1
            conv_layer = TtConv2d(
                Conv2dConfiguration(
                    input_height=layer_args["0"].input_height,
                    input_width=layer_args["0"].input_width,
                    in_channels=inplanes,
                    out_channels=planes,
                    batch_size=layer_args["0"].batch_size,
                    kernel_size=(3, 3),
                    stride=(layer_stride, layer_stride),
                    padding=(dilation, dilation),
                    dilation=(dilation, dilation),
                    weight=weight,
                    bias=bias,
                    activation=conv_config.activation,
                    weights_dtype=conv_config.weights_dtype,
                    output_dtype=ttnn.bfloat16,
                    sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    deallocate_activation=conv_config.deallocate_activation,
                ),
                self.device,
            )

            layers.append(conv_layer)
            inplanes = planes

        return layers

    def _create_avgpool(self, pool_size, parameters):
        """Create average pooling layer"""
        return TtMaxPool2d(
            MaxPool2dConfiguration(
                input_height=parameters.input_height,
                input_width=parameters.input_width,
                channels=self.channels[-1],
                batch_size=parameters.batch_size,
                kernel_size=(pool_size, pool_size),
                stride=(pool_size, pool_size),
            ),
            self.device,
        )

    def _create_fc_layer(self, parameters):
        """Create final classification layer"""
        weight, bias = parameters.weight, parameters.bias

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
        )

        return TtConv2d(
            Conv2dConfiguration(
                input_height=parameters.input_height,
                input_width=parameters.input_width,
                in_channels=self.channels[-1],
                out_channels=self.num_classes,
                batch_size=parameters.batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=weight,
                bias=bias,
                activation=None,
                weights_dtype=conv_config.weights_dtype,
                output_dtype=ttnn.bfloat16,
                sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                math_fidelity=conv_config.math_fidelity,
                fp32_dest_acc_en=conv_config.fp32_dest_acc_en,
                deallocate_activation=conv_config.deallocate_activation,
            ),
            self.device,
        )

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        x = self.level0[0](x)
        y.append(x)

        x = self.level1[0](x)
        y.append(x)

        x = self.level2(x)
        y.append(x)

        x = self.level3(x)
        y.append(x)

        x = self.level4(x)
        y.append(x)

        x = self.level5(x)
        y.append(x)

        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)

            x = self.fc(x)

            batch_size = x.shape[0]
            x = ttnn.reshape(x, (batch_size, -1))

            return x
