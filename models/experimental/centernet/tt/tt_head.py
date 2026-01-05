# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
)


class TtCenterNetHead:
    def __init__(
        self,
        in_channels: int,
        head_config: dict,
        head_conv: int,
        parameters,
        device,
        batch_size: int = 1,
        input_height: int = 128,
        input_width: int = 128,
    ):
        self.device = device
        self.batch_size = batch_size
        self.head_name = list(head_config.keys())[0]
        self.num_classes = head_config[self.head_name]

        self.out_h = input_height
        self.out_w = input_width

        if head_conv > 0:
            # Two-layer head: 3x3 conv + ReLU + 1x1 conv
            self.conv1 = TtConv2d(
                self._make_config(
                    parameters.conv1,
                    batch_size,
                    input_height,
                    input_width,
                    in_channels,
                    head_conv,
                    1,
                    1,
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    kernel_size=(3, 3),
                    padding=(1, 1),
                ),
                device,
            )

            self.conv2 = TtConv2d(
                self._make_config(
                    parameters.conv2,
                    batch_size,
                    input_height,
                    input_width,
                    head_conv,
                    self.num_classes,
                    1,
                    1,
                    activation=None,
                    kernel_size=(1, 1),
                    padding=(0, 0),
                ),
                device,
            )
            self.use_two_layers = True
        else:
            # Single-layer head: 1x1 conv only
            self.conv = TtConv2d(
                self._make_config(
                    parameters.conv,
                    batch_size,
                    input_height,
                    input_width,
                    in_channels,
                    self.num_classes,
                    1,
                    1,
                    activation=None,
                    kernel_size=(1, 1),
                    padding=(0, 0),
                ),
                device,
            )
            self.use_two_layers = False

    def _make_config(
        self, params, bs, h, w, in_ch, out_ch, stride, dilation, activation, kernel_size=(3, 3), padding=(1, 1)
    ):
        weight = params.weight
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)

        bias = getattr(params, "bias", None)
        if bias is not None and isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
            bias = ttnn.from_device(bias)

        return Conv2dConfiguration(
            input_height=h,
            input_width=w,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=bs,
            kernel_size=kernel_size,
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
        )

    def forward(self, x):
        try:
            if self.use_two_layers:
                out = self.conv1(x)
                if out is None:
                    raise ValueError("conv1 returned None")
                out = self.conv2(out)
                if out is None:
                    raise ValueError("conv2 returned None")
            else:
                out = self.conv(x)
                if out is None:
                    raise ValueError("conv returned None")

            return out
        except Exception as e:
            print(f"Error in TtCenterNetHead.forward: {e}")
            print(f"Input shape: {x.shape if x is not None else 'None'}")
            raise
