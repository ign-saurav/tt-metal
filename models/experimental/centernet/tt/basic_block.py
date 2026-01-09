# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    Conv2dConfiguration,
    TtConv2d,
)
from models.common.lightweightmodule import LightweightModule


class TtBasicBlock(LightweightModule):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        dilation: int,
        parameters,
        device,
        layer_args,
    ):
        super(TtBasicBlock, self).__init__()
        self.device = device
        self.batch_size = layer_args.conv1.batch_size
        self.planes = planes

        self.conv1 = TtConv2d(
            self._make_config(
                parameters.conv1,
                layer_args.conv1.batch_size,
                layer_args.conv1.input_height,
                layer_args.conv1.input_width,
                inplanes,
                planes,
                stride,
                dilation,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            ),
            device,
        )

        self.conv2 = TtConv2d(
            self._make_config(
                parameters.conv2,
                layer_args.conv2.batch_size,
                layer_args.conv2.input_height,
                layer_args.conv2.input_width,
                planes,
                planes,
                1,
                dilation,
                activation=None,
            ),
            device,
        )

    def _make_config(self, params, bs, h, w, in_ch, out_ch, stride, dilation, activation):
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
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(dilation, dilation),
            dilation=(dilation, dilation),
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
        )

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = ttnn.reshape(out, residual.shape)
        out = ttnn.add(out, residual)
        out = ttnn.relu(out)

        return out
