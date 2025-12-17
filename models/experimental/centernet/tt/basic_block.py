# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
)


class TtBasicBlock:
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        dilation: int,
        parameters,
        device,
        batch_size: int = 1,
        input_height: int = 512,
        input_width: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.planes = planes

        self.out_h = (input_height - 1) // stride + 1
        self.out_w = (input_width - 1) // stride + 1

        self.conv1 = TtConv2d(
            self._make_config(
                parameters.conv1,
                batch_size,
                input_height,
                input_width,
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
                batch_size,
                self.out_h,
                self.out_w,
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
            sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
        )

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = ttnn.reshape(out, (self.batch_size, self.out_h, self.out_w, self.planes))
        out = ttnn.add(out, residual)
        out = ttnn.relu(out)

        return out
