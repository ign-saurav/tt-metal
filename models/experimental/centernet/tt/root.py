# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d, AutoShardedStrategyConfiguration
from models.common.lightweightmodule import LightweightModule


class TtRoot(LightweightModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        parameters,
        layer_args,
        device,
        residual: bool = False,
    ):
        super(TtRoot, self).__init__()
        self.device = device

        # Calculate padding based on kernel_size
        padding = (kernel_size - 1) // 2

        # Create convolution configuration
        self.conv = TtConv2d(
            self._make_config(
                parameters.conv,
                layer_args.conv.batch_size,
                layer_args.conv.input_height,
                layer_args.conv.input_width,
                in_channels,
                out_channels,
                stride=1,
                dilation=1,
                padding=padding,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                # activation=None,
            ),
            device,
        )

    def _make_config(self, params, bs, h, w, in_ch, out_ch, stride, dilation, padding, activation):
        weight = params.weight
        bias = params.bias
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)
        if isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
            bias = ttnn.from_device(bias)

        return Conv2dConfiguration(
            input_height=h,
            input_width=w,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=bs,
            kernel_size=(1, 1),  # 1x1 convolution
            stride=(stride, stride),
            padding=(padding, padding),
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

    def forward(self, *x):
        converted_tensors = []
        for tensor in x:
            if tensor.is_sharded():
                tensor = ttnn.sharded_to_interleaved(tensor, ttnn.L1_MEMORY_CONFIG)
            converted_tensors.append(tensor)

        x = ttnn.concat(converted_tensors, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.conv(x)

        return x
