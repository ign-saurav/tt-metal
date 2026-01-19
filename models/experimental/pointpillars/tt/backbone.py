# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
)


class TtBackbone:
    def __init__(
        self,
        in_channel,
        out_channels,
        layer_nums,
        layer_strides,
        parameters,
        device,
        batch_size=1,
        input_height=1,
        input_width=432 * 496,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        current_height = input_height
        current_width = input_width
        current_channels = in_channel

        self.multi_blocks = []
        self.output_shapes = []

        for i, stride in enumerate(layer_strides):
            block_convs = []
            block_out_channels = out_channels[i]

            conv_config = self._create_conv_config(
                parameters=parameters[f"block_{i}"]["conv_0"],
                batch_size=batch_size,
                input_height=current_height,
                input_width=current_width,
                in_channels=current_channels,
                out_channels=block_out_channels,
                stride=(stride, stride),
                block_idx=i,
            )
            block_convs.append(TtConv2d(conv_config, device))

            current_height = current_height // stride
            current_width = current_width // stride
            current_channels = block_out_channels

            for j in range(layer_nums[i]):
                conv_config = self._create_conv_config(
                    parameters=parameters[f"block_{i}"][f"conv_{j+1}"],
                    batch_size=batch_size,
                    input_height=current_height,
                    input_width=current_width,
                    in_channels=current_channels,
                    out_channels=block_out_channels,
                    stride=(1, 1),
                    block_idx=i,
                )
                block_convs.append(TtConv2d(conv_config, device))

            self.multi_blocks.append(block_convs)
            self.output_shapes.append((batch_size, current_height, current_width, block_out_channels))

    def _create_conv_config(
        self,
        parameters,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        stride,
        block_idx,
    ):
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)

        bias = None
        if hasattr(parameters, "bias") and parameters.bias is not None:
            bias = parameters.bias
            if isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
                bias = ttnn.from_device(bias)

        math_fidelity = ttnn.MathFidelity.HiFi4 if block_idx == 2 else ttnn.MathFidelity.HiFi2

        act_block_h = 32 if out_channels >= 128 else 64

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            weight=weight,
            bias=bias,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=HeightShardedStrategyConfiguration(
                reshard_if_not_optimal=True, act_block_h_override=act_block_h
            ),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reallocate_halo_output=True,
        )

    def forward(self, x):
        """
        x: ttnn tensor (b, h, w, c) in NHWC format. Default: (1, 496, 432, 64)
        return: list[]. Default: [(1, 248, 216, 64), (1, 124, 108, 128), (1, 62, 54, 256)]
        """
        outs = []
        for i, block_convs in enumerate(self.multi_blocks):
            for conv in block_convs:
                x = conv(x)
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, self.output_shapes[i])
            outs.append(x)
        return outs
