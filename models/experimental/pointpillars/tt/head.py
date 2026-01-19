# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
)


class TtHead:
    def __init__(
        self,
        in_channel,
        n_anchors,
        n_classes,
        parameters,
        device,
        batch_size=1,
        input_height=248,
        input_width=216,
        dtype=ttnn.bfloat16,
    ):
        self.dtype = dtype

        self.conv_cls = TtConv2d(
            self._create_conv_config(
                parameters=parameters["conv_cls"],
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                in_channels=in_channel,
                out_channels=n_anchors * n_classes,
            ),
            device,
        )

        self.conv_reg = TtConv2d(
            self._create_conv_config(
                parameters=parameters["conv_reg"],
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                in_channels=in_channel,
                out_channels=n_anchors * 7,
            ),
            device,
        )

        self.conv_dir_cls = TtConv2d(
            self._create_conv_config(
                parameters=parameters["conv_dir_cls"],
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                in_channels=in_channel,
                out_channels=n_anchors * 2,
            ),
            device,
        )

    def _create_conv_config(
        self,
        parameters,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
    ):
        # Move weights from device to host for proper conv2d preparation
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)

        bias = None
        if hasattr(parameters, "bias") and parameters.bias is not None:
            bias = parameters.bias
            if isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
                bias = ttnn.from_device(bias)

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            weight=weight,
            bias=bias,
            activation=None,
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            deallocate_activation=False,
            enable_act_double_buffer=False,
        )

    def forward(self, x):
        bbox_cls_pred = self.conv_cls(x)
        bbox_cls_pred = ttnn.to_memory_config(bbox_cls_pred, ttnn.DRAM_MEMORY_CONFIG)

        bbox_pred = self.conv_reg(x)
        bbox_pred = ttnn.to_memory_config(bbox_pred, ttnn.DRAM_MEMORY_CONFIG)

        bbox_dir_cls_pred = self.conv_dir_cls(x)
        bbox_dir_cls_pred = ttnn.to_memory_config(bbox_dir_cls_pred, ttnn.DRAM_MEMORY_CONFIG)

        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
