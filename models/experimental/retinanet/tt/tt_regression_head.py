# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import os
from models.tt_cnn.tt.builder import (
    TtConv2d,
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
)


class Conv2dNormActivation:
    def __init__(
        self,
        device,
        parameters,
        model_config,
    ):
        self.device = device
        self.parameters = parameters
        self.model_config = model_config
        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "0") == "1"
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]
        self.conv_config = Conv2dConfiguration(
            input_height=parameters["input_height"],
            input_width=parameters["input_width"],
            in_channels=256,
            out_channels=256,
            batch_size=1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=1,
            dilation=(1, 1),
            weight=parameters["conv_weight"],
            bias=parameters["conv_bias"],
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, x):
        shape_list = list(x.shape)
        N, H_out, W_out, C = shape_list[-4:]

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        x = self.conv(x)

        if self.fallback_on_groupnorm:
            x_nchw = ttnn.to_torch(x).permute(0, 3, 1, 2)
            x_normalized = torch.nn.functional.group_norm(
                x_nchw, num_groups=32, weight=self.torch_norm_weight, bias=self.torch_norm_bias, eps=1e-5
            ).permute(0, 2, 3, 1)
            x = ttnn.from_torch(x_normalized, device=self.device, dtype=ttnn.bfloat16)
        else:
            x = ttnn.group_norm(x, num_groups=32, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5)

        return x


def ttnn_retinanet_regression_head(
    fpn_heads,
    parameters,
    device,
    model_config,
    num_anchors=9,
):
    all_bbox_regression = []

    for fpn_idx, x in enumerate(fpn_heads):
        N = x.shape[-4]
        H_actual = x.shape[-3]
        W_actual = x.shape[-2]
        C = x.shape[-1]

        for conv_idx in range(4):
            conv_key = f"conv_block_{fpn_idx}_{conv_idx}"

            parameters[conv_key]["input_height"] = H_actual
            parameters[conv_key]["input_width"] = W_actual

            conv_block = Conv2dNormActivation(
                device=device,
                parameters=parameters[conv_key],
                model_config=model_config,
            )
            x = conv_block(x)

        bbox_key = f"bbox_reg_{fpn_idx}"

        parameters[bbox_key]["input_height"] = H_actual
        parameters[bbox_key]["input_width"] = W_actual

        final_conv_config = Conv2dConfiguration(
            input_height=H_actual,
            input_width=W_actual,
            in_channels=256,
            out_channels=36,
            batch_size=1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=1,
            dilation=(1, 1),
            weight=parameters[bbox_key]["conv_weight"],
            bias=parameters[bbox_key]["conv_bias"],
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )

        tt_bbox_reg_conv = TtConv2d(final_conv_config, device)
        bbox_regression = tt_bbox_reg_conv(x)

        N, H_final, W_final, C_final = bbox_regression.shape
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final, W_final, num_anchors, 4))
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final * W_final * num_anchors, 4))
        all_bbox_regression.append(bbox_regression)

    output = ttnn.concat(all_bbox_regression, dim=1)
    return output
