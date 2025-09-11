# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTChannelAttention(LightweightModule):
    def __init__(self, device, parameters, num_feat, squeeze_factor=16, memory_config=None):
        super().__init__()

        self.device = device
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.num_feat = num_feat
        self.squeeze_factor = squeeze_factor

        # Extract preprocessed parameters
        self.conv1_weight = parameters["conv1"]["weight"]
        self.conv1_bias = parameters["conv1"]["bias"]
        self.conv2_weight = parameters["conv2"]["weight"]
        self.conv2_bias = parameters["conv2"]["bias"]

    def forward(self, x):
        original_x = x
        original_shape = x.shape

        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.global_avg_pool2d(x, memory_config=self.memory_config)

        if original_shape[-1] == 180:
            x = ttnn.slice(
                x,
                starts=[0, 0, 0, 0],  # Start indices for each dimension
                ends=[original_shape[0], 1, 1, 180],  # End indices - slice to 180 in last dim
                steps=[1, 1, 1, 1],  # Step size for each dimension
            )

        x = ttnn.linear(
            x,
            self.conv1_weight,
            bias=self.conv1_bias,
            memory_config=self.memory_config,
            activation="relu",
        )

        x = ttnn.linear(
            x,
            self.conv2_weight,
            bias=self.conv2_bias,
            memory_config=self.memory_config,
        )

        # Sigmoid activation
        x = ttnn.sigmoid(x)

        batch_size, height, width, channels = original_shape
        attention_weights = ttnn.reshape(x, [batch_size, 1, 1, channels])
        attention_weights = ttnn.repeat(attention_weights, [1, height, width, 1], memory_config=self.memory_config)

        # Element-wise multiplication with original input
        output = ttnn.multiply(original_x, attention_weights, memory_config=self.memory_config)

        return output
