# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTMlp(LightweightModule):
    def __init__(
        self, device, in_features, hidden_features=None, out_features=None, parameters=None, dtype=ttnn.bfloat16
    ):
        self.device = device

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dtype = dtype

        # Initialize weights and biases based on available inputs
        # Use preprocessed parameters
        self.fc1_weight = parameters["fc1"]["weight"]
        self.fc1_bias = parameters["fc1"]["bias"]
        self.fc2_weight = parameters["fc2"]["weight"]
        self.fc2_bias = parameters["fc2"]["bias"]

    def forward(self, x):
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="gelu",
            dtype=self.dtype,
        )

        x = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        return x
