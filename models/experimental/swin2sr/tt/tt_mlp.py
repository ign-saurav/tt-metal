# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtSwin2SRMLP:
    def __init__(self, device, parameters, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG):
        self.device = device
        self.parameters = parameters
        self.activation = activation
        self.memory_config = memory_config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

    def __call__(self, x):
        fc1_w = self.parameters.fc1.weight
        fc1_b = getattr(self.parameters.fc1, "bias", None)

        x = ttnn.linear(
            x,
            fc1_w,
            bias=fc1_b,
            activation=self.activation,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
        )

        fc2_w = self.parameters.fc2.weight
        fc2_b = getattr(self.parameters.fc2, "bias", None)

        x = ttnn.linear(
            x,
            fc2_w,
            bias=fc2_b,
            activation=None,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
        )

        return x
