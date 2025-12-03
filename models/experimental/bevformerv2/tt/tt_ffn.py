# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtFFN:
    """Feed Forward Network - ttnn implementation.

    Args:
        params: Parameters object containing weights and biases for linear layers
        device: Device to run operations on
    """

    def __init__(self, params, device):
        self.device = device
        self.params = params

        # Extract weights and biases from params
        # Assuming params structure: params.layers[0][0] (first linear), params.layers[1] (second linear)
        if hasattr(params, "layers"):
            self.linear1_weight = params.layers[0][0].weight
            self.linear1_bias = params.layers[0][0].bias
            self.linear2_weight = params.layers[1].weight
            self.linear2_bias = params.layers[1].bias
        else:
            # Alternative structure
            self.linear1_weight = params.linear1.weight
            self.linear1_bias = params.linear1.bias
            self.linear2_weight = params.linear2.weight
            self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        """Forward pass of FFN.

        Args:
            x: Input tensor
            identity: Identity tensor for residual connection

        Returns:
            Output tensor with residual connection
        """
        if identity is None:
            identity = x

        # First linear + ReLU
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)

        return x
