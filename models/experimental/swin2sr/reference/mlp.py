# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

###########################################################
# Adapted from https://github.com/mv-lab/swin2sr/tree/main
# licensed under Apache-2.0 license.
###########################################################


import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron module for Swin2SR.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout rate (deprecated, not used). Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, in_features).

        Returns:
            Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
