# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
import torch


class Conv2dNormActivation:
    def __init__(
        self,
        layer,
        device=None,
        batch_size=1,
        input_height=1,
        input_width=1,
        activation_layer=None,
    ):
        # if activation_layer == ttnn.relu:
        #     # self.activation_layer = None
        #     activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        # else:
        #     self.activation_layer = activation_layer
        #     activation = None

        self.activation_layer = activation_layer
        self.conv_config = Conv2dConfiguration.from_torch(
            layer, input_height=input_height, input_width=input_width, batch_size=batch_size
        )

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        # input_tensor = post_conv_reshape(input_tensor, out_height=_out_height, out_width=_out_width)
        if self.activation_layer is not None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


class TtExtrasBackbone:
    # def __init__(self, size: int, input_channels: int, batch_size: int, parameters: list, device):
    def __init__(self, size: int, input_channels: int, batch_size: int, device, torch_model, torch_input):
        self.batch_size = batch_size
        self.device = device

        layers = []
        x = torch_input
        for i, layer in enumerate(torch_model):
            print(layer.__class__.__name__, x.shape)
            layers.append(
                Conv2dNormActivation(
                    layer,
                    device=device,
                    batch_size=batch_size,
                    input_height=x.shape[-2],
                    input_width=x.shape[-1],
                    activation_layer=ttnn.relu,
                )
            )
            x = torch.nn.functional.relu(layer(x), inplace=True)

        self.block = layers

    def __call__(self, device, input):
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

        return result
