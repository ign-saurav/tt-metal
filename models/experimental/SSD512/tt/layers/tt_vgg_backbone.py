# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d, TtMaxPool2d, MaxPool2dConfiguration


class Conv2dNormActivation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        activation_layer=None,
    ):
        # if activation_layer == ttnn.relu:
        #     # self.activation_layer = None
        #     activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        # else:
        #     self.activation_layer = activation_layer
        #     activation = None

        # self.conv_config = Conv2dConfiguration.from_torch(
        #     layer, input_height=input_height, input_width=input_width, batch_size=batch_size
        # )
        self.conv_config = conv_config
        self.activation_layer = activation_layer

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        # input_tensor = post_conv_reshape(input_tensor, out_height=_out_height, out_width=_out_width)
        if self.activation_layer is not None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


class Maxpool2DOperation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        # activation_layer=None,
    ):
        self.conv_config = conv_config
        # self.activation_layer = activation_layer

        # self.conv = TtConv2d(self.conv_config, device)
        self.pool = TtMaxPool2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        input_tensor = self.pool(input_tensor)

        return input_tensor


class TtVGGBackbone:
    def __init__(self, conv_config_layer, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        layers = []
        for i, conv_config in enumerate(conv_config_layer):
            # Explicitly distinguish between Conv2dNormActivation and Maxpool2DOperation by checking type or attribute unique to each
            if isinstance(conv_config, Conv2dConfiguration):
                # This is a conv config, instantiate Conv2dNormActivation
                layers.append(
                    Conv2dNormActivation(
                        device=device,
                        conv_config=conv_config,
                        activation_layer=ttnn.relu,
                    )
                )
            elif isinstance(conv_config, MaxPool2dConfiguration):
                # This is a maxpool config, instantiate Maxpool2DOperation
                layers.append(
                    Maxpool2DOperation(
                        device=device,
                        conv_config=conv_config,
                        # activation_layer=ttnn.relu,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer configuration found: {type(conv_config)}")

            if i > 2:
                break
        self.block = layers

    def __call__(self, device, input):
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

        return result
