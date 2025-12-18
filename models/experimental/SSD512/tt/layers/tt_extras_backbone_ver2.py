# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
import torch


@dataclass
class ExtraBlockConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


# class ConvBlock:
#     def __init__(
#         self,
#         input_height: int,
#         input_width: int,
#         config: ExtraBlockConfig,
#         batch_size: int,
#         parameters: dict,
#         device,
#     ):
#         self.config = config
#         self.out_height = (input_height + 2 * config.padding - config.kernel_size) // config.stride + 1
#         self.out_width = (input_width + 2 * config.padding - config.kernel_size) // config.stride + 1

#         conv_cfg = _create_conv_config_from_params(
#             input_height=input_height,
#             input_width=input_width,
#             in_channels=config.in_channels,
#             out_channels=config.out_channels,
#             batch_size=batch_size,
#             parameters=parameters,
#             device=device,
#             kernel_size=(config.kernel_size, config.kernel_size),
#             stride=(config.stride, config.stride),
#             padding=(config.padding, config.padding),
#         )
#         self.conv = TtConv2d(conv_cfg, device)

#     def __call__(self, x):
#         x = self.conv(x)
#         if x.is_sharded():
#             x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
#         x = ttnn.relu(x)
#         return x, self.out_height, self.out_width


class Conv2dNormActivation:
    def __init__(
        self,
        # kernel_size: Union[int, Tuple[int, ...]] = 3,
        # stride: Union[int, Tuple[int, ...]] = 1,
        # padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        # dilation=1,
        # groups=1,
        # input_shape,
        layer,
        # parameters=None,
        device=None,
        batch_size=1,
        # device=None,
        input_height=1,
        input_width=1,
        activation_layer=ttnn.relu,
    ):
        if activation_layer == ttnn.relu:
            self.activation_layer = None
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        else:
            self.activation_layer = activation_layer
            activation = None

        # # Normalize integer parameters to tuples
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)
        # if isinstance(stride, int):
        #     stride = (stride, stride)
        # if isinstance(dilation, int):
        #     dilation = (dilation, dilation)

        # if padding is None:
        #     padding = (kernel_size[0] - 1) // 2 * dilation[0]

        # # Normalize padding to tuple if it's an integer
        # if isinstance(padding, int):
        #     padding = (padding, padding)
        self.conv_config = Conv2dConfiguration.from_torch(
            layer, input_height=input_height, input_width=input_width, batch_size=batch_size
        )
        # self.conv_config = _create_conv_config_from_params(
        #     input_height=input_height,
        #     input_width=input_width,
        #     in_channels=parameters[0]["weight"].shape[1] * groups,
        #     out_channels=parameters[0]["weight"].shape[0],
        #     kernel_size=kernel_size,
        #     batch_size=1,
        #     parameters=parameters[0],
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     activation=activation,
        #     sharding_strategy=AutoShardedStrategyConfiguration(),
        # )
        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        # input_tensor = post_conv_reshape(input_tensor, out_height=_out_height, out_width=_out_width)
        if self.activation_layer is not None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


# class ExtraBlock:
#     def __init__(
#         self,
#         input_height: int,
#         input_width: int,
#         config: ExtraBlockConfig,
#         batch_size: int,
#         parameters: dict,
#         device,
#     ):
#         self.block = ConvBlock(input_height, input_width, config, batch_size, parameters, device)

#     def __call__(self, x):
#         return self.block(x)


class TtExtrasBackbone:
    # def __init__(self, size: int, input_channels: int, batch_size: int, parameters: list, device):
    def __init__(
        self, size: int, input_channels: int, batch_size: int, parameters: list, device, torch_model, torch_input
    ):
        self.size = size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels

        layers = []
        x = torch_input
        for i, layer in enumerate(torch_model):
            print(layer.__class__.__name__, x.shape)
            # parameters[i]['input_height']=x.shape[-2]
            # parameters[i]['input_width']=x.shape[-1]
            # parameters[i]['input_channel']=x.shape[-2]
            # myconf=Conv2dConfiguration.from_torch(layer, input_height=x.shape[-2], input_width=x.shape[-1], batch_size=batch_size)
            layers.append(
                Conv2dNormActivation(
                    layer,
                    # parameters=parameters,
                    device=device,
                    batch_size=batch_size,
                    input_height=x.shape[-2],
                    input_width=x.shape[-1]
                    # activation_layer=None,
                    # parameters=parameters[k],
                    # device=device,
                    # input_height=input_height // stride,
                    # input_width=input_width // stride,
                )
            )
            x = torch.nn.functional.relu(layer(x), inplace=True)
            # parameters[i]['output_channel']=x.shape[-3]
            # print(layer.)
        # torch_output = x
        self.block = layers

    def __call__(self, device, input):
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

        # if self.use_res_connect:
        #     result += input
        return result
