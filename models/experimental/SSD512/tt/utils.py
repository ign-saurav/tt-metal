# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    L1FullSliceStrategyConfiguration,
    AutoShardedStrategyConfiguration,
)
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
import torch.nn as nn


def post_conv_reshape(x, out_height=1, out_width=1):
    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], out_height, out_width, x.shape[3]))
    return ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)


conv_config_optmised = {
    "weights_dtype": ttnn.bfloat8_b,
    "output_dtype": ttnn.bfloat8_b,
    "activation_dtype": ttnn.bfloat8_b,
    "math_fidelity": ttnn.MathFidelity.LoFi,
    "sharding_strategy": AutoShardedStrategyConfiguration(),
    "slice_strategy": L1FullSliceStrategyConfiguration(),
    # "deallocate_activation": True,
}
# Optimized config for MaxPool2d layers (only includes applicable parameters)
pool_config = {
    "dtype": ttnn.bfloat16,  # Map activation_dtype to dtype for MaxPool2d
    "slice_strategy": L1FullSliceStrategyConfiguration(),
}


def create_config_layers(torch_model, torch_input, model_config=conv_config_optmised, return_out=False):
    conv_config_layers = []
    # with torch.no_grad():
    x = torch_input
    for i, layer in enumerate(torch_model):
        # Create Conv2dConfiguration from the current layer, given torch input height, width, batch_size
        if isinstance(layer, nn.Conv2d):
            conv_config_layers.append(
                Conv2dConfiguration.from_torch(
                    layer,
                    input_height=x.shape[-2],
                    input_width=x.shape[-1],
                    batch_size=x.shape[0],
                    **model_config,
                )
            )
        elif isinstance(layer, nn.MaxPool2d):
            conv_config_layers.append(
                MaxPool2dConfiguration.from_torch(
                    layer,
                    input_height=x.shape[-2],
                    input_width=x.shape[-1],
                    channels=x.shape[-3],
                    batch_size=x.shape[0],
                    **pool_config,
                )
            )
        x = layer(x)
    if return_out:
        return conv_config_layers, x
    return conv_config_layers


class Conv2dNormActivation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        activation_layer=None,
    ):
        self.conv_config = conv_config
        self.activation_layer = activation_layer

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        if self.activation_layer is not None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


class Maxpool2DOperation:
    def __init__(
        self,
        device=None,
        conv_config=None,
    ):
        self.conv_config = conv_config

        self.pool = TtMaxPool2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        input_tensor = self.pool(input_tensor)

        return input_tensor
