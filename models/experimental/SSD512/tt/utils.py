# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)


conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    device,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    sharding_strategy=AutoShardedStrategyConfiguration(),
) -> Conv2dConfiguration:
    weight = parameters["weight"]
    bias = parameters.get("bias")

    if not isinstance(weight, ttnn.Tensor):
        weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)
        if device is not None:
            weight_ttnn = ttnn.to_device(weight_ttnn, device)
            if bias_ttnn is not None:
                bias_ttnn = ttnn.to_device(bias_ttnn, device)
    else:
        weight_ttnn = weight
        bias_ttnn = bias

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=weight_ttnn,
        bias=bias_ttnn,
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
    )


# def extract_extras_parameters_from_torch(torch_model):
#     parameters = []
#     for torch_layer in torch_model:
#         weight = torch_layer.weight.data
#         bias = torch_layer.bias.data if torch_layer.bias is not None else None
#         parameters.append({"weight": weight, "bias": bias})
#     return parameters

# def extract_extras_parameters_from_torch(torch_model):
#     parameters = []
#     for torch_layer in torch_model:
#         layer_params = {
#             "weight": torch_layer.weight.data,
#             "bias": torch_layer.bias.data if torch_layer.bias is not None else None,
#         }

#         # Store conv metadata if this is a Conv2d
#         if isinstance(torch_layer, torch.nn.Conv2d):
#             layer_params.update(
#                 {
#                     "kernel_size": torch_layer.kernel_size,  # e.g. (3, 3)
#                     "stride": torch_layer.stride,            # e.g. (1, 1)
#                     "padding": torch_layer.padding,          # e.g. (1, 1)
#                     "dilation": torch_layer.dilation,
#                     "groups": torch_layer.groups,
#                 }
#             )

#         parameters.append(layer_params)

#     return parameters


import torch.nn as nn


def extract_extras_parameters_from_torch(torch_model, input_height, input_width, batch_size=1):
    """
    torch_model: typically torch_model.extras (nn.Sequential)
    input_height, input_width: spatial size of the tensor coming into extras
    """
    parameters = []
    h, w = input_height, input_width

    for layer in torch_model:
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            bias = layer.bias.data if layer.bias is not None else None

            # store weights + shape metadata
            layer_params = {
                "weight": weight,
                "bias": bias,
                "batch_size": batch_size,
                "input_height": h,
                "input_width": w,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
                "dilation": layer.dilation,
                "groups": layer.groups,
            }
            parameters.append(layer_params)

            # update h, w for the next layer using the conv2d formula
            kh, kw = (
                layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            )
            sh, sw = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            ph, pw = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            dh, dw = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)

            h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        else:
            # non-conv layers: assume no spatial change (adjust if you add pools, etc.)
            continue

    return parameters


# Helper function to create Conv2dConfiguration from parameters
def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    sharding_strategy=AutoShardedStrategyConfiguration(),
) -> Conv2dConfiguration:
    """
    Conv2dConfiguration from parameters dict for SqueezeExcitation.
    """

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
    )
