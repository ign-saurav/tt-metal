# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
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
    sharding_strategy=None,
    slice_strategy=None,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    config_tensors_in_dram=True,
) -> Conv2dConfiguration:
    import torch

    weight = parameters["weight"]
    bias = parameters.get("bias")

    # Convert TTNN tensors back to torch if needed (to avoid "bias not properly prepared" errors)
    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.to_torch(weight)

    if bias is not None and isinstance(bias, ttnn.Tensor):
        bias = ttnn.to_torch(bias)
        # Ensure bias is 1D
        if len(bias.shape) > 1:
            bias = bias.squeeze()

    # Ensure we have torch tensors
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"Weight must be torch.Tensor, got {type(weight)}")

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise ValueError(f"Bias must be torch.Tensor, got {type(bias)}")
        # Ensure bias is 1D
        if len(bias.shape) != 1:
            bias = bias.flatten()

    # Now convert to TTNN using the proper converter
    weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)

    # Move to device
    if device is not None:
        weight_ttnn = ttnn.to_device(weight_ttnn, device)
        if bias_ttnn is not None:
            bias_ttnn = ttnn.to_device(bias_ttnn, device)

    if sharding_strategy is None:
        sharding_strategy = AutoShardedStrategyConfiguration()

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
        slice_strategy=slice_strategy,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
        config_tensors_in_dram=config_tensors_in_dram,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def extract_extras_parameters_from_torch(torch_model):
    import torch.nn as nn

    parameters = []
    for torch_layer in torch_model:
        if isinstance(torch_layer, nn.Conv2d):
            # Clone to ensure we have independent tensors
            weight = torch_layer.weight.data.clone().detach()
            bias = torch_layer.bias.data.clone().detach() if torch_layer.bias is not None else None
            parameters.append({"weight": weight, "bias": bias})
    return parameters


def extract_vgg_parameters_from_torch(torch_model):
    import torch.nn as nn

    parameters = []
    for torch_layer in torch_model:
        if isinstance(torch_layer, nn.Conv2d):
            # Clone to ensure we have independent tensors
            weight = torch_layer.weight.data.clone().detach()
            bias = torch_layer.bias.data.clone().detach() if torch_layer.bias is not None else None
            parameters.append({"weight": weight, "bias": bias})
    return parameters


def extract_multibox_parameters_from_torch(torch_model):
    import torch.nn as nn

    parameters = []
    for torch_layer in torch_model:
        if isinstance(torch_layer, nn.Conv2d):
            # Clone to ensure we have independent tensors
            weight = torch_layer.weight.data.clone().detach()
            bias = torch_layer.bias.data.clone().detach() if torch_layer.bias is not None else None
            parameters.append({"weight": weight, "bias": bias})
    return parameters


def extract_extras_parameters_from_torch(torch_model):
    import torch.nn as nn

    parameters = []
    for torch_layer in torch_model:
        if isinstance(torch_layer, nn.Conv2d):
            # Clone to ensure we have independent tensors
            weight = torch_layer.weight.data.clone().detach()
            bias = torch_layer.bias.data.clone().detach() if torch_layer.bias is not None else None
            parameters.append({"weight": weight, "bias": bias})
    return parameters
