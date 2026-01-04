# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass

from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
)


@dataclass
class ConvTransposeConfig:
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    weight: ttnn.Tensor
    bias: ttnn.Tensor


def create_conv2d_config(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    kernel_size,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    model_config: dict = None,
    conv_config: dict = None,
    activation=None,
    math_fidelity=None,
    weights_dtype=None,
    activation_dtype=None,
    output_dtype=None,
    shard_layout=None,
    deallocate_activation=True,
    reallocate_halo_output=False,
    reshard_if_not_optimal=False,
    act_block_h_override=0,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    config_tensors_in_dram=True,
) -> Conv2dConfiguration:
    """Create a Conv2dConfiguration for use with TtConv2d builder API."""
    # Extract from model_config if provided
    if model_config is not None:
        math_fidelity = math_fidelity or model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4)
        weights_dtype = weights_dtype or model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16)
        activation_dtype = activation_dtype or model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16)
        output_dtype = output_dtype or model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16)
    else:
        math_fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
        weights_dtype = weights_dtype or ttnn.bfloat16
        activation_dtype = activation_dtype or ttnn.bfloat16
        output_dtype = output_dtype or ttnn.bfloat16

    # Extract from conv_config if provided
    if conv_config is not None:
        activation = activation if activation is not None else conv_config.get("activation")
        shard_layout = shard_layout if shard_layout is not None else conv_config.get("shard_layout")
        deallocate_activation = conv_config.get("deallocate_activation", deallocate_activation)
        reallocate_halo_output = conv_config.get("reallocate_halo_output", reallocate_halo_output)
        enable_act_double_buffer = conv_config.get("enable_act_double_buffer", enable_act_double_buffer)
        enable_weights_double_buffer = conv_config.get("enable_weights_double_buffer", enable_weights_double_buffer)
        packer_l1_acc = conv_config.get("packer_l1_acc", packer_l1_acc)

    # Normalize to tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 4:
        padding = (padding[0], padding[2])
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Handle weights - convert to TTNN format if needed
    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.to_torch(weight)
    if bias is not None and isinstance(bias, ttnn.Tensor):
        bias = ttnn.to_torch(bias)
        if len(bias.shape) > 1:
            bias = bias.flatten()

    ttnn_weight, ttnn_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)

    # Create sharding strategy
    sharding_strategy = AutoShardedStrategyConfiguration()
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        sharding_strategy = HeightShardedStrategyConfiguration(
            reshard_if_not_optimal=reshard_if_not_optimal,
            act_block_h_override=act_block_h_override,
        )
    elif shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        sharding_strategy = BlockShardedStrategyConfiguration(
            reshard_if_not_optimal=reshard_if_not_optimal,
            act_block_h_override=act_block_h_override,
        )

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight=ttnn_weight,
        bias=ttnn_bias,
        activation=activation,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        output_dtype=output_dtype,
        math_fidelity=math_fidelity,
        sharding_strategy=sharding_strategy,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=reallocate_halo_output,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        config_tensors_in_dram=config_tensors_in_dram,
    )


def create_maxpool_config(
    input_height: int,
    input_width: int,
    channels: int,
    batch_size: int,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=(0, 0),
    dilation=(1, 1),
    ceil_mode: bool = False,
    dtype=ttnn.bfloat16,
) -> MaxPool2dConfiguration:
    """Create a MaxPool2dConfiguration for use with TtMaxPool2d builder API."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return MaxPool2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        dtype=dtype,
    )


def post_process_conv_output(
    output_tensor,
    batch_size: int,
    out_height: int,
    out_width: int,
    out_channels: int = None,
    to_dram: bool = True,
    reshape_4d: bool = True,
):
    """Post-process conv2d output tensor - handle sharding and reshape."""
    if output_tensor.is_sharded():
        memory_config = ttnn.DRAM_MEMORY_CONFIG if to_dram else ttnn.L1_MEMORY_CONFIG
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config)

    if reshape_4d:
        channels = out_channels if out_channels is not None else output_tensor.shape[-1]
        shape = output_tensor.shape

        if len(shape) == 3:
            output_tensor = ttnn.reshape(output_tensor, (batch_size, out_height, out_width, channels))
        elif len(shape) == 4 and shape[1] == 1 and shape[2] == out_height * out_width:
            output_tensor = ttnn.reshape(output_tensor, (batch_size, out_height, out_width, channels))
        elif len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] == batch_size * out_height * out_width:
            output_tensor = ttnn.reshape(output_tensor, (batch_size, out_height, out_width, channels))

    return output_tensor


def run_conv2d_with_builder(
    device,
    input_tensor,
    weight,
    bias,
    batch_size: int,
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    kernel_size,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    activation=None,
    model_config: dict = None,
    shard_layout=None,
    deallocate_activation=True,
    reallocate_halo_output=False,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    act_block_h_override=0,
    conv_cache: dict = None,
    cache_key=None,
    post_process=True,
):
    """Run conv2d using the TtConv2d builder API with caching and post-processing."""
    if model_config is None:
        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

    if cache_key is None:
        cache_key = (batch_size, input_height, input_width, out_channels)

    tt_conv = None
    if conv_cache is not None and cache_key in conv_cache:
        tt_conv = conv_cache[cache_key]

    if tt_conv is None:
        config = create_conv2d_config(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            output_dtype=model_config["ACTIVATIONS_DTYPE"],
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            reallocate_halo_output=reallocate_halo_output,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
            act_block_h_override=act_block_h_override,
        )
        tt_conv = TtConv2d(config, device)
        if conv_cache is not None:
            conv_cache[cache_key] = tt_conv

    output_tensor, (out_height, out_width) = tt_conv(input_tensor, return_output_dim=True)

    if post_process:
        output_tensor = post_process_conv_output(
            output_tensor,
            batch_size=batch_size,
            out_height=out_height,
            out_width=out_width,
            out_channels=out_channels,
        )

    return output_tensor, out_height, out_width


def ensure_memory_config(tensor, target_memory_config=None, reference_tensor=None):
    """
    Ensure tensor has the specified memory config or matches reference tensor.

    Args:
        tensor: Tensor to check/convert
        target_memory_config: Target memory config (if specified)
        reference_tensor: Reference tensor to match memory config (if target not specified)

    Returns:
        Tensor with correct memory config
    """
    if target_memory_config is not None:
        if tensor.memory_config() != target_memory_config:
            tensor = ttnn.to_memory_config(tensor, target_memory_config)
    elif reference_tensor is not None:
        if tensor.memory_config() != reference_tensor.memory_config():
            tensor = ttnn.to_memory_config(tensor, reference_tensor.memory_config())

    return tensor
