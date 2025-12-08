# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


def create_conv2d_configuration(
    conv_args,
    conv_pth,
    device,
    activation=None,
    activation_dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat8_b,
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    is_blk=False,
    dealloc_act=False,
    act_block_h=None,
    model_configs: BevFormerV2ModelConfig | None = None,
    layer_path: str | None = None,
    **kwargs,
) -> Conv2dConfiguration:
    """Create a Conv2dConfiguration from conv_args and conv_pth, compatible with TtConv2d."""
    # Apply high-level configuration (if provided) before constructing TTNN configs.
    if model_configs is not None:
        settings = model_configs.get_effective_conv_settings(layer_path)
        # Config object supplies defaults; explicit arguments still win.
        if activation_dtype is ttnn.bfloat16:
            activation_dtype = settings.activation_dtype
        if weights_dtype is ttnn.bfloat8_b:
            weights_dtype = settings.weights_dtype
        if shard_layout is ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            shard_layout = settings.shard_layout
        if act_block_h is None:
            act_block_h = settings.act_block_h
        if dealloc_act is False:
            dealloc_act = settings.deallocate_activation

    if is_blk:
        shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    # Determine sharding strategy
    if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        sharding_strategy = BlockShardedStrategyConfiguration()
    elif shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if act_block_h is not None:
            sharding_strategy = HeightShardedStrategyConfiguration(act_block_h_override=act_block_h)
        else:
            sharding_strategy = HeightShardedStrategyConfiguration()
    else:
        sharding_strategy = AutoShardedStrategyConfiguration()

    # Get compute config settings
    if model_configs is not None:
        settings = model_configs.get_effective_conv_settings(layer_path)
        math_fidelity = settings.math_fidelity
        fp32_dest_acc_en = settings.fp32_dest_acc_en
        packer_l1_acc = settings.packer_l1_acc
        enable_act_double_buffer = settings.enable_act_double_buffer
    else:
        math_fidelity = ttnn.MathFidelity.HiFi4
        fp32_dest_acc_en = True
        packer_l1_acc = True
        enable_act_double_buffer = False

    # Extract conv parameters - handle both dict and object access
    if isinstance(conv_args, dict):
        # conv_args is already a dict with conv parameters
        conv = conv_args
    elif hasattr(conv_args, "conv"):
        # conv_args is an object with a .conv attribute (e.g., FPN case)
        conv = conv_args.conv
    else:
        # conv_args is the conv object itself
        conv = conv_args

    # Extract weight and bias from conv_pth (handles both dict and attribute access)
    if isinstance(conv_pth, dict):
        weight = conv_pth.get("weight", conv_pth)
        bias = conv_pth.get("bias", None)
    elif hasattr(conv_pth, "weight"):
        weight = conv_pth.weight
        bias = getattr(conv_pth, "bias", None)
    else:
        # conv_pth might be the weight tensor itself
        weight = conv_pth
        bias = None

    # Extract conv parameters - handle both dict and object access
    if isinstance(conv, dict):
        input_height = conv.get("input_height")
        input_width = conv.get("input_width")
        in_channels = conv.get("in_channels")
        out_channels = conv.get("out_channels")
        batch_size = conv.get("batch_size")
        kernel_size = conv.get("kernel_size")
        stride = conv.get("stride")
        padding = conv.get("padding")
        dilation = conv.get("dilation", (1, 1))
        groups = conv.get("groups", 1)
    else:
        # conv is an object with attributes
        input_height = conv.input_height
        input_width = conv.input_width
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        batch_size = conv.batch_size
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = getattr(conv, "dilation", (1, 1))
        groups = getattr(conv, "groups", 1)

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
        weight=weight,
        bias=bias,
        activation=activation,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        output_dtype=activation_dtype,
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        enable_act_double_buffer=enable_act_double_buffer,
        deallocate_activation=dealloc_act,
        **kwargs,
    )
