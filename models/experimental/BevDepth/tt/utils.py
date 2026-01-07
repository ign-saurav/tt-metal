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

    if conv_config is not None:
        activation = activation if activation is not None else conv_config.get("activation")
        shard_layout = shard_layout if shard_layout is not None else conv_config.get("shard_layout")
        deallocate_activation = conv_config.get("deallocate_activation", deallocate_activation)
        reallocate_halo_output = conv_config.get("reallocate_halo_output", reallocate_halo_output)
        enable_act_double_buffer = conv_config.get("enable_act_double_buffer", enable_act_double_buffer)
        enable_weights_double_buffer = conv_config.get("enable_weights_double_buffer", enable_weights_double_buffer)
        packer_l1_acc = conv_config.get("packer_l1_acc", packer_l1_acc)

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

    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.to_torch(weight)
    if bias is not None and isinstance(bias, ttnn.Tensor):
        bias = ttnn.to_torch(bias)
        if len(bias.shape) > 1:
            bias = bias.flatten()

    ttnn_weight, ttnn_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)

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
    if target_memory_config is not None:
        if tensor.memory_config() != target_memory_config:
            tensor = ttnn.to_memory_config(tensor, target_memory_config)
    elif reference_tensor is not None:
        if tensor.memory_config() != reference_tensor.memory_config():
            tensor = ttnn.to_memory_config(tensor, reference_tensor.memory_config())

    return tensor


def run_ttnn_inference(device, params, imgs, mats_dict):
    """
    Run TTNN inference on BEVDepth model.

    Args:
        device: TTNN device
        params: Model parameters dictionary
        imgs: Input images tensor
        mats_dict: Transformation matrices dictionary

    Returns:
        torch_preds: Predictions in torch format
    """
    from loguru import logger
    from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
    from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations

    logger.info("Running TTNN inference...")

    # Get actual image dimensions from input
    _, _, _, _, img_h, img_w = imgs.shape
    logger.info(f"TTNN input image size: {img_h}x{img_w}")

    # LSS configuration matching BEVDepth official config (256x704)
    lss_conf = {
        "x_bound": [-51.2, 51.2, 0.8],
        "y_bound": [-51.2, 51.2, 0.8],
        "z_bound": [-5.0, 3.0, 0.2],
        "d_bound": [2.0, 58.0, 0.5],
        "final_dim": [img_h, img_w],
        "downsample_factor": 16,
        "output_channels": 80,
    }

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "batch_size": 1,
        "neck_in_channels": [256, 512, 1024, 2048],
        "neck_out_channels": [128, 128, 128, 128],
        "neck_upsample_strides": [0.25, 0.5, 1, 2],
        "depthnet_in_channels": 512,
        "depthnet_mid_channels": 512,
        "depthnet_context_channels": 80,
        "depthnet_depth_channels": 112,
        "use_torch_fallback": True,
    }

    ttnn_backbone = TtBaseLSSFPN(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    head_model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
    }
    ttnn_head = TtBEVDepthHead(
        parameters=params["head"],
        model_config=head_model_config,
        layer_optimisations=head_optimisations,
        device=device,
    )

    ttnn_bev_feature = ttnn_backbone(imgs, mats_dict, is_return_depth=False)

    ttnn_bev_input = ttnn.from_torch(
        ttnn_bev_feature.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_bev_input = ttnn.to_device(ttnn_bev_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_output = ttnn_head(ttnn_bev_input, device=device)

    # Convert TTNN output to torch format
    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    torch_preds = []

    for task_idx in range(len(ttnn_output)):
        task_dict = {}
        for key in output_keys:
            ttnn_tensor, shape = ttnn_output[task_idx][key]
            tensor_torch = ttnn.to_torch(ttnn_tensor)
            # TTNN output is [N, H, W, C] format - permute to [N, C, H, W]
            if len(tensor_torch.shape) == 4:
                # shape is (out_h, out_w) from the new builder API
                tensor_torch = tensor_torch.permute(0, 3, 1, 2).contiguous()
            task_dict[key] = tensor_torch
        torch_preds.append([task_dict])

    return torch_preds
