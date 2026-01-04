# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger

from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

from mmcv.cnn import ConvModule
from models.experimental.BevDepth.reference.bevdepth.layers.heads.resnet import BasicBlock
from models.experimental.BevDepth.reference.bevdepth.layers.necks.second_fpn import SECONDFPN
from models.experimental.BevDepth.reference.bevdepth.layers.heads.resnet import ResNet
from models.experimental.BevDepth.reference.bevdepth.layers.heads.bev_depth_head import BEVDepthHead


def fold_batch_norm2d_into_conv_transpose2d(conv_transpose, bn, mesh_mapper=None):
    """Fold BatchNorm2d parameters into ConvTranspose2d weights and bias

    Note: ConvTranspose2d weight shape is (in_channels, out_channels, kernel_h, kernel_w)
    while Conv2d weight shape is (out_channels, in_channels, kernel_h, kernel_w).
    So we need to apply the scale to dimension 1 (out_channels) instead of dimension 0.
    """
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into ConvTranspose2d")

    weight = conv_transpose.weight.data
    running_mean = bn.running_mean
    running_var = bn.running_var.data
    eps = bn.eps
    scale = bn.weight.data
    shift = bn.bias.data

    # Fold batch norm into conv transpose weights
    # ConvTranspose2d weight shape: (in_channels, out_channels, kernel_h, kernel_w)
    # BatchNorm scale shape: (out_channels,)
    # Apply scale to dimension 1 (out_channels dimension)
    scale_factor = (scale / torch.sqrt(running_var + eps))[None, :, None, None]
    weight = weight * scale_factor
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))

    weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    return weight, bias


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    if isinstance(model, ConvModule):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        parameters["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    elif isinstance(model, BasicBlock):
        parameters["conv1"] = {}
        parameters["conv2"] = {}
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.norm1)
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2, model.norm2)
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["conv2"]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        if model.downsample is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
            parameters["downsample"]["bias"] = ttnn.from_torch(
                torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
            )
    elif isinstance(model, ResNet):
        parameters["conv1"] = {}
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        for child_name, child in model.named_children():
            if child_name in ["conv1", "bn1", "relu"]:
                continue
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    elif isinstance(model, SECONDFPN):
        for i, deblock in enumerate(model.deblocks):
            conv_transpose = deblock[0]
            bn = deblock[1]

            weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv_transpose2d(
                conv_transpose, bn, mesh_mapper=mesh_mapper
            )

            parameters[f"deblock_{i}"] = {}
            parameters[f"deblock_{i}"]["weight"] = weight_ttnn
            parameters[f"deblock_{i}"]["bias"] = bias_ttnn
    elif isinstance(
        model,
        (BEVDepthHead),
    ):
        # Let the sub-modules handle their own preprocessing
        for child_name, child in model.named_children():
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    elif isinstance(model, torch.nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            parameters["bias"] = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    elif isinstance(model, torch.nn.ConvTranspose2d):
        parameters["weight"] = ttnn.from_torch(model.weight, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            parameters["bias"] = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor


def fuse_conv_bn_weights_unified(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm into conv weights for inference."""
    conv_weight = conv_weight.float() if conv_weight.dtype != torch.float32 else conv_weight
    bn_weight = (
        bn_weight.float() if isinstance(bn_weight, torch.Tensor) and bn_weight.dtype != torch.float32 else bn_weight
    )
    bn_bias = bn_bias.float() if isinstance(bn_bias, torch.Tensor) and bn_bias.dtype != torch.float32 else bn_bias
    bn_mean = bn_mean.float() if isinstance(bn_mean, torch.Tensor) and bn_mean.dtype != torch.float32 else bn_mean
    bn_var = bn_var.float() if isinstance(bn_var, torch.Tensor) and bn_var.dtype != torch.float32 else bn_var

    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

    if conv_bias is not None:
        conv_bias = conv_bias.float() if conv_bias.dtype != torch.float32 else conv_bias
    else:
        conv_bias = torch.zeros(conv_weight.shape[0], dtype=torch.float32, device=conv_weight.device)

    bn_bias_val = bn_bias if bn_bias is not None else torch.zeros_like(bn_mean)
    fused_bias = bn_bias_val + scale * (conv_bias - bn_mean)

    return fused_weight, fused_bias


def prepare_depthnet_parameters(state_dict, in_channels=512, mid_channels=256, depth_channels=112):
    class Parameters:
        pass

    params = Parameters()

    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.depth_net.",
        "img_backbone.depth_net.",
        "backbone.depth_net.",
        "depth_net.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find depth_net prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No depth_net keys found in checkpoint")

    try:
        reduce_conv_weight = state_dict[f"{prefix}reduce_conv.0.weight"].float()
        reduce_conv_bias = state_dict.get(f"{prefix}reduce_conv.0.bias", None)
        reduce_bn_weight = state_dict.get(f"{prefix}reduce_conv.1.weight", None)
        reduce_bn_bias = state_dict.get(f"{prefix}reduce_conv.1.bias", None)
        reduce_bn_mean = state_dict.get(f"{prefix}reduce_conv.1.running_mean", None)
        reduce_bn_var = state_dict.get(f"{prefix}reduce_conv.1.running_var", None)

        if reduce_bn_weight is not None and reduce_bn_mean is not None and reduce_bn_var is not None:
            reduce_bn_eps = state_dict.get(f"{prefix}reduce_conv.1.eps", 1e-5)
            if isinstance(reduce_bn_eps, torch.Tensor):
                reduce_bn_eps = reduce_bn_eps.item()
            fused_reduce_weight, fused_reduce_bias = fuse_conv_bn_weights_unified(
                reduce_conv_weight,
                reduce_conv_bias,
                reduce_bn_weight,
                reduce_bn_bias,
                reduce_bn_mean,
                reduce_bn_var,
                eps=reduce_bn_eps,
            )
            params.reduce_weight = fused_reduce_weight.to(torch.bfloat16)
            params.reduce_bias = fused_reduce_bias.to(torch.bfloat16)
        else:
            params.reduce_weight = reduce_conv_weight.to(torch.bfloat16)
            params.reduce_bias = reduce_conv_bias.to(torch.bfloat16) if reduce_conv_bias is not None else None
    except KeyError as e:
        logger.error(f"Failed to load reduce_conv: {e}")
        raise

    params.depth_mlp = Parameters()
    params.depth_mlp.fc1_weight = state_dict[f"{prefix}depth_mlp.fc1.weight"].to(torch.bfloat16)
    params.depth_mlp.fc1_bias = state_dict.get(f"{prefix}depth_mlp.fc1.bias", None)
    if params.depth_mlp.fc1_bias is not None:
        params.depth_mlp.fc1_bias = params.depth_mlp.fc1_bias.to(torch.bfloat16)
    params.depth_mlp.fc2_weight = state_dict[f"{prefix}depth_mlp.fc2.weight"].to(torch.bfloat16)
    params.depth_mlp.fc2_bias = state_dict.get(f"{prefix}depth_mlp.fc2.bias", None)
    if params.depth_mlp.fc2_bias is not None:
        params.depth_mlp.fc2_bias = params.depth_mlp.fc2_bias.to(torch.bfloat16)

    params.context_mlp = Parameters()
    params.context_mlp.fc1_weight = state_dict[f"{prefix}context_mlp.fc1.weight"].to(torch.bfloat16)
    params.context_mlp.fc1_bias = state_dict.get(f"{prefix}context_mlp.fc1.bias", None)
    if params.context_mlp.fc1_bias is not None:
        params.context_mlp.fc1_bias = params.context_mlp.fc1_bias.to(torch.bfloat16)
    params.context_mlp.fc2_weight = state_dict[f"{prefix}context_mlp.fc2.weight"].to(torch.bfloat16)
    params.context_mlp.fc2_bias = state_dict.get(f"{prefix}context_mlp.fc2.bias", None)
    if params.context_mlp.fc2_bias is not None:
        params.context_mlp.fc2_bias = params.context_mlp.fc2_bias.to(torch.bfloat16)

    params.depth_se = Parameters()
    params.depth_se.conv_reduce_weight = state_dict[f"{prefix}depth_se.conv_reduce.weight"].to(torch.bfloat16)
    params.depth_se.conv_reduce_bias = state_dict.get(f"{prefix}depth_se.conv_reduce.bias", None)
    if params.depth_se.conv_reduce_bias is not None:
        params.depth_se.conv_reduce_bias = params.depth_se.conv_reduce_bias.to(torch.bfloat16)
    params.depth_se.conv_expand_weight = state_dict[f"{prefix}depth_se.conv_expand.weight"].to(torch.bfloat16)
    params.depth_se.conv_expand_bias = state_dict.get(f"{prefix}depth_se.conv_expand.bias", None)
    if params.depth_se.conv_expand_bias is not None:
        params.depth_se.conv_expand_bias = params.depth_se.conv_expand_bias.to(torch.bfloat16)

    params.context_se = Parameters()
    params.context_se.conv_reduce_weight = state_dict[f"{prefix}context_se.conv_reduce.weight"].to(torch.bfloat16)
    params.context_se.conv_reduce_bias = state_dict.get(f"{prefix}context_se.conv_reduce.bias", None)
    if params.context_se.conv_reduce_bias is not None:
        params.context_se.conv_reduce_bias = params.context_se.conv_reduce_bias.to(torch.bfloat16)
    params.context_se.conv_expand_weight = state_dict[f"{prefix}context_se.conv_expand.weight"].to(torch.bfloat16)
    params.context_se.conv_expand_bias = state_dict.get(f"{prefix}context_se.conv_expand.bias", None)
    if params.context_se.conv_expand_bias is not None:
        params.context_se.conv_expand_bias = params.context_se.conv_expand_bias.to(torch.bfloat16)

    params.mlp_bn = Parameters()
    params.mlp_bn.weight = state_dict.get(f"{prefix}bn.weight", None)
    params.mlp_bn.bias = state_dict.get(f"{prefix}bn.bias", None)
    params.mlp_bn.running_mean = state_dict.get(f"{prefix}bn.running_mean", None)
    params.mlp_bn.running_var = state_dict.get(f"{prefix}bn.running_var", None)
    params.mlp_bn.eps = 1e-5

    params.context_weight = state_dict[f"{prefix}context_conv.weight"].to(torch.bfloat16)
    params.context_bias = state_dict.get(f"{prefix}context_conv.bias", None)
    if params.context_bias is not None:
        params.context_bias = params.context_bias.to(torch.bfloat16)

    for i in range(3):
        block = Parameters()

        conv1_weight = state_dict[f"{prefix}depth_conv.{i}.conv1.weight"].float()
        conv1_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv1.bias", None)
        if conv1_bias is not None:
            conv1_bias = conv1_bias.float()

        bn1_key_weight = f"{prefix}depth_conv.{i}.bn1.weight"
        bn1_key_bias = f"{prefix}depth_conv.{i}.bn1.bias"
        bn1_key_mean = f"{prefix}depth_conv.{i}.bn1.running_mean"
        bn1_key_var = f"{prefix}depth_conv.{i}.bn1.running_var"
        if bn1_key_weight not in state_dict:
            bn1_key_weight = f"{prefix}depth_conv.{i}.norm1.weight"
            bn1_key_bias = f"{prefix}depth_conv.{i}.norm1.bias"
            bn1_key_mean = f"{prefix}depth_conv.{i}.norm1.running_mean"
            bn1_key_var = f"{prefix}depth_conv.{i}.norm1.running_var"

        bn1_weight = state_dict.get(bn1_key_weight, None)
        bn1_bias = state_dict.get(bn1_key_bias, None)
        bn1_mean = state_dict.get(bn1_key_mean, None)
        bn1_var = state_dict.get(bn1_key_var, None)

        if bn1_weight is not None and bn1_mean is not None and bn1_var is not None:
            bn1_eps = state_dict.get(f"{prefix}depth_conv.{i}.bn1.eps", None)
            if bn1_eps is None:
                bn1_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm1.eps", 1e-5)
            if isinstance(bn1_eps, torch.Tensor):
                bn1_eps = bn1_eps.item()
            fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
                conv1_weight, conv1_bias, bn1_weight, bn1_bias, bn1_mean, bn1_var, eps=bn1_eps
            )
            block.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
            block.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
        else:
            block.conv1_weight = conv1_weight.to(torch.bfloat16)
            block.conv1_bias = conv1_bias.to(torch.bfloat16) if conv1_bias is not None else None

        conv2_weight = state_dict[f"{prefix}depth_conv.{i}.conv2.weight"].float()
        conv2_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv2.bias", None)
        if conv2_bias is not None:
            conv2_bias = conv2_bias.float()

        bn2_key_weight = f"{prefix}depth_conv.{i}.bn2.weight"
        bn2_key_bias = f"{prefix}depth_conv.{i}.bn2.bias"
        bn2_key_mean = f"{prefix}depth_conv.{i}.bn2.running_mean"
        bn2_key_var = f"{prefix}depth_conv.{i}.bn2.running_var"
        if bn2_key_weight not in state_dict:
            bn2_key_weight = f"{prefix}depth_conv.{i}.norm2.weight"
            bn2_key_bias = f"{prefix}depth_conv.{i}.norm2.bias"
            bn2_key_mean = f"{prefix}depth_conv.{i}.norm2.running_mean"
            bn2_key_var = f"{prefix}depth_conv.{i}.norm2.running_var"

        bn2_weight = state_dict.get(bn2_key_weight, None)
        bn2_bias = state_dict.get(bn2_key_bias, None)
        bn2_mean = state_dict.get(bn2_key_mean, None)
        bn2_var = state_dict.get(bn2_key_var, None)

        if bn2_weight is not None and bn2_mean is not None and bn2_var is not None:
            bn2_eps = state_dict.get(f"{prefix}depth_conv.{i}.bn2.eps", None)
            if bn2_eps is None:
                bn2_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm2.eps", 1e-5)
            if isinstance(bn2_eps, torch.Tensor):
                bn2_eps = bn2_eps.item()
            fused_conv2_weight, fused_conv2_bias = fuse_conv_bn_weights_unified(
                conv2_weight, conv2_bias, bn2_weight, bn2_bias, bn2_mean, bn2_var, eps=bn2_eps
            )
            block.conv2_weight = fused_conv2_weight.to(torch.bfloat16)
            block.conv2_bias = fused_conv2_bias.to(torch.bfloat16)
        else:
            block.conv2_weight = conv2_weight.to(torch.bfloat16)
            block.conv2_bias = conv2_bias.to(torch.bfloat16) if conv2_bias is not None else None

        setattr(params, f"block{i+1}", block)

    params.aspp = Parameters()

    for branch_idx, branch_name in enumerate(["aspp1", "aspp2", "aspp3", "aspp4"], 1):
        atrous_weight = state_dict[f"{prefix}depth_conv.3.{branch_name}.atrous_conv.weight"].float()
        bn_key_weight = f"{prefix}depth_conv.3.{branch_name}.bn.weight"
        bn_key_bias = f"{prefix}depth_conv.3.{branch_name}.bn.bias"
        bn_key_mean = f"{prefix}depth_conv.3.{branch_name}.bn.running_mean"
        bn_key_var = f"{prefix}depth_conv.3.{branch_name}.bn.running_var"

        bn_weight = state_dict.get(bn_key_weight, None)
        bn_bias = state_dict.get(bn_key_bias, None)
        bn_mean = state_dict.get(bn_key_mean, None)
        bn_var = state_dict.get(bn_key_var, None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            fused_weight, fused_bias = fuse_conv_bn_weights_unified(
                atrous_weight, None, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5
            )
            setattr(params.aspp, f"{branch_name}_weight", fused_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", fused_bias.to(torch.bfloat16))
        else:
            setattr(params.aspp, f"{branch_name}_weight", atrous_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", None)

    global_weight = state_dict[f"{prefix}depth_conv.3.global_avg_pool.1.weight"].float()
    global_bn_weight = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.weight", None)
    global_bn_bias = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.bias", None)
    global_bn_mean = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_mean", None)
    global_bn_var = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_var", None)

    if global_bn_weight is not None and global_bn_mean is not None and global_bn_var is not None:
        fused_global_weight, fused_global_bias = fuse_conv_bn_weights_unified(
            global_weight, None, global_bn_weight, global_bn_bias, global_bn_mean, global_bn_var, eps=1e-5
        )
        params.aspp.global_weight = fused_global_weight.to(torch.bfloat16)
        params.aspp.global_bias = fused_global_bias.to(torch.bfloat16)
    else:
        params.aspp.global_weight = global_weight.to(torch.bfloat16)
        params.aspp.global_bias = None

    conv1_weight = state_dict[f"{prefix}depth_conv.3.conv1.weight"].float()
    conv1_bn_weight = state_dict.get(f"{prefix}depth_conv.3.bn1.weight", None)
    conv1_bn_bias = state_dict.get(f"{prefix}depth_conv.3.bn1.bias", None)
    conv1_bn_mean = state_dict.get(f"{prefix}depth_conv.3.bn1.running_mean", None)
    conv1_bn_var = state_dict.get(f"{prefix}depth_conv.3.bn1.running_var", None)

    if conv1_bn_weight is not None and conv1_bn_mean is not None and conv1_bn_var is not None:
        fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
            conv1_weight, None, conv1_bn_weight, conv1_bn_bias, conv1_bn_mean, conv1_bn_var, eps=1e-5
        )
        params.aspp.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
    else:
        params.aspp.conv1_weight = conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = None

    params.dcn_weight = state_dict[f"{prefix}depth_conv.4.weight"].to(torch.bfloat16)
    params.dcn_bias = state_dict.get(f"{prefix}depth_conv.4.bias", None)
    if params.dcn_bias is not None:
        params.dcn_bias = params.dcn_bias.to(torch.bfloat16)

    try:
        conv_offset_weight = state_dict[f"{prefix}depth_conv.4.conv_offset.weight"]
        conv_offset_bias = state_dict.get(f"{prefix}depth_conv.4.conv_offset.bias", None)
        offset_channels = conv_offset_weight.shape[0]
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=conv_offset_bias is not None
        )
        params.dcn_conv_offset.weight.data = conv_offset_weight
        if conv_offset_bias is not None:
            params.dcn_conv_offset.bias.data = conv_offset_bias
        params.dcn_conv_offset.eval()
    except KeyError:
        offset_channels = 18
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        params.dcn_conv_offset.weight.data.zero_()
        params.dcn_conv_offset.bias.data.zero_()
        params.dcn_conv_offset.eval()

    params.final_weight = state_dict[f"{prefix}depth_conv.5.weight"].to(torch.bfloat16)
    params.final_bias = state_dict.get(f"{prefix}depth_conv.5.bias", None)
    if params.final_bias is not None:
        params.final_bias = params.final_bias.to(torch.bfloat16)

    return params
