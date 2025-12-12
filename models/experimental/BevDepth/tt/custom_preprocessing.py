# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

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
