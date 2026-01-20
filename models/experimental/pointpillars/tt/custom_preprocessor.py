# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import ModuleArgs, fold_batch_norm2d_into_conv2d
import torch
import ttnn
from models.experimental.pointpillars.reference.pointpillars import (
    PillarEncoder,
    Backbone,
    Neck,
    Head,
    PointPillars,
)
from ttnn.dot_access import make_dot_access_dict


class ConvArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_module_args(model):
    if isinstance(
        model,
        (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.ConvTranspose2d,
        ),
    ):
        if (model, torch.nn.ConvTranspose2d):
            return ConvArgs(
                in_channels=model.in_channels,
                out_channels=model.out_channels,
                kernel_size=model.kernel_size,
                stride=model.stride,
                padding=model.padding,
                dilation=model.dilation,
                groups=model.groups,
                padding_mode=model.padding_mode,
                output_padding=model.output_padding,
            )
        return ConvArgs(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
            groups=model.groups,
            padding_mode=model.padding_mode,
        )
    else:
        module_args = {}
        for child_name, child in model.named_children():
            module_args[child_name] = infer_module_args(child)

    return make_dot_access_dict(module_args, ignore_types=(ModuleArgs,))


def fold_batch_norm1d_into_conv1d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm1d must have track_running_stats=True to be folded into Conv1d")

    weight = conv.weight
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias

    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None]

    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    bias = bias.reshape(1, 1, 1, -1)

    return weight, bias


def fold_batch_norm2d_into_conv_transpose2d(conv_transpose, bn):
    """Fold BatchNorm2d into ConvTranspose2d weights."""
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into ConvTranspose2d")

    weight = conv_transpose.weight
    bias = conv_transpose.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias

    weight = weight * (scale / torch.sqrt(running_var + eps))[None, :, None, None]

    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


def _extract_backbone(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess Backbone parameters with fused BatchNorm."""
    assert isinstance(model, Backbone)

    for i in range(len(model.multi_blocks)):
        block = model.multi_blocks[i]
        parameters[f"block_{i}"] = {}

        conv_idx = 0
        for j in range(0, len(block), 3):
            conv_layer = block[j]
            bn_layer = block[j + 1]

            weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)

            parameters[f"block_{i}"][f"conv_{conv_idx}"] = {}
            parameters[f"block_{i}"][f"conv_{conv_idx}"]["weight"] = ttnn.from_torch(
                weight, dtype=dtype, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters[f"block_{i}"][f"conv_{conv_idx}"]["bias"] = ttnn.from_torch(
                bias, dtype=dtype, mesh_mapper=mesh_mapper
            )
            parameters[f"block_{i}"][f"conv_{conv_idx}"]["conv_args"] = infer_module_args(conv_layer)

            conv_idx += 1

    return parameters


def _extract_pillar_encoder(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess PillarEncoder parameters with fused BatchNorm."""
    assert isinstance(model, PillarEncoder)
    parameters["conv"] = {}

    weight, bias = fold_batch_norm1d_into_conv1d(model.conv, model.bn)
    parameters["conv"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
    parameters["conv"]["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
    parameters["conv_args"] = infer_module_args(model)

    return parameters


def _extract_neck(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess Neck parameters with fused BatchNorm."""
    assert isinstance(model, Neck)

    for i in range(len(model.decoder_blocks)):
        block = model.decoder_blocks[i]
        parameters[f"decoder_{i}"] = {}

        conv_transpose_layer = block[0]
        bn_layer = block[1]

        if i == 0:
            weight, bias = fold_batch_norm2d_into_conv_transpose2d(conv_transpose_layer, bn_layer)
            parameters[f"decoder_{i}"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
            bias = bias.reshape((1, 1, 1, -1))
            parameters[f"decoder_{i}"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)
            parameters[f"decoder_{i}"]["conv_args"] = infer_module_args(conv_transpose_layer)
        else:
            weight = conv_transpose_layer.weight
            parameters[f"decoder_{i}"]["weight"] = weight
            parameters[f"decoder_{i}"]["conv_args"] = infer_module_args(conv_transpose_layer)
            bias = conv_transpose_layer.bias

            if bias is not None:
                bias = bias.reshape((1, 1, 1, -1))
                parameters[f"decoder_{i}"]["bias"] = bias, dtype = dtype, mesh_mapper = mesh_mapper
            else:
                out_channels = conv_transpose_layer.out_channels
                zero_bias = torch.zeros(1, 1, 1, out_channels)
                parameters[f"decoder_{i}"]["bias"] = zero_bias

            parameters[f"decoder_{i}"]["bn_weight"] = ttnn.from_torch(
                bn_layer.weight.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters[f"decoder_{i}"]["bn_bias"] = ttnn.from_torch(
                bn_layer.bias.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters[f"decoder_{i}"]["bn_running_mean"] = ttnn.from_torch(
                bn_layer.running_mean.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters[f"decoder_{i}"]["bn_running_var"] = ttnn.from_torch(
                bn_layer.running_var.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )

    return parameters


def _extract_head(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess Head parameters."""
    assert isinstance(model, Head)

    # Process conv_cls
    parameters["conv_cls"] = {}
    parameters["conv_cls"]["weight"] = ttnn.from_torch(model.conv_cls.weight, dtype=dtype)
    bias = model.conv_cls.bias.reshape((1, 1, 1, -1))
    parameters["conv_cls"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
    parameters["conv_cls"]["conv_args"] = infer_module_args(model.conv_cls)

    # Process conv_reg
    parameters["conv_reg"] = {}
    parameters["conv_reg"]["weight"] = ttnn.from_torch(model.conv_reg.weight, dtype=dtype)
    bias = model.conv_reg.bias.reshape((1, 1, 1, -1))
    parameters["conv_reg"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
    parameters["conv_reg"]["conv_args"] = infer_module_args(model.conv_reg)

    # Process conv_dir_cls
    parameters["conv_dir_cls"] = {}
    parameters["conv_dir_cls"]["weight"] = ttnn.from_torch(model.conv_dir_cls.weight, dtype=dtype)
    bias = model.conv_dir_cls.bias.reshape((1, 1, 1, -1))
    parameters["conv_dir_cls"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
    parameters["conv_dir_cls"]["conv_args"] = infer_module_args(model.conv_dir_cls)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for PointPillars models."""
    parameters = {}
    weight_dtype = ttnn.bfloat16

    if isinstance(model, PointPillars):
        parameters["pillar_encoder"] = {}
        parameters["pillar_encoder"] = _extract_pillar_encoder(
            model.pillar_encoder, parameters["pillar_encoder"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )

        parameters["backbone"] = {}
        parameters["backbone"] = _extract_backbone(
            model.backbone, parameters["backbone"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )

        parameters["neck"] = {}
        parameters["neck"] = _extract_neck(model.neck, parameters["neck"], dtype=weight_dtype, mesh_mapper=mesh_mapper)

        parameters["head"] = {}
        parameters["head"] = _extract_head(model.head, parameters["head"], dtype=weight_dtype, mesh_mapper=mesh_mapper)

    if isinstance(model, PillarEncoder):
        parameters["pillar_encoder"] = {}
        parameters["pillar_encoder"] = _extract_pillar_encoder(
            model, parameters["pillar_encoder"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )

    elif isinstance(model, Backbone):
        parameters["backbone"] = {}
        parameters["backbone"] = _extract_backbone(
            model, parameters["backbone"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )

    elif isinstance(model, Neck):
        parameters["neck"] = {}
        parameters["neck"] = _extract_neck(model, parameters["neck"], dtype=weight_dtype, mesh_mapper=mesh_mapper)

    elif isinstance(model, Head):
        parameters["head"] = {}
        parameters["head"] = _extract_head(model, parameters["head"], dtype=weight_dtype, mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Create a custom preprocessor with mesh mapping support."""

    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
