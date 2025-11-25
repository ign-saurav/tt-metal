# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import (
    ModuleArgs,
)
import torch
import ttnn
from models.experimental.pointpillars.reference.model.pointpillars import PillarEncoder
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
        ),
    ):
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

    weight = conv.weight  # Shape: [out_channels, in_channels, kernel_size]
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias

    # For 1D: scale factor applied per output channel
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None]

    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    # For 1D convolutions, bias shape should be [1, 1, -1] instead of [1, 1, 1, -1]
    bias = bias.reshape(1, 1, 1, -1)

    return weight, bias


def _extract_pillar_encoder(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess PillarEncoder parameters with fused BatchNorm."""
    assert isinstance(model, PillarEncoder)
    parameters["pillar_encoder"]["conv"] = {}

    # Use the helper function
    weight, bias = fold_batch_norm1d_into_conv1d(model.conv, model.bn)
    parameters["pillar_encoder"]["conv"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
    parameters["pillar_encoder"]["conv"]["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
    parameters["pillar_encoder"]["conv_args"] = infer_module_args(model)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for PointPillars models."""
    parameters = {}
    weight_dtype = ttnn.bfloat16
    if isinstance(model, PillarEncoder):
        parameters["pillar_encoder"] = {}
        parameters = _extract_pillar_encoder(model, parameters, dtype=weight_dtype, mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Create a custom preprocessor with mesh mapping support."""

    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
