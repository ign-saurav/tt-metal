# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

from models.experimental.BevDepth.tests.ref_bev_depth_head import BEVDepthHead, ConvModule
from models.experimental.BevDepth.tests.ref_bev_depth_neck import SECONDFPN

# from models.experimental.BevDepth.reference.utils import Conv2D


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
    elif isinstance(model, torch.nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            parameters["bias"] = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    elif isinstance(model, torch.nn.Sequential):
        # Handle Sequential with ConvTranspose2d + BatchNorm2d + ReLU
        if len(model) >= 2:
            if isinstance(model[0], torch.nn.ConvTranspose2d) and isinstance(model[1], torch.nn.BatchNorm2d):
                weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv_transpose2d(
                    model[0], model[1], mesh_mapper=mesh_mapper
                )
                parameters["weight"] = weight_ttnn
                parameters["bias"] = bias_ttnn
            else:
                # Let sub-modules handle their own preprocessing
                for child_name, child in model.named_children():
                    parameters[child_name] = convert_torch_model_to_ttnn_model(
                        child,
                        name=f"{name}.{child_name}",
                        custom_preprocessor=custom_preprocessor_func,
                        convert_to_ttnn=convert_to_ttnn,
                        ttnn_module_args=ttnn_module_args,
                    )
        else:
            # Let sub-modules handle their own preprocessing
            for child_name, child in model.named_children():
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=f"{name}.{child_name}",
                    custom_preprocessor=custom_preprocessor_func,
                    convert_to_ttnn=convert_to_ttnn,
                    ttnn_module_args=ttnn_module_args,
                )
    elif isinstance(model, torch.nn.ConvTranspose2d):
        parameters["weight"] = ttnn.from_torch(model.weight, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            parameters["bias"] = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    elif isinstance(model, torch.nn.ModuleList):
        # Handle ModuleList (e.g., deblocks in SECONDFPN)
        for idx, child in enumerate(model):
            parameters[idx] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{idx}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    elif isinstance(model, SECONDFPN):
        # Handle SECONDFPN by processing its deblocks ModuleList
        # The ModuleList will be handled recursively, but we can also handle it here explicitly
        for child_name, child in model.named_children():
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    elif isinstance(
        model,
        (BEVDepthHead,),
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

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
