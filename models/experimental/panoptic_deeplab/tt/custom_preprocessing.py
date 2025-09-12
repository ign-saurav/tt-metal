# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
import ttnn


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    for name, module in model.named_children():
        if hasattr(module, "__getitem__"):
            # If it's a Sequential or similar
            if len(module) > 1 and hasattr(module[0], "weight") and hasattr(module[1], "weight"):
                # Assume Conv + BN, fold BN into Conv
                weight, bias = fold_batch_norm2d_into_conv2d(module[0], module[1])
            elif hasattr(module[0], "weight"):
                # Just a Conv, no BN
                weight = module[0].weight.clone().detach().contiguous()
                bias = (
                    module[0].bias.clone().detach().contiguous()
                    if module[0].bias is not None
                    else torch.zeros(module[0].out_channels)
                )
            else:
                continue
        elif hasattr(module, "weight"):
            # Single Conv2d
            weight = module.weight.clone().detach().contiguous()
            bias = (
                module.bias.clone().detach().contiguous()
                if module.bias is not None
                else torch.zeros(module.out_channels)
            )
        else:
            continue

        parameters[name] = {}
        parameters[name]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters[name]["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
