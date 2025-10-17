# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d


# ------------------------------------------------------
# Helper to fold Conv + BatchNorm and convert to TTNN
# ------------------------------------------------------
def conv_bn_to_params(conv, bn, mesh_mapper):
    """Fold BN into Conv and return TTNN weight/bias tensors."""
    if bn is None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


# ------------------------------------------------------
# Recursive TTNN preprocessor for RetinaNet
# ------------------------------------------------------
def custom_preprocessor(
    model,
    name=None,
    ttnn_module_args=None,
    convert_to_ttnn=True,
    custom_preprocessor_func=None,
    mesh_mapper=None,
):
    """
    Recursively preprocess RetinaNet layers into TTNN-compatible tensors,
    folding Conv+BN automatically. `name` is optional.
    """
    parameters = {}

    # Conv2d layer: fold BN if attached via _bn
    if isinstance(model, torch.nn.Conv2d):
        bn = getattr(model, "_bn", None)  # optional: store BN in conv._bn
        parameters = conv_bn_to_params(model, bn, mesh_mapper)
        return parameters

    # Module / Sequential: recursively process children
    elif isinstance(model, torch.nn.Module):
        for child_name, child in model.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=child_full_name,
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
        return parameters

    return parameters


# ------------------------------------------------------
# Factory to create mesh-aware preprocessor
# ------------------------------------------------------
def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Returns a mesh-aware TTNN preprocessor function."""

    def custom_mesh_preprocessor(model, name=None, ttnn_module_args=None, convert_to_ttnn=True):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
