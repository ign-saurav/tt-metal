# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
from models.experimental.centernet.reference.dlav0 import (
    BasicBlock,
    Root,
    Tree,
    DLA,
    DLAUp,
    Identity,
)


def _extract_basic_block(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess BasicBlock parameters with fused BatchNorm."""
    assert isinstance(model, BasicBlock)
    parameters["conv1"] = {}
    parameters["conv2"] = {}
    weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
    bias = bias.reshape((1, 1, 1, -1))
    parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)
    weight, bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
    bias = bias.reshape((1, 1, 1, -1))
    parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def _extract_root(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess Root parameters with fused BatchNorm."""
    assert isinstance(model, Root)
    parameters["conv"] = {}
    weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
    bias = bias.reshape((1, 1, 1, -1))
    parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def _extract_tree(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess Tree parameters with recursive structure."""
    assert isinstance(model, Tree)

    if hasattr(model, "tree1"):
        if isinstance(model.tree1, BasicBlock):
            parameters["tree1"] = {}
            _extract_basic_block(model.tree1, parameters["tree1"], dtype=dtype, mesh_mapper=mesh_mapper)
        elif isinstance(model.tree1, Tree):
            parameters["tree1"] = {}
            _extract_tree(model.tree1, parameters["tree1"], dtype=dtype, mesh_mapper=mesh_mapper)

    if hasattr(model, "tree2"):
        if isinstance(model.tree2, BasicBlock):
            parameters["tree2"] = {}
            _extract_basic_block(model.tree2, parameters["tree2"], dtype=dtype, mesh_mapper=mesh_mapper)
        elif isinstance(model.tree2, Tree):
            parameters["tree2"] = {}
            _extract_tree(model.tree2, parameters["tree2"], dtype=dtype, mesh_mapper=mesh_mapper)

    if hasattr(model, "root") and model.root is not None:
        parameters["root"] = {}
        _extract_root(model.root, parameters["root"], dtype=dtype, mesh_mapper=mesh_mapper)

    if hasattr(model, "project") and model.project is not None:
        parameters["project"] = {}
        weight, bias = fold_batch_norm2d_into_conv2d(model.project[0], model.project[1])
        bias = bias.reshape((1, 1, 1, -1))
        parameters["project"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
        parameters["project"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def _extract_conv_level(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess conv level parameters (Sequential of Conv-BN-ReLU)."""
    assert isinstance(model, nn.Sequential)

    conv_idx = 0
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Conv2d):
            if i + 1 < len(model) and isinstance(model[i + 1], nn.BatchNorm2d):
                bn_layer = model[i + 1]
                weight, bias = fold_batch_norm2d_into_conv2d(layer, bn_layer)
                bias = bias.reshape((1, 1, 1, -1))

                parameters[f"conv{conv_idx}"] = {}
                parameters[f"conv{conv_idx}"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
                parameters[f"conv{conv_idx}"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)
                conv_idx += 1

    return parameters


def _extract_dla(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess DLA model parameters."""
    assert isinstance(model, DLA)

    parameters["base_layer"] = {}
    weight, bias = fold_batch_norm2d_into_conv2d(model.base_layer[0], model.base_layer[1])
    bias = bias.reshape((1, 1, 1, -1))
    parameters["base_layer"]["conv"] = {}
    parameters["base_layer"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["base_layer"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level0"] = {}
    _extract_conv_level(model.level0, parameters["level0"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level1"] = {}
    _extract_conv_level(model.level1, parameters["level1"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level2"] = {}
    _extract_tree(model.level2, parameters["level2"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level3"] = {}
    _extract_tree(model.level3, parameters["level3"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level4"] = {}
    _extract_tree(model.level4, parameters["level4"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["level5"] = {}
    _extract_tree(model.level5, parameters["level5"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["fc"] = {}
    fc_weight = model.fc.weight
    fc_bias = model.fc.bias.reshape((1, 1, 1, -1))
    parameters["fc"]["weight"] = ttnn.from_torch(fc_weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["fc"]["bias"] = ttnn.from_torch(fc_bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def _extract_dla_up(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess DLAUp parameters with multiple IDAUp modules."""
    assert isinstance(model, DLAUp)

    for i in range(len(model.channels) - 1):
        ida_name = f"ida_{i}"
        ida = getattr(model, ida_name)

        parameters[ida_name] = {}

        for j in range(len(ida.channels)):
            proj_name = f"proj_{j}"
            proj = getattr(ida, proj_name)
            if not isinstance(proj, Identity):
                weight, bias = fold_batch_norm2d_into_conv2d(proj[0], proj[1])
                parameters[ida_name][proj_name] = {}
                parameters[ida_name][proj_name]["weight"] = ttnn.from_torch(
                    weight, dtype=dtype, mesh_mapper=mesh_mapper
                )
                if bias is not None:
                    bias = bias.reshape((1, 1, 1, -1))
                    parameters[ida_name][proj_name]["bias"] = ttnn.from_torch(
                        bias, dtype=dtype, mesh_mapper=mesh_mapper
                    )
                else:
                    parameters[ida_name][proj_name]["bias"] = None

        for j in range(len(ida.channels)):
            up_name = f"up_{j}"
            up = getattr(ida, up_name)
            if not isinstance(up, Identity):
                parameters[ida_name][up_name] = {}
                parameters[ida_name][up_name]["weight"] = ttnn.from_torch(
                    up.weight, dtype=dtype, mesh_mapper=mesh_mapper
                )
                parameters[ida_name][up_name]["bias"] = None

        for j in range(1, len(ida.channels)):
            node_name = f"node_{j}"
            node = getattr(ida, node_name)
            weight, bias = fold_batch_norm2d_into_conv2d(node[0], node[1])
            parameters[ida_name][node_name] = {}
            parameters[ida_name][node_name]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
            if bias is not None:
                bias = bias.reshape((1, 1, 1, -1))
                parameters[ida_name][node_name]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)
            else:
                parameters[ida_name][node_name]["bias"] = None

    return parameters


def _extract_dla_seg(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess DLASeg model parameters."""
    from models.experimental.centernet.reference.dlav0 import DLASeg

    assert isinstance(model, DLASeg)

    parameters["base"] = {}
    _extract_dla(model.base, parameters["base"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["dla_up"] = {}
    _extract_dla_up(model.dla_up, parameters["dla_up"], dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["heads"] = {}
    for head_name in model.heads:
        head_module = getattr(model, head_name)
        head_params = {}

        if isinstance(head_module, nn.Sequential):
            conv1_weight = head_module[0].weight
            conv1_bias = head_module[0].bias
            head_params["conv1"] = {}
            head_params["conv1"]["weight"] = ttnn.from_torch(conv1_weight, dtype=dtype, mesh_mapper=mesh_mapper)
            head_params["conv1"]["bias"] = ttnn.from_torch(
                conv1_bias.reshape(1, 1, 1, -1), dtype=dtype, mesh_mapper=mesh_mapper
            )

            conv2_weight = head_module[2].weight
            conv2_bias = head_module[2].bias
            head_params["conv2"] = {}
            head_params["conv2"]["weight"] = ttnn.from_torch(conv2_weight, dtype=dtype, mesh_mapper=mesh_mapper)
            head_params["conv2"]["bias"] = ttnn.from_torch(
                conv2_bias.reshape(1, 1, 1, -1), dtype=dtype, mesh_mapper=mesh_mapper
            )
        else:
            conv_weight = head_module.weight
            conv_bias = head_module.bias
            head_params["weight"] = ttnn.from_torch(conv_weight, dtype=dtype, mesh_mapper=mesh_mapper)
            head_params["bias"] = ttnn.from_torch(conv_bias.reshape(1, 1, 1, -1), dtype=dtype, mesh_mapper=mesh_mapper)

        parameters["heads"][head_name] = head_params

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for Centernet, DLA, DLAUp, and DLASeg models."""
    parameters = {}
    weight_dtype = ttnn.bfloat16

    from models.experimental.centernet.reference.dlav0 import DLASeg

    if isinstance(model, DLASeg):
        parameters["dla_seg"] = {}
        parameters["dla_seg"] = _extract_dla_seg(
            model, parameters["dla_seg"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )
        return parameters
    elif isinstance(model, DLAUp):
        parameters["dla_up"] = {}
        parameters["dla_up"] = _extract_dla_up(model, parameters["dla_up"], dtype=weight_dtype, mesh_mapper=mesh_mapper)
        return parameters
    elif isinstance(model, DLA):
        parameters["dla"] = {}
        parameters["dla"] = _extract_dla(model, parameters["dla"], dtype=weight_dtype, mesh_mapper=mesh_mapper)
        return parameters
    elif isinstance(model, BasicBlock):
        parameters["basic_block"] = {}
        parameters["basic_block"] = _extract_basic_block(
            model, parameters["basic_block"], dtype=weight_dtype, mesh_mapper=mesh_mapper
        )
        return parameters
    elif isinstance(model, Root):
        parameters["root"] = {}
        parameters["root"] = _extract_root(model, parameters["root"], dtype=weight_dtype, mesh_mapper=mesh_mapper)
        return parameters
    elif isinstance(model, Tree):
        parameters["tree"] = {}
        parameters["tree"] = _extract_tree(model, parameters["tree"], dtype=weight_dtype, mesh_mapper=mesh_mapper)
        return parameters
    else:
        return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Create a custom preprocessor with mesh mapping support."""

    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
