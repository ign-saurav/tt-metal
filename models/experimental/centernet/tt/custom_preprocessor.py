# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
import ttnn
from models.experimental.centernet.reference.network.dlav0 import (
    BasicBlock,
    Root,
    Tree,
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

    # Extract tree1 (either BasicBlock or Tree)
    if hasattr(model, "tree1"):
        if isinstance(model.tree1, BasicBlock):
            parameters["tree1"] = {}
            _extract_basic_block(model.tree1, parameters["tree1"], dtype=dtype, mesh_mapper=mesh_mapper)
        elif isinstance(model.tree1, Tree):
            parameters["tree1"] = {}
            _extract_tree(model.tree1, parameters["tree1"], dtype=dtype, mesh_mapper=mesh_mapper)

    # Extract tree2 (either BasicBlock or Tree)
    if hasattr(model, "tree2"):
        if isinstance(model.tree2, BasicBlock):
            parameters["tree2"] = {}
            _extract_basic_block(model.tree2, parameters["tree2"], dtype=dtype, mesh_mapper=mesh_mapper)
        elif isinstance(model.tree2, Tree):
            parameters["tree2"] = {}
            _extract_tree(model.tree2, parameters["tree2"], dtype=dtype, mesh_mapper=mesh_mapper)

    # Extract root if present (only at leaf levels)
    if hasattr(model, "root") and model.root is not None:
        parameters["root"] = {}
        _extract_root(model.root, parameters["root"], dtype=dtype, mesh_mapper=mesh_mapper)

    # Extract project if present (1x1 conv + BatchNorm)
    if hasattr(model, "project") and model.project is not None:
        parameters["project"] = {}
        weight, bias = fold_batch_norm2d_into_conv2d(model.project[0], model.project[1])
        bias = bias.reshape((1, 1, 1, -1))
        parameters["project"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
        parameters["project"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for Centernet models."""
    parameters = {}
    weight_dtype = ttnn.bfloat16

    if isinstance(model, BasicBlock):
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
