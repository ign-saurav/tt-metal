# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.model_preprocessing import make_parameter_dict
from ttnn.model_preprocessing import ModuleArgs, fold_batch_norm2d_into_conv2d
import torch
import ttnn
from models.experimental.centernet.reference.network.dlav0 import DLAUp, Identity
from models.experimental.centernet.reference.network.dlav0 import (
    BasicBlock,
    Root,
)
from ttnn.dot_access import make_dot_access_dict


def fold_bn_into_conv(conv, bn):
    """
    Fold BatchNorm parameters into Conv2d weights.

    Args:
        conv: nn.Conv2d layer
        bn: nn.BatchNorm2d layer

    Returns:
        folded_weight: Tensor (out_channels, in_channels, kH, kW)
        folded_bias: Tensor (out_channels,)
    """
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = gamma / std

    folded_weight = conv.weight.data * scale.view(-1, 1, 1, 1)
    folded_bias = beta - mean * scale

    return folded_weight, folded_bias


def create_basic_block_preprocessor():
    """
    Creates a custom preprocessor for BasicBlock that folds BatchNorm into Conv weights.
    """

    def preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if hasattr(model, "conv1") and hasattr(model, "bn1"):
            weight, bias = fold_bn_into_conv(model.conv1, model.bn1)
            parameters["conv1"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            }

        if hasattr(model, "conv2") and hasattr(model, "bn2"):
            weight, bias = fold_bn_into_conv(model.conv2, model.bn2)
            parameters["conv2"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            }

        return parameters

    return preprocessor


def create_centernet_head_preprocessor():
    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}

        # Check if it's a Sequential with the expected structure
        if isinstance(model, torch.nn.Sequential) and len(model) == 3:
            # Handle conv layers at indices 0 and 2
            for i, layer_name in enumerate(["conv1", "conv2"]):
                layer_idx = 0 if i == 0 else 2
                conv_layer = model[layer_idx]

                parameters[layer_name] = {}
                parameters[layer_name]["weight"] = ttnn.from_torch(conv_layer.weight, dtype=ttnn.float32)
                if conv_layer.bias is not None:
                    bias = conv_layer.bias.reshape((1, 1, 1, -1))
                    parameters[layer_name]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
                else:
                    parameters[layer_name]["bias"] = None

        return parameters

    return custom_preprocessor


def create_head_preprocessor():
    """
    Creates a custom preprocessor for CenterNet heads.
    """

    def preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if hasattr(model, "__class__") and "Sequential" in str(type(model)):
            # Head with intermediate conv (head_conv > 0)
            for i, layer in enumerate(model):
                if hasattr(layer, "weight") and hasattr(layer, "bias"):
                    weight = layer.weight.data
                    bias = layer.bias.data
                    parameters[i] = {
                        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                        "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                    }
        else:
            # Single conv head (head_conv = 0)
            if hasattr(model, "weight") and hasattr(model, "bias"):
                weight = model.weight.data
                bias = model.bias.data
                parameters["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
                parameters["bias"] = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        return parameters

    return preprocessor


def create_root_preprocessor():
    def custom_root_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, Root):
            parameters["conv"] = {}
            parameters["conv"]["weight"] = ttnn.from_torch(model.conv.weight, dtype=ttnn.bfloat16)
            if model.conv.bias is not None:
                parameters["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(model.conv.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )
        return parameters

    return custom_root_preprocessor


def preprocess_tree_parameters(model):
    """
    Preprocess tree parameters by traversing the tree structure and folding BatchNorm.
    """
    parameters = {}

    def preprocess_subtree(subtree):
        """Recursively preprocess a subtree (Tree or BasicBlock)"""
        if hasattr(subtree, "conv1"):  # It's a BasicBlock
            # Use basic block preprocessing
            basic_preprocessor = create_basic_block_preprocessor()
            return basic_preprocessor(subtree, "", None)
        else:  # It's a Tree
            sub_params = {}
            if hasattr(subtree, "tree1"):
                sub_params["tree1"] = preprocess_subtree(subtree.tree1)
            if hasattr(subtree, "tree2"):
                sub_params["tree2"] = preprocess_subtree(subtree.tree2)
            if hasattr(subtree, "root"):
                # Preprocess root conv + bn
                if hasattr(subtree.root, "conv") and hasattr(subtree.root, "bn"):
                    weight, bias = fold_bn_into_conv(subtree.root.conv, subtree.root.bn)
                    sub_params["root"] = {
                        "conv": {
                            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                            "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                        }
                    }
                else:
                    sub_params["root"] = {}
            return sub_params

    if hasattr(model, "tree1"):
        parameters["tree1"] = preprocess_subtree(model.tree1)
    if hasattr(model, "tree2"):
        parameters["tree2"] = preprocess_subtree(model.tree2)

    # Handle root, downsample, project at the top level
    if hasattr(model, "root"):
        if hasattr(model.root, "conv") and hasattr(model.root, "bn"):
            weight, bias = fold_bn_into_conv(model.root.conv, model.root.bn)
            parameters["root"] = {
                "conv": {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                    "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                }
            }
        else:
            parameters["root"] = {}
    if hasattr(model, "downsample") and model.downsample is not None:
        parameters["downsample"] = {}
    if hasattr(model, "project") and model.project is not None:
        if hasattr(model.project, "conv") and hasattr(model.project, "bn"):
            weight, bias = fold_bn_into_conv(model.project[0], model.project[1])
            parameters["project"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            }

    return make_parameter_dict(parameters)


def create_tree_preprocessor():
    """
    Creates a custom preprocessor for Tree that uses manual parameter preprocessing.
    """

    def preprocessor(model, name, ttnn_module_args):
        return preprocess_tree_parameters(model)

    return preprocessor


class ConvArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_module_args(model):
    if isinstance(
        model,
        (
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


def _extract_basic_block(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess BasicBlock parameters with fused BatchNorm."""
    assert isinstance(model, BasicBlock)
    parameters["conv"] = {}

    weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
    parameters["conv"]["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
    parameters["conv"]["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
    parameters["conv_args"] = infer_module_args(model)

    return parameters


# def _extract_neck(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
#     """Extract and preprocess Neck parameters with fused BatchNorm."""
#     assert isinstance(model, Neck)

#     for i in range(len(model.decoder_blocks)):
#         block = model.decoder_blocks[i]
#         parameters[f"decoder_{i}"] = {}

#         conv_transpose_layer = block[0]
#         bn_layer = block[1]

#         if i == 0:
#             weight, bias = fold_batch_norm2d_into_conv_transpose2d(conv_transpose_layer, bn_layer)
#             parameters[f"decoder_{i}"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
#             bias = bias.reshape((1, 1, 1, -1))
#             parameters[f"decoder_{i}"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)
#             parameters[f"decoder_{i}"]["conv_args"] = infer_module_args(conv_transpose_layer)
#         else:
#             weight = conv_transpose_layer.weight
#             parameters[f"decoder_{i}"]["weight"] = weight
#             parameters[f"decoder_{i}"]["conv_args"] = infer_module_args(conv_transpose_layer)
#             bias = conv_transpose_layer.bias

#             if bias is not None:
#                 bias = bias.reshape((1, 1, 1, -1))
#                 parameters[f"decoder_{i}"]["bias"] = bias, dtype = dtype, mesh_mapper = mesh_mapper
#             else:
#                 out_channels = conv_transpose_layer.out_channels
#                 zero_bias = torch.zeros(1, 1, 1, out_channels)
#                 parameters[f"decoder_{i}"]["bias"] = zero_bias

#             parameters[f"decoder_{i}"]["bn_weight"] = ttnn.from_torch(
#                 bn_layer.weight.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
#             )
#             parameters[f"decoder_{i}"]["bn_bias"] = ttnn.from_torch(
#                 bn_layer.bias.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
#             )
#             parameters[f"decoder_{i}"]["bn_running_mean"] = ttnn.from_torch(
#                 bn_layer.running_mean.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
#             )
#             parameters[f"decoder_{i}"]["bn_running_var"] = ttnn.from_torch(
#                 bn_layer.running_var.view(1, -1, 1, 1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
#             )

#     return parameters


# def _extract_head(model, parameters, dtype=ttnn.bfloat16, mesh_mapper=None):
#     """Extract and preprocess Head parameters."""
#     assert isinstance(model, Head)

#     # Process conv_cls
#     parameters["conv_cls"] = {}
#     parameters["conv_cls"]["weight"] = ttnn.from_torch(model.conv_cls.weight, dtype=dtype)
#     bias = model.conv_cls.bias.reshape((1, 1, 1, -1))
#     parameters["conv_cls"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
#     parameters["conv_cls"]["conv_args"] = infer_module_args(model.conv_cls)

#     # Process conv_reg
#     parameters["conv_reg"] = {}
#     parameters["conv_reg"]["weight"] = ttnn.from_torch(model.conv_reg.weight, dtype=dtype)
#     bias = model.conv_reg.bias.reshape((1, 1, 1, -1))
#     parameters["conv_reg"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
#     parameters["conv_reg"]["conv_args"] = infer_module_args(model.conv_reg)

#     # Process conv_dir_cls
#     parameters["conv_dir_cls"] = {}
#     parameters["conv_dir_cls"]["weight"] = ttnn.from_torch(model.conv_dir_cls.weight, dtype=dtype)
#     bias = model.conv_dir_cls.bias.reshape((1, 1, 1, -1))
#     parameters["conv_dir_cls"]["bias"] = ttnn.from_torch(bias, dtype=dtype)
#     parameters["conv_dir_cls"]["conv_args"] = infer_module_args(model.conv_dir_cls)

#     return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for Centernet models."""
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


def create_dla_up_preprocessor():
    """Creates a custom preprocessor for DLAUp that handles multiple IDAUp modules."""

    def preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, DLAUp):
            # Process each IDAUp module within DLAUp
            for i in range(len(model.channels) - 1):
                ida_name = f"ida_{i}"
                ida = getattr(model, ida_name)

                # Create nested parameters for each IDAUp
                parameters[ida_name] = {}

                # Process projection layers in this IDAUp
                for j in range(len(ida.channels)):
                    proj_name = f"proj_{j}"
                    proj = getattr(ida, proj_name)
                    if not isinstance(proj, Identity):
                        weight, bias = fold_batch_norm2d_into_conv2d(proj[0], proj[1])
                        parameters[ida_name][proj_name] = {}
                        parameters[ida_name][proj_name]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
                        if bias is not None:
                            bias = bias.reshape((1, 1, 1, -1))
                            parameters[ida_name][proj_name]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
                        else:
                            parameters[ida_name][proj_name]["bias"] = None

                # Process upsampling layers in this IDAUp
                for j in range(len(ida.channels)):
                    up_name = f"up_{j}"
                    up = getattr(ida, up_name)
                    if not isinstance(up, Identity):
                        parameters[ida_name][up_name] = {}
                        parameters[ida_name][up_name]["weight"] = ttnn.from_torch(up.weight, dtype=ttnn.bfloat16)
                        parameters[ida_name][up_name]["bias"] = None

                # Process node layers in this IDAUp
                for j in range(1, len(ida.channels)):
                    node_name = f"node_{j}"
                    node = getattr(ida, node_name)
                    weight, bias = fold_batch_norm2d_into_conv2d(node[0], node[1])
                    parameters[ida_name][node_name] = {}
                    parameters[ida_name][node_name]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
                    if bias is not None:
                        bias = bias.reshape((1, 1, 1, -1))
                        parameters[ida_name][node_name]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
                    else:
                        parameters[ida_name][node_name]["bias"] = None

        return parameters

    return preprocessor
