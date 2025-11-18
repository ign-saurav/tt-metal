# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
)

# from efficientnet_pytorch import EfficientNet
from models.experimental.pointpillars.reference.model.pointpillars import PillarEncoder

# from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
# from models.experimental.efficientdetd0.reference.modules import BiFPN, Regressor, Classifier, SeparableConvBlock


def _preprocess_conv_bn_parameter(conv, bn, *, dtype=ttnn.bfloat16, mesh_mapper=None):
    parameters = {}
    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv, bn)
    parameters["weight"] = ttnn.from_torch(conv_weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["bias"] = ttnn.from_torch(conv_bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)
    return parameters


def _preprocess_conv_params(conv, *, dtype=ttnn.bfloat16, mesh_mapper=None):
    parameters = {}
    weight = conv.weight
    bias = conv.bias
    parameters["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["bias"] = ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)
    return parameters

    # def _extract_pillar_encoder(model, bn=None, dtype=ttnn.bfloat16, mesh_mapper=None):
    #     assert isinstance(model, PillarEncoder)
    #     parameters = {}

    #     return parameters

    # def custom_preprocessor(
    #     model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
    # ):
    #     parameters = {}
    #     weight_dtype = ttnn.bfloat16

    #     if isinstance(model, PillarEncoder):
    #         parameters = _extract_pillar_encoder(model, dtype=weight_dtype, mesh_mapper=mesh_mapper)
    # elif isinstance(
    #     model,
    #     (
    #         Regressor,
    #         Classifier,
    #     ),
    # ):
    #     parameters["conv_list"] = {}
    #     parameters["header_list"] = {}
    #     # Creating batchnorm folded conv weights; multiple copies of conv, one for each pyramid layer
    #     for layer_num, pyramid_layer_bn_list in enumerate(model.bn_list):
    #         parameters["conv_list"][layer_num] = {}
    #         for id, bn in enumerate(pyramid_layer_bn_list):
    #             parameters["conv_list"][layer_num][id] = _extract_seperable_conv(
    #                 model.conv_list[id], bn, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #             )
    #         parameters["header_list"][layer_num] = _extract_seperable_conv(
    #             model.header, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #         )
    # elif isinstance(model, BiFPN):
    #     # Let the sub-modules handle their own preprocessing
    #     for child_name, child in model.named_children():
    #         if isinstance(child, SeparableConvBlock):
    #             parameters[child_name] = _extract_seperable_conv(child, dtype=weight_dtype, mesh_mapper=mesh_mapper)
    #         elif isinstance(child, nn.Sequential) and len(child) > 1:
    #             if isinstance(child[0], nn.Conv2d) and isinstance(child[1], nn.BatchNorm2d):
    #                 parameters[child_name] = {}
    #                 parameters[child_name][0] = _preprocess_conv_bn_parameter(
    #                     child[0], child[1], dtype=weight_dtype, mesh_mapper=mesh_mapper
    #                 )
    #             else:
    #                 continue  # Maxpool case
    #     # Store attention weights if using fast attention
    #     if model.attention:
    #         parameters["p6_w1"] = ttnn.from_torch(
    #             model.p6_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p5_w1"] = ttnn.from_torch(
    #             model.p5_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p4_w1"] = ttnn.from_torch(
    #             model.p4_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p3_w1"] = ttnn.from_torch(
    #             model.p3_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p4_w2"] = ttnn.from_torch(
    #             model.p4_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p5_w2"] = ttnn.from_torch(
    #             model.p5_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p6_w2"] = ttnn.from_torch(
    #             model.p6_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    #         parameters["p7_w2"] = ttnn.from_torch(
    #             model.p7_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    #         )
    # elif isinstance(model, EfficientNet):
    #     parameters = {}
    #     parameters["_conv_stem"] = _preprocess_conv_bn_parameter(
    #         model._conv_stem, model._bn0, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #     )
    #     parameters["_blocks"] = {}
    #     for idx, block in enumerate(model._blocks):
    #         block_parameters = {}
    #         if hasattr(block, "_expand_conv"):
    #             block_parameters["_expand_conv"] = _preprocess_conv_bn_parameter(
    #                 block._expand_conv, block._bn0, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #             )
    #         block_parameters["_depthwise_conv"] = _preprocess_conv_bn_parameter(
    #             block._depthwise_conv, block._bn1, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #         )
    #         block_parameters["_se_reduce"] = _preprocess_conv_params(
    #             block._se_reduce, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #         )
    #         block_parameters["_se_expand"] = _preprocess_conv_params(
    #             block._se_expand, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #         )
    #         block_parameters["_project_conv"] = _preprocess_conv_bn_parameter(
    #             block._project_conv, block._bn2, dtype=weight_dtype, mesh_mapper=mesh_mapper
    #         )
    #         parameters["_blocks"][idx] = block_parameters
    # elif isinstance(model, EfficientDetBackbone):
    #     # Let the sub-modules handle their own preprocessing
    #     for child_name, child in model.named_children():
    #         parameters[child_name] = convert_torch_model_to_ttnn_model(
    #             child,
    #             name=f"{name}.{child_name}",
    #             custom_preprocessor=custom_preprocessor_func,
    #             convert_to_ttnn=convert_to_ttnn,
    #             ttnn_module_args=ttnn_module_args,
    #         )

    return parameters


# def create_custom_mesh_preprocessor(mesh_mapper=None):
#     def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
#         return custom_preprocessor(
#             model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
#         )

#     return custom_mesh_preprocessor


# class ModuleArgs(dict):
#     __getattr__ = dict.__getitem__
#     __delattr__ = dict.__delitem__

#     def __repr__(self):
#         return super().__repr__()


# def register_layer_hooks(model, layer_type):
#     """Register hooks on all instances of a given layer type."""
#     layer_info = {}

#     def make_hook(name):
#         def hook_fn(module, input, output):
#             # input and output are tuples
#             input_shape = tuple(input[0].shape) if isinstance(input, (tuple, list)) else tuple(input.shape)
#             output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else tuple(output[0].shape)

#             if name not in layer_info:
#                 layer_info[name] = {}

#             layer_info[name][len(layer_info[name])] = ModuleArgs(
#                 kernel_size=getattr(module, "kernel_size", None),
#                 stride=getattr(module, "stride", None),
#                 padding=getattr(module, "padding", None),
#                 padding_mode=getattr(module, "padding_mode", None),
#                 dilation=getattr(module, "dilation", None),
#                 groups=getattr(module, "groups", None),
#                 in_channels=getattr(module, "in_channels", None),
#                 out_channels=getattr(module, "out_channels", None),
#                 batch_size=input_shape[0],
#                 input_height=input_shape[-2],
#                 input_width=input_shape[-1],
#                 output_shape=output_shape,
#             )

#         return hook_fn

#     hooks = []
#     for name, module in model.named_modules():
#         if isinstance(module, layer_type):
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     return layer_info, hooks


# def _expand_dotted_keys(flat_dict):
#     result = {}

#     for key, value in flat_dict.items():
#         parts = key.split(".")
#         current = result
#         for i, part in enumerate(parts):
#             # convert numeric keys to int
#             if part.isdigit():
#                 part = int(part)

#             if i == len(parts) - 1:
#                 # last part — assign the value
#                 current[part] = value
#             else:
#                 if part not in current or not isinstance(current[part], dict):
#                     current[part] = {}
#                 current = current[part]
#     return result


# def _fix_layername(layer_info):
#     structured_info = {}
#     for layer_name, instances in layer_info.items():
#         if len(instances) > 1:
#             # Cases where same layer is called multiple times in model's forward call
#             op_name = layer_name[layer_name.rfind(".") + 1 :]
#             layer_tree = layer_name[: layer_name.rfind(".")]
#             for idx, instance in instances.items():
#                 if "conv_list" in layer_tree:
#                     # Fix for nested loop in forward call, we need the instance index to be right after "conv_list" in params
#                     updated_layer_tree = (
#                         layer_tree[: layer_tree.rfind(".") + 1] + str(idx) + layer_tree[layer_tree.rfind(".") :]
#                     )
#                     updated_layer_name = updated_layer_tree + "." + op_name
#                 else:
#                     updated_layer_name = layer_tree + f".{idx}." + op_name
#                 structured_info[updated_layer_name] = instance
#         else:
#             structured_info[layer_name] = instances[0]
#     return structured_info


# def _make_dot_accessible_args(layer_info):
#     structured_info = _fix_layername(layer_info)
#     structured_args = _expand_dotted_keys(structured_info)
#     return make_dot_access_dict(structured_args, ignore_types=(ModuleArgs,))


# def infer_torch_module_args(model, input, layer_type=(nn.Conv2d, nn.MaxPool2d)):
#     """Run forward pass and collect layer information."""
#     model.eval()

#     layer_info, hooks = register_layer_hooks(model, layer_type)

#     with torch.no_grad():
#         _ = model(input)

#     # Remove hooks to avoid memory leaks
#     for h in hooks:
#         h.remove()

#     return _make_dot_accessible_args(layer_info)

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.pointpillars.reference.model.pointpillars import PillarEncoder


def _preprocess_conv1d_bn_parameter(conv, bn, *, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Fold BatchNorm1d into Conv1d weights and convert to TTNN tensors."""
    parameters = {}

    # Extract weights
    conv_weight = conv.weight.data.squeeze(-1).transpose(-2, -1)  # (in_channel, out_channel)

    # Fold batch norm
    bn_weight = bn.weight.data
    bn_bias = bn.bias.data
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    eps = bn.eps

    scale = bn_weight / torch.sqrt(bn_var + eps)
    shift = bn_bias - bn_mean * scale

    # Convert to TTNN
    parameters["weight"] = ttnn.from_torch(conv_weight, dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT)
    parameters["bn_scale"] = ttnn.from_torch(
        scale.reshape(1, 1, 1, -1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    )
    parameters["bn_shift"] = ttnn.from_torch(
        shift.reshape(1, 1, 1, -1), dtype=dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    )

    return parameters


def _extract_pillar_encoder(model, dtype=ttnn.bfloat16, mesh_mapper=None):
    """Extract and preprocess PillarEncoder parameters with fused BatchNorm."""
    assert isinstance(model, PillarEncoder)
    parameters = {}

    # Use the helper function
    parameters["conv"] = _preprocess_conv1d_bn_parameter(model.conv, model.bn, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """Custom preprocessor for PointPillars models."""
    parameters = {}
    weight_dtype = ttnn.bfloat16

    if isinstance(model, PillarEncoder):
        parameters = _extract_pillar_encoder(model, dtype=weight_dtype, mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    """Create a custom preprocessor with mesh mapping support."""

    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
