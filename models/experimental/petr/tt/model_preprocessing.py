# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import numpy as np
from ttnn.dot_access import make_dot_access_dict
from ttnn.torch_tracer import trace
from ttnn.model_preprocessing import infer_ttnn_module_args
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    fold_batch_norm2d_into_conv2d,
    preprocess_model_parameters,
)
from torch.nn import Conv2d, Linear
from torch import nn

from models.experimental.petr.reference.petr_head import PETRHead, pos2posemb3d
from models.experimental.petr.reference.cp_fpn import CPFPN
from models.experimental.petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module, _OSA_stage
from models.experimental.petr.reference.petr_transformer import PETRTransformer


def move_to_device(object, device):
    if isinstance(object, dict):
        for key in object.keys():
            object[key] = move_to_device(object[key], device)
    elif isinstance(object, list):
        for i in range(len(object)):
            object[i] = move_to_device(object[i], device)
    elif isinstance(object, ttnn.Tensor):
        object = ttnn.to_device(object, device)
    return object


def stem_parameters_preprocess(model):
    parameters = {}
    if isinstance(model, VoVNetCP):
        if hasattr(model, "stem"):
            layers = list(model.stem.named_children())

        for i, (name, layer) in enumerate(layers):
            if "conv" in name:
                conv_name, conv_layer = layers[i]
                norm_name, norm_layer = layers[i + 1]
                prefix = conv_name.split("/")[0]
                if prefix not in parameters:
                    parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

    return parameters


def create_custom_preprocessor_cpfpn(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CPFPN):
            parameters["lateral_convs"] = {}
            for i, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][i] = {}
                parameters["lateral_convs"][i]["conv"] = {}
                parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
                    child.conv.weight, dtype=ttnn.bfloat16
                )
                parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
            parameters["fpn_convs"] = {}
            for i, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][i] = {}
                parameters["fpn_convs"][i]["conv"] = {}
                parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
                parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
        return parameters

    return custom_preprocessor


def create_custom_preprocessor_vovnetcp(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, eSEModule):
            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
        if isinstance(model, _OSA_module):
            if hasattr(model, "conv_reduction"):
                first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv_reduction[0], model.conv_reduction[1])
                parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i, layers in enumerate(model.layers):
                first_layer_name = list(layers.named_children())[0][0]
                prefix = first_layer_name.split("/")[0]
                parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            concat_weight, concat_bias = fold_batch_norm2d_into_conv2d(model.concat[0], model.concat[1])
            parameters[base_name]["weight"] = ttnn.from_torch(concat_weight, dtype=ttnn.bfloat16)
            parameters[base_name]["bias"] = ttnn.from_torch(
                torch.reshape(concat_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(
                torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )
        if isinstance(model, _OSA_stage):
            if isinstance(model, _OSA_module):
                if hasattr(model, "conv_reduction"):
                    first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                    base_name = first_layer_name.split("/")[0]
                    parameters[base_name] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        model.conv_reduction[0], model.conv_reduction[1]
                    )
                    parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[base_name]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                for i, layers in enumerate(model.layers):
                    first_layer_name = list(layers.named_children())[0][0]
                    prefix = first_layer_name.split("/")[0]
                    parameters[prefix] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                first_layer_name, _ = list(model.concat.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias

                parameters["fc"] = {}
                parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
                parameters["fc"]["bias"] = ttnn.from_torch(
                    torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

        return parameters

    return custom_preprocessor


def create_custom_preprocessor_petr_head(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PETRHead):
            parameters["input_proj"] = {}
            parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.bfloat16)
            parameters["input_proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters["cls_branches"] = {}
            for index, child in enumerate(model.cls_branches):
                parameters["cls_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["cls_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["adapt_pos3d"] = {}
            for index, child in enumerate(model.adapt_pos3d):
                parameters["adapt_pos3d"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["position_encoder"] = {}
            for index, child in enumerate(model.position_encoder):
                parameters["position_encoder"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["query_embedding"] = {}
            for index, child in enumerate(model.query_embedding):
                parameters["query_embedding"][index] = {}
                if isinstance(child, Linear):
                    parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                        child.weight, dtype=ttnn.bfloat16
                    )
                    parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                        child.bias, dtype=ttnn.bfloat16
                    )
            parameters["reference_points"] = {}
            parameters["reference_points"]["weight"] = ttnn.from_torch(
                model.reference_points.weight, dtype=ttnn.bfloat16, device=device
            )

        return parameters

    return custom_preprocessor


def get_parameters(torch_model, device):
    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    # transformer module preprocess
    child = torch_model.pts_bbox_head.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, 20, 50),
            torch.zeros((1, 6, 20, 50), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, 20, 50),
        ),
        device=None,
    )
    assert x is not None
    for key in x.keys():
        x[key].module = getattr(child, key)
    parameters_petr_head["transformer"] = x

    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    parameters = {}
    parameters["pts_bbox_head"] = parameters_petr_head
    parameters["img_neck"] = parameters_petr_cpfpn
    parameters["img_backbone"] = parameters_petr_vovnetcp

    stem_parameters = stem_parameters_preprocess(torch_model.img_backbone)
    parameters["stem_parameters"] = stem_parameters

    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)

    query_embedding_input = ttnn.from_torch(
        query_embedding_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    return parameters, query_embedding_input


def generate_petr_inputs():
    torch.manual_seed(42)
    np.random.seed(42)

    B, num_cams, C, H, W = 1, 6, 3, 320, 800

    # input data
    inputs = dict()
    inputs["imgs"] = torch.randn(B, num_cams, C, H, W, dtype=torch.float32)

    # metadata for the input
    scale_x = 800 / 1600
    scale_y = 320 / 900

    # intrinsics for the input used from the nuscenes mini dataset
    intrinsics_original = [
        [
            [1266.417203046554, 0.0, 816.2670197447984],
            [0.0, 1266.417203046554, 491.50706579294757],
            [0.0, 0.0, 1.0],
        ],  # FRONT
        [
            [1260.8474446004698, 0.0, 807.968244525554],
            [0.0, 1260.8474446004698, 495.3344268742088],
            [0.0, 0.0, 1.0],
        ],  # FRONT_RIGHT
        [
            [1272.5979470598488, 0.0, 826.6154927353808],
            [0.0, 1272.5979470598488, 479.75165386361925],
            [0.0, 0.0, 1.0],
        ],  # FRONT_LEFT
        [
            [809.2209905677063, 0.0, 829.2196003259838],
            [0.0, 809.2209905677063, 481.77842384512485],
            [0.0, 0.0, 1.0],
        ],  # BACK
        [
            [1256.7414812095406, 0.0, 792.1125740759628],
            [0.0, 1256.7414812095406, 492.7757465151356],
            [0.0, 0.0, 1.0],
        ],  # BACK_LEFT
        [
            [1259.5137405846733, 0.0, 807.2529053838625],
            [0.0, 1259.5137405846733, 501.19579884916527],
            [0.0, 0.0, 1.0],
        ],  # BACK_RIGHT
    ]

    cam2img = []
    for intrinsic in intrinsics_original:
        K = np.array(intrinsic, dtype=np.float32)
        # Scale for 800x320 resolution
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy

        # Convert to 4x4
        K_4x4 = np.array(
            [[K[0, 0], 0, K[0, 2], 0], [0, K[1, 1], K[1, 2], 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        cam2img.append(K_4x4)

    # extrinsics for the input used from the nuscenes mini dataset
    translations = [
        [1.70079118954, 0.0159456324149, 1.51095763913],
        [1.5508477543, -0.493404796419, 1.49574800619],
        [1.52387798135, 0.494631336551, 1.50932822144],
        [0.0283260309358, 0.00345136761476, 1.57910346144],
        [1.03569100218, 0.484795032713, 1.59097014818],
        [1.0148780988, -0.480568219723, 1.56239545128],
    ]
    # rotations for the input used from the nuscenes mini dataset
    rotations_quat = [
        [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
        [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485],
        [0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068],
        [0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578],
        [0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753],
        [0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
    ]

    # Convert to lidar2cam
    lidar2cam = []
    for trans, rot_quat in zip(translations, rotations_quat):
        x, y, z, w = rot_quat
        # Normalize
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        rot_matrix = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

        # Build 4x4 transformation
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot_matrix.astype(np.float32)
        T[:3, 3] = np.array(trans, dtype=np.float32)

        lidar2cam.append(T)

    lidar2cam = np.stack(lidar2cam)

    # Compute lidar2img
    lidar2img = [cam2img[i] @ lidar2cam[i] for i in range(num_cams)]

    img_metas_dict = dict()
    img_metas_dict["lidar2cam"] = lidar2cam
    img_metas_dict["img_shape"] = (320, 800)
    img_metas_dict["ori_shape"] = (900, 1600)
    img_metas_dict["pad_shape"] = (320, 800)
    img_metas_dict["input_shape"] = (320, 800)
    img_metas_dict["scale_factor"] = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    img_metas_dict["box_type_3d"] = type("LiDARInstance3DBoxes", (), {})
    img_metas_dict["cam2img"] = cam2img
    img_metas_dict["lidar2img"] = lidar2img
    img_metas_dict["flip"] = False
    img_metas_dict["flip_direction"] = None
    img_metas_dict["sample_idx"] = "ca9a282c9e77460f8360f564131a8af5"
    img_metas_dict["scene_token"] = "cc8c0bf57f984915a77078b10eb33198"
    img_metas_dict["timestamp"] = 1532402928147847
    img_metas_dict["pc_range"] = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    modified_batch_img_metas = [img_metas_dict]

    return inputs, modified_batch_img_metas


class ModuleArgs(dict):
    ...


class Conv2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class MaxPool2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_ttnn_module_args_cp_fpn(*, model, run_model, device):
    if run_model is None:
        return None

    with trace():
        output = run_model(model)

    def insert_nested(d, path, value):
        for key in path[:-1]:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            d = d.setdefault(key, {})
        last = path[-1]
        last = int(last) if isinstance(last, str) and last.isdigit() else last
        d[last] = value

    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}

        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes.get("operation")

            if not isinstance(operation, ttnn.tracer.TorchModule):
                continue

            module_path = operation.module.__ttnn_tracer_name__.split(".")
            in_edges = list(graph.in_edges(node, data=True))
            if not in_edges:
                continue

            input_node, _, edge_data = in_edges[0]
            input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

            module = operation.module

            if isinstance(module, torch.nn.Conv2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    Conv2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    ),
                )

            elif isinstance(module, torch.nn.MaxPool2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    MaxPool2dArgs(
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            else:
                nested = _infer_ttnn_module_args(operation.graph)
                if nested:
                    container_name = None
                    container_idx = None
                    for i in range(len(module_path) - 1, -1, -1):
                        part = module_path[i]
                        if isinstance(part, str) and part.isdigit():
                            container_idx = int(part)
                        elif not container_name:
                            container_name = part
                            break

                    if container_name and container_name in nested:
                        nested_content = nested[container_name]
                        if (
                            isinstance(nested_content, dict)
                            and container_idx is not None
                            and container_idx in nested_content
                        ):
                            content_at_index = nested_content[container_idx]
                            path_to_insert = module_path[:-1]
                            insert_nested(ttnn_module_args, path_to_insert + [container_idx], content_at_index)
                        elif isinstance(nested_content, dict):
                            path_to_container = module_path[:-1] if container_idx is not None else module_path
                            target = ttnn_module_args
                            for key in path_to_container:
                                key = int(key) if isinstance(key, str) and key.isdigit() else key
                                target = target.setdefault(key, {})
                            for key, val in nested_content.items():
                                key = int(key) if isinstance(key, str) and key.isdigit() else key
                                target[key] = val
                        else:
                            insert_nested(
                                ttnn_module_args,
                                module_path[:-1] if container_idx is not None else module_path,
                                nested_content,
                            )
                    else:
                        insert_nested(ttnn_module_args, module_path, nested)

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    full_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))
    result = full_args.get("", full_args)

    def collapse_nesting(d):
        if not isinstance(d, dict):
            return d

        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                if key in value and isinstance(value[key], dict):
                    result[key] = collapse_nesting(value[key])
                else:
                    collapsed_value = {}
                    for k, v in value.items():
                        if isinstance(v, dict) and key in v:
                            collapsed_value[k] = collapse_nesting(v[key])
                        else:
                            collapsed_value[k] = collapse_nesting(v)
                    result[key] = collapsed_value
            else:
                result[key] = value
        return result

    return make_dot_access_dict(collapse_nesting(result), ignore_types=(ModuleArgs,))


def create_petr_transformer_input_tensors(
    device,
):
    x = torch.rand(1, 6, 256, 20, 50)
    mask = torch.zeros((1, 6, 20, 50), dtype=torch.bool)
    mask_int = torch.zeros((1, 6, 20, 50))
    query_embed = torch.rand(900, 256)
    pos_embed = torch.rand(1, 6, 256, 20, 50)

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    ttnn_mask = ttnn.from_torch(mask_int, device=device)
    ttnn_query_embed = ttnn.from_torch(query_embed, dtype=ttnn.bfloat16, device=device)
    ttnn_pos_embed = ttnn.from_torch(pos_embed, dtype=ttnn.bfloat16, device=device)

    return x, mask, query_embed, pos_embed, ttnn_x, ttnn_mask, ttnn_query_embed, ttnn_pos_embed


def create_petr_transformer_model_parameters(model: PETRTransformer, x, mask, query_embed, pos_embed, device):
    parameters = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(x, mask, query_embed, pos_embed), device=None
    )
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    return parameters
