# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger
from models.experimental.petr.tt.ttnn_cp_fpn import ttnn_CPFPN
from models.experimental.petr.reference.cp_fpn import CPFPN
from models.experimental.petr.tt.common import create_custom_preprocessor_cpfpn
from ttnn.model_preprocessing import infer_ttnn_module_args
from ttnn.dot_access import make_dot_access_dict
from ttnn.torch_tracer import trace, visualize


class ModuleArgs(dict):
    ...


class Conv2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class ConvTranspose2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class MaxPool2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class GroupNormArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    # ------------------------------------------------------------------
    # Run model under TTNN tracing
    # ------------------------------------------------------------------
    with trace():
        output = run_model(model)

    visualize(output, file_name=ttnn.CONFIG.tmp_dir / "model_graph.svg")

    # ------------------------------------------------------------------
    # Helper: insert value into nested dict using module path
    # ------------------------------------------------------------------
    def insert_nested(d, path, value):
        for key in path[:-1]:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            d = d.setdefault(key, {})
        last = path[-1]
        last = int(last) if isinstance(last, str) and last.isdigit() else last
        d[last] = value

    # ------------------------------------------------------------------
    # Recursive graph walk
    # ------------------------------------------------------------------
    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}

        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes.get("operation")

            if not isinstance(operation, ttnn.tracer.TorchModule):
                continue

            # Full hierarchical module path
            module_path = operation.module.__ttnn_tracer_name__.split(".")

            # Infer input shape (assumes single input edge)
            in_edges = list(graph.in_edges(node, data=True))
            if not in_edges:
                continue

            input_node, _, edge_data = in_edges[0]
            input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

            module = operation.module

            # ----------------------------------------------------------
            # Conv2d
            # ----------------------------------------------------------
            if isinstance(module, torch.nn.Conv2d):
                print(f"Here " * 10)
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

            # ----------------------------------------------------------
            # ConvTranspose2d
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.ConvTranspose2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    ConvTranspose2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        output_padding=module.output_padding,
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

            # ----------------------------------------------------------
            # MaxPool2d
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # GroupNorm
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.GroupNorm):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    GroupNormArgs(
                        num_groups=module.num_groups,
                        num_channels=module.num_channels,
                        eps=module.eps,
                        affine=module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # ----------------------------------------------------------
            # Container / composite module → recurse
            # ----------------------------------------------------------
            else:
                nested = _infer_ttnn_module_args(operation.graph)
                if nested:
                    # When inserting nested results, check if there's redundant nesting
                    # Pattern: path is ['lateral_convs', '0'] and nested is {'lateral_convs': {0: {...}}}
                    # We want to extract the inner content and insert it directly

                    # Find the container name (last non-numeric element in path before the index)
                    container_name = None
                    container_idx = None
                    for i in range(len(module_path) - 1, -1, -1):
                        part = module_path[i]
                        if isinstance(part, str) and part.isdigit():
                            container_idx = int(part)
                        elif not container_name:
                            container_name = part
                            break

                    # If nested has the container name as a key, extract and merge
                    if container_name and container_name in nested:
                        nested_content = nested[container_name]
                        # If nested_content is a dict with the same index, extract that
                        if (
                            isinstance(nested_content, dict)
                            and container_idx is not None
                            and container_idx in nested_content
                        ):
                            # Extract the content at the index
                            content_at_index = nested_content[container_idx]
                            # Insert directly at the container[index] path
                            path_to_insert = module_path[:-1]  # Remove the index, keep container name
                            insert_nested(ttnn_module_args, path_to_insert + [container_idx], content_at_index)
                        elif isinstance(nested_content, dict):
                            # Merge all keys from nested_content
                            path_to_container = module_path[:-1] if container_idx is not None else module_path
                            target = ttnn_module_args
                            for key in path_to_container:
                                key = int(key) if isinstance(key, str) and key.isdigit() else key
                                target = target.setdefault(key, {})
                            # Merge nested_content into target
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

    # ------------------------------------------------------------------
    # Kick off inference from traced graph
    # ------------------------------------------------------------------
    full_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))

    # Root module is stored under empty name ""
    result = full_args.get("", full_args)

    # Post-process to collapse any remaining redundant nesting
    def collapse_nesting(d):
        """Recursively collapse redundant nesting in the result."""
        if not isinstance(d, dict):
            return d

        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Check if value has the same key as a nested dict (redundant nesting)
                # Pattern: {key: {key: {...}}} -> {key: {...}}
                if key in value and isinstance(value[key], dict):
                    result[key] = collapse_nesting(value[key])
                else:
                    # Check if any nested dict has 'key' as a key (for container modules)
                    # Pattern: {0: {'lateral_convs': {0: {...}}}} -> {0: {0: {...}}}
                    collapsed_value = {}
                    for k, v in value.items():
                        if isinstance(v, dict) and key in v:
                            # Extract the inner content at key
                            collapsed_value[k] = collapse_nesting(v[key])
                        else:
                            collapsed_value[k] = collapse_nesting(v)
                    result[key] = collapsed_value
            else:
                result[key] = value
        return result

    return collapse_nesting(result)


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cp_fpn(device, reset_seeds):
    torch_model = CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_cpfpn(None), device=None
    )

    batch_size = 6
    input_a = torch.randn(batch_size, 768, 20, 50)
    input_b = torch.randn(batch_size, 1024, 10, 25)
    torch_output = torch_model([input_a, input_b])
    ttnn_module_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: model([input_a, input_b]), device=device
    )
    print(f"ttnn_module_args.keys(): {ttnn_module_args.keys()}")
    print(f"ttnn_module_args['lateral_convs'].keys(): {ttnn_module_args['lateral_convs'].keys()}")
    print(f"ttnn_module_args['fpn_convs'].keys(): {ttnn_module_args['fpn_convs'].keys()}")
    print(f"ttnn_module_args: {ttnn_module_args}")
    ttnn_model = ttnn_CPFPN(
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2,
        batch_size=batch_size,
        parameters=parameters,
        model_config=model_config,
        model_args=ttnn_module_args,
        device=device,
    )

    ttnn_input_1 = ttnn.from_torch(input_a.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_2 = ttnn.from_torch(input_b.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn_model([ttnn_input_1, ttnn_input_2])

    for i in range(len(ttnn_output)):
        ttnn_output_check = ttnn.to_torch(ttnn_output[i])
        ttnn_output_check = ttnn_output_check.permute(0, 3, 1, 2)
        pcc_threshold = 0.99
        passed, msg = check_with_pcc(torch_output[i], ttnn_output_check, pcc=pcc_threshold)
        assert_with_pcc(ttnn_output_check, torch_output[i], pcc=0.99)
        logger.info(f"cp_fpn layer  passed: " f"PCC={msg}")
