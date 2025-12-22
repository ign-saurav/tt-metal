# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import os
import urllib.request
import pytest
from models.experimental.petr.reference.vovnetcp import (
    VoVNetCP,
    Hsigmoid,
    eSEModule,
    _OSA_stage,
)
from models.experimental.petr.tt.ttnn_vovnetcp import (
    ttnn_hsigmoid,
    ttnn_eSEModule,
    ttnn_osa_stage,
    ttnn_VoVNetCP,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger
from models.experimental.petr.tt.common import (
    create_custom_preprocessor_vovnetcp,
    stem_parameters_preprocess,
)

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

    return make_dot_access_dict(collapse_nesting(result), ignore_types=(ModuleArgs,))


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 256, 1, 1),
        (6, 768, 1, 1),
        (6, 512, 1, 1),
        (6, 1024, 1, 1),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp_hsigmoid(device, reset_seeds, n, c, h, w):
    input_tensor = torch.randn((n, c, h, w))
    torch_model = Hsigmoid()
    torch_output = torch_model(input_tensor)
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_model = ttnn_hsigmoid(device)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_hsigmoid test passed: " f"PCC={msg}")


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 256, 80, 200),
        (6, 768, 20, 50),
        (6, 256, 40, 100),
        (6, 1024, 10, 25),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_esemodule(device, n, c, h, w):
    torch_input_tensor = torch.randn(n, c, h, w)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_model = eSEModule(c)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_module_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: model(torch_input_tensor), device=device
    )
    print(ttnn_module_args)
    ttnn_model = ttnn_eSEModule(parameters["fc"], model_config, ttnn_module_args["fc"], device)

    ttnn_output = ttnn_model(ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_esemodule test passed: " f"PCC={msg}")


@pytest.mark.parametrize(
    "in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num,input_shape",
    [
        (128, 128, 256, 1, 5, 2, [1, 128, 80, 200]),
        (256, 160, 512, 3, 5, 3, [1, 256, 80, 200]),
        (512, 192, 768, 9, 5, 4, [1, 512, 40, 100]),
        (768, 224, 1024, 3, 5, 5, [1, 768, 20, 50]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_osa_stage(
    device, reset_seeds, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, input_shape
):
    torch_input_tensor = torch.randn(input_shape)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    torch_model = _OSA_stage(
        in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    torch_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_osa_stage(
        parameters, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    if len(ttnn_output.shape) == 4 and ttnn_output.shape[2] == 1:
        # Calculate original H and W from the torch output shape
        target_h = torch_output.shape[2]
        target_w = torch_output.shape[3]
        ttnn_output = ttnn_output.reshape(ttnn_output.shape[0], ttnn_output.shape[1], target_h, target_w)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(f"vovnetcp_osa_stage_{stage_num} test passed: " f"PCC={msg}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnetcp(
    device,
):
    torch_input_tensor = torch.randn(1, 3, 320, 800)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        logger.info(f"Downloading PETR weights from {weights_url} ...")
        urllib.request.urlretrieve(weights_url, weights_path)
        logger.info(f"Weights downloaded to {weights_path}")

    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.load_state_dict(
        {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}
    )
    torch_model.eval()
    stem_parameters = stem_parameters_preprocess(torch_model)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_vovnetcp(None), device=None
    )

    with torch.no_grad():
        output = torch_model(torch_input_tensor)

    ttnn_model = ttnn_VoVNetCP(parameters, stem_parameters, device)

    ttnn_output = ttnn_model(device, ttnn_input_tensor)

    # Tensor Postprocessing
    #  Convert TTNN outputs to torch for comparison
    ttnn_out0_torch = ttnn.to_torch(ttnn_output[0]).permute(0, 3, 1, 2)
    ttnn_out1_torch = ttnn.to_torch(ttnn_output[1]).permute(0, 3, 1, 2)

    # Reshape if needed
    if ttnn_out0_torch.shape != output[0].shape:
        ttnn_out0_torch = ttnn_out0_torch.reshape(output[0].shape)
    if ttnn_out1_torch.shape != output[1].shape:
        ttnn_out1_torch = ttnn_out1_torch.reshape(output[1].shape)

    # Compare
    passed0, msg0 = check_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    passed1, msg1 = check_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)

    logger.info("=" * 60)
    logger.info("FINAL BACKBONE RESULTS:")
    logger.info("=" * 60)
    logger.info(f"Stage 4 output PCC: {msg0}")
    logger.info(f"Stage 5 output PCC: {msg1}")
    assert_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    assert_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)
    assert passed0, f"Stage 4 PCC failed: {msg0}"
    assert passed1, f"Stage 5 PCC failed: {msg1}"
