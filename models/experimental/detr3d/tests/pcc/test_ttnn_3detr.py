# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import tracy

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.model_config import Tt3DetrArgs
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.ttnn.utils import (
    box_post_processing as tt_box_post_processing,
    infer_ttnn_module_args,
    NO_FALLBACK,
)
from models.experimental.detr3d.reference.model_3detr import build_3detr, box_post_processing
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import SunrgbdDatasetConfig
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 20000, 3),
    ],
)
@pytest.mark.parametrize("encoder_only", (False,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_3detr_model(encoder_only, input_shape, device):
    torch.manual_seed(0)
    # Configuration flags
    PCC_THRESHOLD = 0.92
    CHECK_AUX_OUTPUTS = False  # Set to True to enable PCC check for auxiliary outputs
    SKIP_KEYS = ["angle_continuous", "objectness_prob"]  # Keys to skip in PCC comparison

    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    # Define the shape and range
    min_val = -1.8827
    max_val = 8.3542
    pc = (max_val - min_val) * torch.rand(input_shape) + min_val
    input_dict = {
        "point_clouds": pc,
        "point_cloud_dims_min": torch.min(pc, 1)[0],
        "point_cloud_dims_max": torch.max(pc, 1)[0],
    }

    ref_module, _ = build_3detr(args, dataset_config)
    load_torch_model_state(ref_module)

    (
        torch_cls_logits,
        torch_center_offset,
        torch_size_normalized,
        torch_angle_logits,
        torch_angle_residual_normalized,
        torch_angle_residual,
        torch_num_layers,
        torch_query_xyz,
        torch_point_cloud_dims,
    ) = ref_module(inputs=input_dict, encoder_only=encoder_only)
    ref_out = box_post_processing(
        torch_cls_logits,
        torch_center_offset,
        torch_size_normalized,
        torch_angle_logits,
        torch_angle_residual_normalized,
        torch_angle_residual,
        torch_num_layers,
        torch_query_xyz,
        torch_point_cloud_dims,
        dataset_config,
    )

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ref_module_parameters.layer_args = {}
    ref_module_parameters.layer_args = infer_ttnn_module_args(
        model=ref_module,
        run_model=lambda model: ref_module(inputs=input_dict, encoder_only=encoder_only),
        device=device,
    )
    if NO_FALLBACK:
        ttnn_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                ttnn_dict[key] = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            else:
                ttnn_dict[key] = value
        input_dict = ttnn_dict

    ttnn_args = Tt3DetrArgs()
    ttnn_args.parameters = ref_module_parameters
    ttnn_args.device = device
    ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

    tracy.signpost("start")
    (
        cls_logits,
        center_offset,
        size_normalized,
        angle_logits,
        angle_residual_normalized,
        angle_residual,
        num_layers,
        query_xyz,
        point_cloud_dims,
    ) = ttnn_module(inputs=input_dict, encoder_only=encoder_only)
    tracy.signpost("stop")

    tt_output = tt_box_post_processing(
        cls_logits,
        center_offset,
        size_normalized,
        angle_logits,
        angle_residual_normalized,
        angle_residual,
        num_layers,
        query_xyz,
        point_cloud_dims,
        dataset_config,
    )

    all_passing = True

    ttnn_outputs, ref_outputs = tt_output["outputs"], ref_out["outputs"]
    ttnn_aux_outputs, ref_aux_outputs = tt_output["aux_outputs"], ref_out["aux_outputs"]

    # Check main outputs
    for key in ref_outputs:
        if key in SKIP_KEYS:
            logger.info(f"Output Key '{key}' - Skipped (in SKIP_KEYS)")
            continue

        passing, pcc_message = comp_pcc(ref_outputs[key], ttnn_outputs[key], PCC_THRESHOLD)
        logger.info(f"Output Key '{key}' PCC: {pcc_message}")
        logger.info(comp_allclose(ref_outputs[key], ttnn_outputs[key]))

        if passing:
            logger.info(f"Output Key '{key}' Test Passed!")
        else:
            logger.warning(f"Output Key '{key}' Test Failed!")
            all_passing = False

    # Check auxiliary outputs only if flag is enabled
    if CHECK_AUX_OUTPUTS:
        for i in range(len(ref_aux_outputs)):
            for key in ref_aux_outputs[i]:
                if key in SKIP_KEYS:
                    logger.info(f"Aux Output {i} Key '{key}' - Skipped (in SKIP_KEYS)")
                    continue

                passing, pcc_message = comp_pcc(ref_aux_outputs[i][key], ttnn_aux_outputs[i][key], PCC_THRESHOLD)
                logger.info(f"Aux Output {i} Key '{key}' PCC: {pcc_message}")
                logger.info(comp_allclose(ref_aux_outputs[i][key], ttnn_aux_outputs[i][key]))

                if passing:
                    logger.info(f"Aux Output {i} Key '{key}' Test Passed!")
                else:
                    logger.warning(f"Aux Output {i} Key '{key}' Test Failed!")
                    all_passing = False
    else:
        logger.info("Auxiliary outputs PCC check is disabled (CHECK_AUX_OUTPUTS=False)")

    assert all_passing, f"One or more outputs failed PCC check with threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 20000, 3),
    ],
)
@pytest.mark.parametrize("encoder_only", (False,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_3detr_model_raw_outputs(encoder_only, input_shape, device):
    torch.manual_seed(0)
    # Configuration flags
    PCC_THRESHOLD = 0.92
    LOAD_REAL_INPUT = False

    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    # Define the shape and range
    if LOAD_REAL_INPUT:
        input_dict = torch.load("models/experimental/detr3d/resources/inputs.pt", map_location="cpu")
        print("REAL INPUTS LOADED")
    else:
        min_val = -1.8827
        max_val = 8.3542
        pc = (max_val - min_val) * torch.rand(input_shape) + min_val
        input_dict = {
            "point_clouds": pc,
            "point_cloud_dims_min": torch.min(pc, 1)[0],
            "point_cloud_dims_max": torch.max(pc, 1)[0],
        }

    ref_module, _ = build_3detr(args, dataset_config)
    load_torch_model_state(ref_module)

    # Get raw outputs from reference model
    (
        torch_cls_logits,
        torch_center_offset,
        torch_size_normalized,
        torch_angle_logits,
        torch_angle_residual_normalized,
        torch_angle_residual,
        torch_num_layers,
        torch_query_xyz,
        torch_point_cloud_dims,
    ) = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ref_module_parameters.layer_args = {}
    ref_module_parameters.layer_args = infer_ttnn_module_args(
        model=ref_module,
        run_model=lambda model: ref_module(inputs=input_dict, encoder_only=encoder_only),
        device=device,
    )

    if NO_FALLBACK:
        ttnn_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                ttnn_dict[key] = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            else:
                ttnn_dict[key] = value
        input_dict = ttnn_dict

    ttnn_args = Tt3DetrArgs()
    ttnn_args.parameters = ref_module_parameters
    ttnn_args.device = device

    ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

    # Get raw outputs from TTNN model
    if NO_FALLBACK:
        (
            cls_logits,
            center_offset,
            size_normalized,
            angle_logits,
            angle_residual_normalized,
            angle_residual,
            num_layers,
            ttnn_query_xyz,
            ttnn_point_cloud_dims,
        ) = ttnn_module(inputs=input_dict, encoder_only=encoder_only)
    else:
        (
            cls_logits,
            center_offset,
            size_normalized,
            angle_logits,
            angle_residual_normalized,
            angle_residual,
            num_layers,
            ttnn_query_xyz,
            ttnn_point_cloud_dims,
        ) = ttnn_module(inputs=input_dict, encoder_only=encoder_only)

    # Convert TTNN outputs to torch tensors for comparison
    cls_logits = ttnn.to_torch(cls_logits)
    center_offset = ttnn.to_torch(center_offset)
    size_normalized = ttnn.to_torch(size_normalized)
    angle_logits = ttnn.to_torch(angle_logits)
    angle_residual_normalized = ttnn.to_torch(angle_residual_normalized)
    angle_residual = ttnn.to_torch(angle_residual)
    ttnn_query_xyz = ttnn.to_torch(ttnn_query_xyz)
    for i in range(len(ttnn_point_cloud_dims)):
        ttnn_point_cloud_dims[i] = ttnn.to_torch(ttnn_point_cloud_dims[i])

    all_passing = True

    # Compare raw outputs without post-processing
    raw_outputs_pairs = [
        ("cls_logits", torch_cls_logits, cls_logits),
        ("center_offset", torch_center_offset, center_offset),
        ("size_normalized", torch_size_normalized, size_normalized),
        ("angle_logits", torch_angle_logits, angle_logits),
        ("angle_residual_normalized", torch_angle_residual_normalized, angle_residual_normalized),
        ("ttnn_query_xyz", torch_query_xyz, ttnn_query_xyz),
        ("angle_residual", torch_angle_residual, angle_residual),
    ]

    for name, ref_output, tt_output in raw_outputs_pairs:
        passing, pcc_message = comp_pcc(ref_output, tt_output, PCC_THRESHOLD)
        logger.info(f"Raw Output '{name}' PCC: {pcc_message}")
        logger.info(comp_allclose(ref_output, tt_output))

        if passing:
            logger.info(f"Raw Output '{name}' Test Passed!")
        else:
            logger.warning(f"Raw Output '{name}' Test Failed!")
            all_passing = False
    for i in range(len(torch_point_cloud_dims)):
        passing, pcc_message = comp_pcc(torch_point_cloud_dims[i], ttnn_point_cloud_dims[i], PCC_THRESHOLD)
        if passing:
            logger.info(f"point_cloud_dims' Test Passed!")
        else:
            logger.warning(f"point_cloud_dims' Test Failed!")
            all_passing = False
    assert all_passing, f"One or more raw outputs failed PCC check with threshold {PCC_THRESHOLD}"
