# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.model.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("nclasses", [3])
def test_pointpillars_full_pipeline(device, nclasses, reset_seeds):
    torch.manual_seed(0)

    # Model parameters
    voxel_size = [0.16, 0.16, 4]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    max_num_points = 32
    max_voxels = (16000, 40000)

    torch_model = PointPillars(
        nclasses=nclasses,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
    )

    # Load pretrained weights from .pth file
    try:
        checkpoint = torch.load("epoch_160.pth", map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        torch_model.load_state_dict(state_dict)
        logger.info("Successfully loaded pretrained weights from epoch_160.pth")
    except FileNotFoundError:
        logger.warning("Checkpoint file 'epoch_160.pth' not found, using random weights")

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
    torch_cls, torch_reg, torch_dir = torch_model(batched_pts)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Create preprocessor
    preprocessor = PointPillarsPreprocessor(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
        parameters=parameters,
        device=device,
    )

    tt_model = TtPointPillars(
        nclasses=nclasses,
        parameters=parameters,
        device=device,
    )

    pillar_features = preprocessor.forward(batched_pts)
    pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))  # NHWC to NCHW [1, 64, 496, 432] to [1, 496, 432, 64]
    pillar_features = ttnn.reshape(
        pillar_features,
        (pillar_features.shape[0], 1, pillar_features.shape[1] * pillar_features.shape[2], pillar_features.shape[3]),
    )

    tt_cls, tt_reg, tt_dir = tt_model.forward(pillar_features)

    # Compare classification output
    tt_cls_torch = tt2torch_tensor(tt_cls)
    tt_cls_torch = tt_cls_torch.permute(0, 3, 1, 2)
    passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.97)
    logger.info(f"Classification PCC: {pcc_cls}")
    assert passing_cls, f"Classification PCC check failed: {pcc_cls}"

    # Compare regression output
    tt_reg_torch = tt2torch_tensor(tt_reg)
    tt_reg_torch = tt_reg_torch.permute(0, 3, 1, 2)
    passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
    logger.info(f"Regression PCC: {pcc_reg}")
    assert passing_reg, f"Regression PCC check failed: {pcc_reg}"

    # Compare direction output
    tt_dir_torch = tt2torch_tensor(tt_dir)
    tt_dir_torch = tt_dir_torch.permute(0, 3, 1, 2)
    passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
    logger.info(f"Direction PCC: {pcc_dir}")
    assert passing_dir, f"Direction PCC check failed: {pcc_dir}"

    logger.info("Full PointPillars pipeline test passed!")
