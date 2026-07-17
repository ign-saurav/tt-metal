# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.pointpillars.common import (
    VOXEL_SIZE,
    POINT_CLOUD_RANGE,
    MAX_NUM_POINTS,
    MAX_VOXELS,
    load_checkpoint,
    download_checkpoint,
)
import tracy


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("nclasses", [3])
def test_pointpillars_full_pipeline(device, nclasses, reset_seeds):
    """Test full PointPillars pipeline comparing PyTorch and TTNN outputs."""
    torch.manual_seed(0)

    torch_model = PointPillars(
        nclasses=nclasses,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=MAX_NUM_POINTS,
        max_voxels=MAX_VOXELS,
    )

    checkpoint_dir = "models/experimental/pointpillars/resources/checkpoint"
    checkpoint_path = download_checkpoint(checkpoint_dir)
    state_dict = load_checkpoint(checkpoint_path)
    if state_dict is not None:
        torch_model.load_state_dict(state_dict)

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
    torch_cls, torch_reg, torch_dir = torch_model(batched_pts)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    preprocessor = PointPillarsPreprocessor(
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=MAX_NUM_POINTS,
        max_voxels=MAX_VOXELS,
        parameters=parameters,
        device=device,
    )

    tt_model = TtPointPillars(
        nclasses=nclasses,
        parameters=parameters,
        device=device,
    )

    pillar_features = preprocessor.forward(batched_pts)
    pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))
    pillar_features = ttnn.reshape(
        pillar_features,
        (pillar_features.shape[0], 1, pillar_features.shape[1] * pillar_features.shape[2], pillar_features.shape[3]),
    )

    tracy.signpost("start")
    tt_cls, tt_reg, tt_dir = tt_model.forward(pillar_features)
    tracy.signpost("stop")

    # Compare classification output
    tt_cls_torch = tt2torch_tensor(tt_cls).permute(0, 3, 1, 2)
    passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.97)
    logger.info(f"Classification PCC: {pcc_cls}")
    assert passing_cls, f"Classification PCC check failed: {pcc_cls}"

    # Compare regression output
    tt_reg_torch = tt2torch_tensor(tt_reg).permute(0, 3, 1, 2)
    passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
    logger.info(f"Regression PCC: {pcc_reg}")
    assert passing_reg, f"Regression PCC check failed: {pcc_reg}"

    # Compare direction output
    tt_dir_torch = tt2torch_tensor(tt_dir).permute(0, 3, 1, 2)
    passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
    logger.info(f"Direction PCC: {pcc_dir}")
    assert passing_dir, f"Direction PCC check failed: {pcc_dir}"

    logger.info("Full PointPillars pipeline test passed!")
