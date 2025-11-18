# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc
from models.experimental.pointpillars.tt.pillar_encoder import TtPillarEncoder
from models.experimental.pointpillars.reference.model.pointpillars import PillarEncoder
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "voxel_size,point_cloud_range,in_channel,out_channel",
    [
        ([0.16, 0.16, 4], [0, -39.68, -3, 69.12, 39.68, 1], 9, 64),
    ],
)
def test_pillar_encoder(device, voxel_size, point_cloud_range, in_channel, out_channel, reset_seeds):
    torch.manual_seed(0)

    # Create reference model
    torch_model = PillarEncoder(voxel_size, point_cloud_range, in_channel, out_channel).eval()

    # Generate test inputs
    num_pillars = 6169
    num_points = 32
    num_features = 4

    pillars = torch.randn(num_pillars, num_points, num_features, dtype=torch.bfloat16)
    coors_batch = torch.randint(0, 4, (num_pillars, 4), dtype=torch.long)
    coors_batch[:, 0] = torch.randint(0, 2, (num_pillars,))
    npoints_per_pillar = torch.randint(1, num_points + 1, (num_pillars,), dtype=torch.long)

    # Get reference output
    torch_output = torch_model(pillars, coors_batch, npoints_per_pillar)

    # Preprocess model parameters using the custom preprocessor
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Create TTNN model with preprocessed parameters
    tt_model = TtPillarEncoder(
        device=device,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        in_channel=in_channel,
        out_channel=out_channel,
        parameters=parameters,
    )

    # Get TTNN output
    tt_output = tt_model.forward(pillars, coors_batch, npoints_per_pillar)

    # Compare outputs
    passing, pcc = comp_pcc(torch_output, tt_output, 0.98)
    logger.info(f"PCC: {pcc}")
    assert passing, f"PCC check failed: {pcc}"

    logger.info("PillarEncoder test passed!")
