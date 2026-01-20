# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pillar_encoder import TtPillarEncoder
from models.experimental.pointpillars.reference.pointpillars import PillarEncoder
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.pointpillars.common import (
    VOXEL_SIZE,
    POINT_CLOUD_RANGE,
    load_checkpoint,
    extract_component_state_dict,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channel,out_channel",
    [
        (9, 64),
    ],
)
def test_pillar_encoder(device, in_channel, out_channel, reset_seeds):
    """Test TtPillarEncoder against PyTorch reference."""
    torch.manual_seed(0)

    torch_model = PillarEncoder(VOXEL_SIZE, POINT_CLOUD_RANGE, in_channel, out_channel)

    state_dict = load_checkpoint("epoch_160.pth")
    if state_dict is not None:
        pillar_encoder_state_dict = extract_component_state_dict(state_dict, "pillar_encoder.")
        torch_model.load_state_dict(pillar_encoder_state_dict)

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    num_pillars = 6169
    num_points = 32
    num_features = 4

    pillars = torch.randn(num_pillars, num_points, num_features, dtype=torch.bfloat16)
    coors_batch = torch.randint(0, 4, (num_pillars, 4), dtype=torch.long)
    coors_batch[:, 0] = torch.randint(0, 2, (num_pillars,))
    npoints_per_pillar = torch.randint(1, num_points + 1, (num_pillars,), dtype=torch.long)

    torch_output = torch_model(pillars, coors_batch, npoints_per_pillar)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    tt_model = TtPillarEncoder(
        device=device,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        in_channel=in_channel,
        out_channel=out_channel,
        parameters=parameters["pillar_encoder"],
    )

    tt_output = tt_model.forward(pillars, coors_batch, npoints_per_pillar)
    tt_output = tt2torch_tensor(tt_output)

    passing, pcc = comp_pcc(torch_output, tt_output, 0.99)
    logger.info(f"PCC: {pcc}")
    assert passing, f"PCC check failed: {pcc}"

    logger.info("PillarEncoder test passed!")
