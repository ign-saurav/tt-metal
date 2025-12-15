# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pillar_encoder import TtPillarEncoder
from models.experimental.pointpillars.reference.model.pointpillars import PillarEncoder
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "voxel_size,point_cloud_range,in_channel,out_channel",
    [
        ([0.16, 0.16, 4], [0, -39.68, -3, 69.12, 39.68, 1], 9, 64),
    ],
)
def test_pillar_encoder(device, voxel_size, point_cloud_range, in_channel, out_channel, reset_seeds):
    torch.manual_seed(0)

    # Create reference model
    torch_model = PillarEncoder(voxel_size, point_cloud_range, in_channel, out_channel)

    try:
        checkpoint = torch.load("epoch_160.pth", map_location="cpu")

        # Extract Backbone weights from the full model checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter only Backbone weights
        pillar_encoder_state_dict = {}
        prefix = "pillar_encoder."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                pillar_encoder_state_dict[new_key] = value

        torch_model.load_state_dict(pillar_encoder_state_dict)
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, using random weights")

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
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
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
