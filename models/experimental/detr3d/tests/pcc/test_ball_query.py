# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.detr3d.reference.torch_pointnet2_ops import BallQuery
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnBallQuery


def print_detailed_comparison(ref_tensor, ttnn_tensor, max_printable=10):
    """Print detailed comparison with mismatches highlighted"""
    print(f"ref:\n{ref_tensor}")
    print(f"ttnn:\n{ttnn_tensor}")

    # Find and print mismatches
    diff = torch.abs(ref_tensor - ttnn_tensor)
    mismatch_positions = torch.nonzero(diff, as_tuple=False)

    if len(mismatch_positions) > 0:
        print(f"\nMismatches found:")
        for i, pos in enumerate(mismatch_positions[:max_printable]):
            idx_tuple = tuple(pos.tolist())
            ref_val = ref_tensor[idx_tuple]
            ttnn_val = ttnn_tensor[idx_tuple]
            print(f"  Position {idx_tuple}: ref={ref_val}, ttnn={ttnn_val}")
    else:
        print("\n✓ No mismatches found")


@pytest.mark.parametrize(
    "radius, nsample, xyz_shape, new_xyz_shape",
    [
        (
            0.4,  # radius
            32,  # nsample
            (1, 2048, 3),  # xyz
            (1, 1024, 3),  # new_xyz
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ball_query_only(
    radius,
    nsample,
    xyz_shape,
    new_xyz_shape,
    device,
):
    torch.manual_seed(420)
    # Create torch BallQuery model
    torch_ball_query = BallQuery(radius=radius, nsample=nsample)

    # Create test data
    xyz = torch.randn(xyz_shape, dtype=torch.float32)
    new_xyz = torch.randn(new_xyz_shape, dtype=torch.float32)

    # Get reference output from torch
    ref_idx = torch_ball_query(xyz=xyz, new_xyz=new_xyz)

    # Create ttnn BallQuery model
    ttnn_ball_query = TtnnBallQuery(
        device=device,
        radius=radius,
        nsample=nsample,
    )

    # Convert inputs to ttnn format
    ttnn_xyz = ttnn.from_torch(xyz, dtype=ttnn.float32, device=device)
    ttnn_new_xyz = ttnn.from_torch(new_xyz, dtype=ttnn.float32, device=device)

    # Get ttnn output
    tt_idx = ttnn_ball_query(xyz=ttnn_xyz, new_xyz=ttnn_new_xyz)
    tt_idx_torch = ttnn.to_torch(tt_idx)
    print_detailed_comparison(ref_idx[0, 0], tt_idx_torch[0, 0], 10)

    # Compare indices
    passing, pcc_message = comp_pcc(ref_idx.float(), tt_idx_torch.float(), 0.99)
    logger.info(f"BallQuery Indices PCC: {pcc_message}")
    logger.info(comp_allclose(ref_idx, tt_idx_torch))

    if passing:
        logger.info("BallQuery Test Passed!")
    else:
        logger.warning("BallQuery Test Failed!")
        assert False, "BallQuery failed PCC check with threshold 0.99"
