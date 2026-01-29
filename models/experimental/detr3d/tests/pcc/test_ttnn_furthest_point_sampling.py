# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.detr3d.reference import torch_pointnet2_ops
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnFurthestPointSampling


@pytest.mark.parametrize(
    "batch_size, num_points, npoint",
    [
        (1, 20000, 64),
        # (1, 20000, 2048), # actual case in 3detr
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_furthest_point_sampling_vs_torch(
    batch_size,
    num_points,
    npoint,
    device,
):
    """Compare TtnnFurthestPointSampling output indices to torch furthest_point_sample."""
    torch.manual_seed(0)
    xyz = torch.randn((batch_size, num_points, 3)) * 0.5 + 1.0

    # Reference: torch implementation
    ref_idx = torch_pointnet2_ops.furthest_point_sample(xyz, npoint)
    assert ref_idx.shape == (batch_size, npoint)
    assert ref_idx.dtype == torch.long

    # TTNN implementation
    ttnn_fps = TtnnFurthestPointSampling()
    points_ttnn = ttnn.from_torch(
        xyz,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_idx = ttnn_fps.forward(points_ttnn, npoint, device)
    tt_idx_torch = ttnn.to_torch(tt_idx)
    assert tt_idx_torch.shape == (batch_size, npoint)

    # Compare indices element-wise
    ref = ref_idx.cpu().long().numpy()
    tt = tt_idx_torch.cpu().long().numpy()
    assert (ref == tt).all(), (
        f"FPS indices differ: ref vs ttnn\n" f"Ref:\n{ref}\nTTNN:\n{tt}\nDiff-count: {(ref != tt).sum()}"
    )
