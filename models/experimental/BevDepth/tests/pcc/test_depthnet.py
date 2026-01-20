# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_depthnet import TtDepthNet
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_depthnet_parameters,
    extract_depthnet_state_dict,
)
from models.experimental.BevDepth.common import download_bevdepth_weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(64, 160)])
@pytest.mark.parametrize("depth_channels", [112])
def test_depthnet_pcc(device, batch_size, height, width, depth_channels):
    from models.experimental.BevDepth.reference.base_lss_fpn import DepthNet

    torch.manual_seed(42)

    torch_input = torch.randn(batch_size, 512, height, width)

    reference_depthnet = DepthNet(
        in_channels=512,
        mid_channels=512,
        context_channels=80,
        depth_channels=depth_channels,
    )

    weights_path = download_bevdepth_weights()
    depthnet_state = extract_depthnet_state_dict(weights_path)

    if len(depthnet_state) == 0:
        pytest.skip("No DepthNet weights in checkpoint")

    reference_depthnet.load_state_dict(
        {k.replace("model.backbone.depth_net.", ""): v for k, v in depthnet_state.items() if "depth_net" in k},
        strict=False,
    )
    reference_depthnet.eval()

    mats_dict = {
        "intrin_mats": torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),
        "ida_mats": torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),
        "sensor2ego_mats": torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),
        "bda_mat": torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    }

    with torch.no_grad():
        try:
            ref_output = reference_depthnet(torch_input, mats_dict=mats_dict)
        except Exception:
            pytest.skip("Reference model requires camera parameters")

    depthnet_params = prepare_depthnet_parameters(
        depthnet_state,
        in_channels=512,
        mid_channels=512,
        depth_channels=depth_channels,
    )

    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    }

    ttnn_depthnet = TtDepthNet(
        device=device,
        parameters=depthnet_params,
        in_channels=512,
        mid_channels=512,
        context_channels=80,
        depth_channels=depth_channels,
        model_config=model_config,
    )

    torch_input_hwc = torch_input.permute(0, 2, 3, 1).contiguous()
    ttnn_input = ttnn.from_torch(
        torch_input_hwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    ttnn_output = ttnn_depthnet(ttnn_input, batch_size=batch_size, mats_dict=mats_dict)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2).contiguous()

    ref_depth = ref_output[:, :depth_channels, :, :]
    ttnn_depth = ttnn_output_torch[:, :depth_channels, :, :]

    pcc_result = comp_pcc(ref_depth, ttnn_depth)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    logger.info(f"DepthNet: PCC = {pcc_value:.6f}")
    assert pcc_value > 0.99, f"DepthNet PCC {pcc_value:.6f} is below threshold 0.99"
