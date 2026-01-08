# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
from models.experimental.BevDepth.common import load_reference_model, create_reference_inputs
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_backbone_parameters,
    prepare_neck_parameters,
    prepare_depthnet_parameters,
    extract_depthnet_state_dict,
)
from models.experimental.BevDepth.common import download_bevdepth_weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_backbone(device):
    torch.manual_seed(42)

    reference_model = load_reference_model()
    torch_backbone = reference_model.model.backbone
    torch_backbone.eval()

    torch_input_imgs, mats_dict = create_reference_inputs(
        batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=704
    )

    lss_conf = {
        "x_bound": [-51.2, 51.2, 0.8],
        "y_bound": [-51.2, 51.2, 0.8],
        "z_bound": [-5.0, 3.0, 0.2],
        "d_bound": [2.0, 58.0, 0.5],
        "final_dim": [256, 704],
        "downsample_factor": 16,
        "output_channels": 80,
    }

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "batch_size": 1,
        "neck_in_channels": [256, 512, 1024, 2048],
        "neck_out_channels": [128, 128, 128, 128],
        "neck_upsample_strides": [0.25, 0.5, 1, 2],
        "depthnet_in_channels": 512,
        "depthnet_mid_channels": 512,
        "depthnet_context_channels": 80,
        "depthnet_depth_channels": 112,
    }

    backbone_params = prepare_backbone_parameters()
    neck_params = prepare_neck_parameters()
    depthnet_params = prepare_depthnet_parameters(extract_depthnet_state_dict(download_bevdepth_weights()))

    ttnn_model = TtBaseLSSFPN(
        device=device,
        backbone_parameters=backbone_params,
        neck_parameters=neck_params,
        depthnet_parameters=depthnet_params,
        lss_conf=lss_conf,
        model_config=model_config,
    )

    with torch.no_grad():
        torch_output = torch_backbone(torch_input_imgs, mats_dict, is_return_depth=False)

    ttnn_output = ttnn_model(torch_input_imgs, mats_dict, is_return_depth=False)

    ref_output = torch_output.cpu().float()
    ttnn_output_float = ttnn_output.cpu().float() if isinstance(ttnn_output, torch.Tensor) else ttnn_output

    pcc_result = comp_pcc(ref_output, ttnn_output_float)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    logger.info(f"PCC: {pcc_value:.6f}")
    assert pcc_value > 0.99, f"PCC {pcc_value:.6f} is below threshold 0.99"
