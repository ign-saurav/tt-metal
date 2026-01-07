# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
import tracy
from models.experimental.BevDepth.common import (
    create_reference_inputs,
)
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_all_parameters_from_reference,
)
from models.experimental.BevDepth.tt.ttnn_bevdepth import TtBEVDepth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bevdepth(device):
    torch.manual_seed(42)

    params, reference_model = prepare_all_parameters_from_reference(device)
    torch_model = reference_model.model
    torch_model.eval()

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

    ttnn_model = TtBEVDepth(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        head_parameters=params["head"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    with torch.no_grad():
        ref_output = torch_model(torch_input_imgs, mats_dict)

    tracy.signpost("start")
    ttnn_output = ttnn_model(torch_input_imgs, mats_dict)
    tracy.signpost("stop")

    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_task_groups = len(ref_output)

    for task_idx in range(num_task_groups):
        for key in output_keys:
            ref_tensor = ref_output[task_idx][0][key]
            ttnn_tensor, _ = ttnn_output[task_idx][key]
            ttnn_tensor_torch = ttnn.to_torch(ttnn_tensor, device=device)

            expected_shape = ref_tensor.shape
            ttnn_tensor_torch = torch.reshape(
                ttnn_tensor_torch,
                (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
            )
            ttnn_tensor_torch = torch.permute(ttnn_tensor_torch, (0, 3, 1, 2))

            ref_tensor = ref_tensor.float().contiguous()
            ttnn_tensor_torch = ttnn_tensor_torch.float().contiguous()

            pcc_result = comp_pcc(ref_tensor, ttnn_tensor_torch)
            pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

            logger.info(f"  Head {task_idx}, key '{key}': PCC = {pcc_value:.10f}")
            assert pcc_value > 0.97, f"PCC value {pcc_value} is less than 0.97"
