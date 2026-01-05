# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
import tracy

# Import common utilities
from models.experimental.BevDepth.common import (
    create_dummy_inputs,
)

# Import parameter preparation functions from custom_preprocessing
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_all_parameters_from_reference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bevdepth_e2e(device):
    """
    Full BEVDepth E2E test: backbone → head with single weight loading.
    Reports only the final output PCC (TTNN vs Reference).
    """
    from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
    from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations

    # Set seeds for reproducibility
    torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True

    logger.info("=== BEVDepth E2E Test ===")

    params, reference_model = prepare_all_parameters_from_reference(device)
    if params is None:
        pytest.skip("Reference model not available")
        return

    torch_model = reference_model.model
    torch_model.eval()

    torch_input_imgs, mats_dict = create_dummy_inputs(batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=704)

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
        "use_torch_fallback": True,
    }

    logger.info("Initializing TTNN model...")
    ttnn_backbone = TtBaseLSSFPN(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    head_model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
    }
    ttnn_head = TtBEVDepthHead(
        parameters=params["head"],
        model_config=head_model_config,
        layer_optimisations=head_optimisations,
        device=device,
    )

    logger.info("Running reference model...")
    with torch.no_grad():
        # Get BEV feature from reference model for comparison
        ref_bev_feature = torch_model.backbone(torch_input_imgs, mats_dict, is_return_depth=False)
        ref_output = torch_model.head(ref_bev_feature)

    logger.info("Running TTNN model...")
    ttnn_bev_feature = ttnn_backbone(torch_input_imgs, mats_dict, is_return_depth=False)

    # bev_pcc_result = comp_pcc(ref_bev_feature, ttnn_bev_feature)
    # bev_pcc_value = bev_pcc_result[1] if isinstance(bev_pcc_result, tuple) else bev_pcc_result
    # logger.info(f"BEV feature PCC: {bev_pcc_value:.10f}")
    ref_bev_input = ttnn.from_torch(
        ref_bev_feature.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    ref_bev_input = ttnn.to_device(ref_bev_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_bev_input = ttnn.from_torch(
        ttnn_bev_feature.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_bev_input = ttnn.to_device(ttnn_bev_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    tracy.signpost("start")
    ttnn_output = ttnn_head(ttnn_bev_input, device=device)
    tracy.signpost("stop")

    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_task_groups = len(ref_output)

    logger.info("\n" + "=" * 60)
    logger.info("BEVDepth E2E PCC Results - Per Head Task Output")
    logger.info("=" * 60)

    for task_idx in range(num_task_groups):
        logger.info(f"\nHead {task_idx}:")
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
