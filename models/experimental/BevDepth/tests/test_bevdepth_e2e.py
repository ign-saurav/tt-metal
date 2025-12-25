# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc
import tracy


def download_bevdepth_weights():
    """Download BEVDepth pretrained weights"""
    import urllib.request
    import os

    url = "https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
    weights_path = "/tmp/bevdepth_weights.pth"

    if not os.path.exists(weights_path):
        logger.info(f"Downloading weights from {url}")
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Downloaded weights to {weights_path}")

    return weights_path


def load_reference_model():
    """Load the reference BEVDepth model."""
    from models.experimental.BevDepth.reference.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
        BEVDepthLightningModel,
    )

    logger.info("Loading reference BEVDepth model...")
    lightning_model = BEVDepthLightningModel()
    checkpoint_path = download_bevdepth_weights()

    import os

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return None

    lightning_model.load_checkpoint(checkpoint_path, verbose=False)
    lightning_model.model.eval()
    return lightning_model


def create_dummy_inputs(batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=704):
    """Create dummy input images and transformation matrices."""
    imgs = torch.randn((batch_size, num_sweeps, num_cameras, 3, img_h, img_w), dtype=torch.float32, requires_grad=False)

    mats_dict = {
        "sensor2ego_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "intrin_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "ida_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "sensor2sensor_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "bda_mat": torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    }

    return imgs, mats_dict


def prepare_backbone_parameters():
    """Prepare parameters for ResNet50 backbone."""
    from models.experimental.BevDepth.tests.test_bevdepth_backbone import (
        extract_backbone_state_dict,
        fuse_batchnorm_into_conv,
        prepare_ttnn_parameters,
    )

    logger.info("Preparing backbone parameters...")
    checkpoint_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(checkpoint_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)
    return prepare_ttnn_parameters(backbone_state)


def prepare_neck_parameters():
    """Prepare parameters for SECONDFPN neck."""
    from models.experimental.BevDepth.tests.test_bevdepth_backbone import extract_neck_state_dict
    from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters

    logger.info("Preparing neck parameters...")
    checkpoint_path = download_bevdepth_weights()
    neck_state = extract_neck_state_dict(checkpoint_path)

    in_channels = [256, 512, 1024, 2048]
    out_channels = [128, 128, 128, 128]
    upsample_strides = [0.25, 0.5, 1, 2]
    return prepare_secondfpn_parameters(
        neck_state,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides,
    )


def prepare_depthnet_parameters():
    """Prepare parameters for DepthNet."""
    from models.experimental.BevDepth.tests.test_bevdepth_backbone import extract_depthnet_state_dict
    from models.experimental.BevDepth.tt.ttnn_depthnet import prepare_depthnet_parameters

    logger.info("Preparing depthnet parameters...")
    checkpoint_path = download_bevdepth_weights()
    depthnet_state = extract_depthnet_state_dict(checkpoint_path)

    return prepare_depthnet_parameters(
        depthnet_state,
        in_channels=512,
        mid_channels=512,
        depth_channels=112,
    )


def prepare_head_parameters(device):
    """Prepare parameters for BEVDepthHead."""
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor

    logger.info("Preparing head parameters...")
    reference_model = load_reference_model()
    if reference_model is None:
        return None

    torch_head = reference_model.model.head
    torch_head.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_head,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )
    return parameters


def prepare_all_parameters_from_reference(device):
    """Load reference model once and prepare all parameters."""
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor
    from models.experimental.BevDepth.tests.test_bevdepth_backbone import (
        extract_backbone_state_dict,
        extract_neck_state_dict,
        extract_depthnet_state_dict,
        fuse_batchnorm_into_conv,
        prepare_ttnn_parameters,
    )
    from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters
    from models.experimental.BevDepth.tt.ttnn_depthnet import prepare_depthnet_parameters as prep_depthnet

    logger.info("Loading reference model and preparing all parameters...")

    reference_model = load_reference_model()
    if reference_model is None:
        return None, None

    checkpoint_path = download_bevdepth_weights()

    backbone_state = extract_backbone_state_dict(checkpoint_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)
    backbone_params = prepare_ttnn_parameters(backbone_state)

    neck_state = extract_neck_state_dict(checkpoint_path)
    neck_params = prepare_secondfpn_parameters(
        neck_state,
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2],
    )

    depthnet_state = extract_depthnet_state_dict(checkpoint_path)
    depthnet_params = prep_depthnet(
        depthnet_state,
        in_channels=512,
        mid_channels=512,
        depth_channels=112,
    )

    torch_head = reference_model.model.head
    torch_head.eval()
    head_params = preprocess_model_parameters(
        initialize_model=lambda: torch_head,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )

    params = {
        "backbone": backbone_params,
        "neck": neck_params,
        "depthnet": depthnet_params,
        "head": head_params,
    }

    return params, reference_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bevdepth_e2e(device):
    """
    Full BEVDepth E2E test: backbone → head with single weight loading.
    Reports only the final output PCC (TTNN vs Reference).
    """
    from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
    from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations

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
        ref_output = torch_model(torch_input_imgs, mats_dict)

    logger.info("Running TTNN model...")
    ttnn_bev_feature = ttnn_backbone(torch_input_imgs, mats_dict, is_return_depth=False)

    ttnn_bev_input = ttnn.from_torch(
        ttnn_bev_feature.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_bev_input = ttnn.to_device(ttnn_bev_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tracy.signpost("TTNN Head Forward Start")
    ttnn_output = ttnn_head(ttnn_bev_input, device=device)
    tracy.signpost("TTNN Head Forward End")

    # Check all 36 outputs (6 task groups × 6 output types)
    # Reference structure: ref_output[task_group][batch_idx][key] -> tensor
    # TTNN structure: ttnn_output[task_group][key] -> (tensor, shape)
    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    num_task_groups = len(ref_output)

    logger.info(
        f"Checking {num_task_groups} task groups × {len(output_keys)} outputs = {num_task_groups * len(output_keys)} total outputs"
    )

    all_pcc_values = []
    failed_outputs = []

    logger.info("\n" + "=" * 60)
    logger.info("BEVDepth E2E PCC Results - Per Head Task Output")
    logger.info("=" * 60)

    for task_idx in range(num_task_groups):
        logger.info(f"\nHead {task_idx}:")
        for key in output_keys:
            # Reference: ref_output[task_idx] is a list, [0] gets first batch
            ref_tensor = ref_output[task_idx][0][key]
            ttnn_tensor, _ = ttnn_output[task_idx][key]
            ttnn_tensor_torch = ttnn.to_torch(ttnn_tensor, device=device)

            # Reshape from NHWC to NCHW for comparison
            # Use the same logic as test_bevdepth_head.py for consistency
            # Expected shape is (N, C, H, W), TTNN output is (N, H, W, C)
            expected_shape = ref_tensor.shape
            ttnn_tensor_torch = torch.reshape(
                ttnn_tensor_torch,
                (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
            )
            ttnn_tensor_torch = torch.permute(ttnn_tensor_torch, (0, 3, 1, 2))

            # Ensure same dtype for comparison (use float32 for highest precision)
            ref_tensor = ref_tensor.float()
            ttnn_tensor_torch = ttnn_tensor_torch.float()

            pcc_result = comp_pcc(ref_tensor, ttnn_tensor_torch)
            pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
            all_pcc_values.append(pcc_value)

            pcc_threshold = 0.97
            logger.info(f"  Head {task_idx}, key '{key}': PCC = {pcc_value:.10f}")

            # Debug info for failing cases
            if pcc_value < pcc_threshold:
                diff = ref_tensor - ttnn_tensor_torch
                abs_diff = torch.abs(diff)
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                ref_range = ref_tensor.max().item() - ref_tensor.min().item()
                logger.warning(
                    f"  DEBUG task{task_idx}.{key}: max_abs_diff={max_diff:.6f}, "
                    f"mean_abs_diff={mean_diff:.6f}, ref_range={ref_range:.6f}, "
                    f"ref_shape={ref_tensor.shape}, tt_shape={ttnn_tensor_torch.shape}"
                )
                failed_outputs.append(f"task{task_idx}.{key}: {pcc_value:.6f}")

    if failed_outputs:
        logger.warning(f"\nOutputs below threshold ({len(failed_outputs)}/{len(all_pcc_values)}):")
        for fail in failed_outputs:
            logger.warning(f"  {fail}")

    assert (
        len(failed_outputs) == 0
    ), f"E2E PCC check failed: {len(failed_outputs)}/{len(all_pcc_values)} outputs below threshold."
