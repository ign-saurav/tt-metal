# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test TTNN DepthNet against reference implementation"""

import torch
import ttnn
import pytest
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_depthnet_parameters,
    extract_depthnet_state_dict,
)
from models.experimental.BevDepth.common import download_bevdepth_weights


def load_reference_depthnet(depth_channels=112):
    """Load reference DepthNet from BEVDepth"""
    try:
        from models.experimental.BevDepth.reference.bevdepth.layers.backbones.base_lss_fpn import DepthNet

        depth_net = DepthNet(
            in_channels=512,
            mid_channels=512,
            context_channels=80,
            depth_channels=depth_channels,
        )
        return depth_net
    except ImportError as e:
        logger.warning(f"BEVDepth reference not available: {e}")
        return None


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(64, 160)])
@pytest.mark.parametrize("depth_channels", [112])
def test_depthnet_pcc(device, batch_size, height, width, depth_channels):
    """Test TTNN DepthNet against reference"""
    from models.experimental.BevDepth.tt.ttnn_depthnet import DepthNet_TTNN

    # Create synthetic FPN output
    torch_input = torch.randn(batch_size, 512, height, width)

    # Load reference model
    reference_depthnet = load_reference_depthnet(depth_channels=depth_channels)
    if reference_depthnet is None:
        logger.warning("Skipping test - BEVDepth reference not available")
        pytest.skip("BEVDepth reference not available")
        return

    # Download and load weights
    weights_path = download_bevdepth_weights()
    depthnet_state = extract_depthnet_state_dict(weights_path)

    if len(depthnet_state) == 0:
        logger.error("No DepthNet weights found in checkpoint")
        pytest.skip("No DepthNet weights in checkpoint")
        return

    # Load weights into reference
    reference_depthnet.load_state_dict(
        {k.replace("model.backbone.depth_net.", ""): v for k, v in depthnet_state.items() if "depth_net" in k},
        strict=False,
    )
    reference_depthnet.eval()

    # DepthNet requires all camera parameters in mats_dict
    num_sweeps = 1
    num_cameras = 1

    # Capture intermediate outputs for debugging
    ref_intermediate_outputs = {}
    ref_intermediate_inputs = {}

    def make_hook(name, capture_input=False):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                ref_intermediate_outputs[name] = output.detach().cpu().clone()
            else:
                ref_intermediate_outputs[name] = output

            if capture_input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    ref_intermediate_inputs[name] = input[0].detach().cpu().clone()

        return hook

    # Register hooks for intermediate layers
    hooks = []

    if hasattr(reference_depthnet, "reduce_conv"):
        hooks.append(reference_depthnet.reduce_conv.register_forward_hook(make_hook("reduce_conv")))

    if hasattr(reference_depthnet, "context_se"):
        hooks.append(reference_depthnet.context_se.register_forward_hook(make_hook("context_se")))

    if hasattr(reference_depthnet, "context_conv"):
        hooks.append(reference_depthnet.context_conv.register_forward_hook(make_hook("context_conv")))

    if hasattr(reference_depthnet, "depth_se"):
        hooks.append(reference_depthnet.depth_se.register_forward_hook(make_hook("depth_se")))

    if hasattr(reference_depthnet, "depth_conv"):
        depth_conv = reference_depthnet.depth_conv
        for i in range(3):
            if len(depth_conv) > i:
                block = depth_conv[i]
                hooks.append(block.register_forward_hook(make_hook(f"block{i+1}", capture_input=True)))
        if len(depth_conv) > 3:
            hooks.append(depth_conv[3].register_forward_hook(make_hook("aspp", capture_input=True)))
        if len(depth_conv) > 4:
            hooks.append(depth_conv[4].register_forward_hook(make_hook("dcn", capture_input=True)))
        if len(depth_conv) > 5:
            hooks.append(depth_conv[5].register_forward_hook(make_hook("final_depth_conv", capture_input=True)))

    mats_dict = {
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
        "sensor2ego_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "bda_mat": torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    }

    with torch.no_grad():
        try:
            ref_output = reference_depthnet(torch_input, mats_dict=mats_dict)
        except Exception as e:
            logger.warning(f"Reference model failed with camera params: {e}")
            pytest.skip("Reference model requires camera parameters")
            return
        finally:
            for hook in hooks:
                hook.remove()

    # Prepare TTNN parameters
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

    ttnn_depthnet = DepthNet_TTNN(
        device=device,
        parameters=depthnet_params,
        in_channels=512,
        mid_channels=512,
        context_channels=80,
        depth_channels=depth_channels,
        model_config=model_config,
    )

    # Store reference intermediate outputs for step-by-step PCC computation
    ttnn_depthnet.step_pcc_ref_outputs = ref_intermediate_outputs
    ttnn_depthnet.step_pcc_ref_inputs = ref_intermediate_inputs

    # Store reference layer instances for debugging
    if hasattr(reference_depthnet, "depth_conv"):
        depth_conv = reference_depthnet.depth_conv
        ttnn_depthnet.ref_block1 = depth_conv[0] if len(depth_conv) > 0 else None
        ttnn_depthnet.ref_block2 = depth_conv[1] if len(depth_conv) > 1 else None
        ttnn_depthnet.ref_block3 = depth_conv[2] if len(depth_conv) > 2 else None
        ttnn_depthnet.ref_aspp = depth_conv[3] if len(depth_conv) > 3 else None
        ttnn_depthnet.ref_dcn = depth_conv[4] if len(depth_conv) > 4 else None
        ttnn_depthnet.ref_final_conv = depth_conv[5] if len(depth_conv) > 5 else None
        logger.info("Stored reference layer instances for debugging")

    # Convert input to TTNN format (B, H, W, C)
    torch_input_hwc = torch_input.permute(0, 2, 3, 1).contiguous()
    ttnn_input = ttnn.from_torch(
        torch_input_hwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    # TTNN forward
    ttnn_output = ttnn_depthnet(ttnn_input, batch_size=batch_size, mats_dict=mats_dict)

    # Compare outputs
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2).contiguous()

    # Split depth and context for comparison
    ref_depth = ref_output[:, :depth_channels, :, :]
    ttnn_depth = ttnn_output_torch[:, :depth_channels, :, :]

    pcc_result = comp_pcc(ref_depth, ttnn_depth)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    logger.info(f"DepthNet: PCC = {pcc_value:.6f}")
    logger.info(f"  Reference shape: {ref_depth.shape}")
    logger.info(f"  TTNN shape: {ttnn_depth.shape}")

    assert pcc_value > 0.99, f"DepthNet PCC {pcc_value:.6f} is below threshold 0.99"

    logger.info("DepthNet passed PCC check!")
    return pcc_value


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    try:
        logger.info("Testing DepthNet...")
        depthnet_pcc = test_depthnet_pcc(device, batch_size=1, height=64, width=160, depth_channels=112)
        print(f"\nDepthNet PCC: {depthnet_pcc:.6f}")
    finally:
        ttnn.close_device(device)
