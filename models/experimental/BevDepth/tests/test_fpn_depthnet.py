# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import pytest
from loguru import logger
from models.common.utility_functions import comp_pcc


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


def load_reference_fpn():
    """Load reference SECONDFPN from BEVDepth reference"""
    try:
        from models.experimental.BevDepth.reference.bevdepth.layers.necks.second_fpn import SECONDFPN

        neck = SECONDFPN(
            in_channels=[256, 512, 128],
            out_channels=[128, 128, 128],  # All deblocks output 128 channels
            upsample_strides=[4, 2, 1],
            # upsample_strides=[1, 2, 4],
        )
        return neck
    except ImportError as e:
        logger.warning(f"BEVDepth reference not available: {e}")
        return None


def load_reference_depthnet(depth_channels=112):
    """Load reference DepthNet from BEVDepth"""
    try:
        from models.experimental.BevDepth.reference.bevdepth.layers.backbones.base_lss_fpn import DepthNet

        depth_net = DepthNet(
            in_channels=512,
            mid_channels=512,  # Match checkpoint: mid_channels=512
            context_channels=80,  # Match checkpoint: context_channels=80 (from output_channels)
            depth_channels=depth_channels,  # Match checkpoint: depth_channels=112
        )
        return depth_net
    except ImportError as e:
        logger.warning(f"BEVDepth reference not available: {e}")
        return None


def extract_fpn_depthnet_state_dict(checkpoint_path):
    """Extract FPN and DepthNet weights from BEVDepth checkpoint"""
    import gc

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Debug: Check checkpoint structure
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        del checkpoint
        gc.collect()
    else:
        state_dict = checkpoint

    # Debug: Show sample keys
    all_keys = list(state_dict.keys())
    logger.info(f"Total keys in state_dict: {len(all_keys)}")
    logger.info(f"First 20 keys: {all_keys[:20]}")

    # Find keys with 'neck' or 'depth'
    neck_keys = [k for k in all_keys if "neck" in k.lower()]
    depth_keys = [k for k in all_keys if "depth" in k.lower()]
    backbone_keys = [k for k in all_keys if "backbone" in k.lower()]

    logger.info(f"Keys with 'neck': {len(neck_keys)} - Sample: {neck_keys[:5]}")
    logger.info(f"Keys with 'depth': {len(depth_keys)} - Sample: {depth_keys[:5]}")
    logger.info(f"Keys with 'backbone': {len(backbone_keys)} - Sample: {backbone_keys[:5]}")

    # Extract with multiple possible prefixes
    fpn_depthnet_state = {}

    # Try different prefix patterns
    patterns = [
        ("model.backbone.img_neck.", "img_neck"),
        ("model.backbone.depth_net.", "depth_net"),
        ("img_backbone.img_neck.", "img_neck"),
        ("img_backbone.depth_net.", "depth_net"),
        ("backbone.img_neck.", "img_neck"),
        ("backbone.depth_net.", "depth_net"),
        ("img_neck.", "img_neck"),
        ("depth_net.", "depth_net"),
    ]

    for pattern, category in patterns:
        for key, value in state_dict.items():
            if pattern in key:
                fpn_depthnet_state[key] = value
                logger.debug(f"Matched {key} with pattern {pattern}")

    del state_dict
    gc.collect()

    logger.info(f"Extracted {len(fpn_depthnet_state)} FPN/DepthNet parameters")

    if len(fpn_depthnet_state) == 0:
        logger.error("No FPN/DepthNet parameters found!")
        logger.info("This might not be a BEVDepth checkpoint or keys are different")
        # Return the full state dict for inspection
        checkpoint_reloaded = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint_reloaded:
            return checkpoint_reloaded["state_dict"]
        return checkpoint_reloaded

    return fpn_depthnet_state


@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(64, 160)])
def test_secondfpn_pcc(device, batch_size, height, width):
    """Test TTNN SECONDFPN against reference"""
    from models.experimental.BevDepth.tt.ttnn_secondfpn import SECONDFPN_TTNN, prepare_secondfpn_parameters

    # Match reference model config from base_exp.py - uses all 4 ResNet layers
    in_channels = [256, 512, 1024, 2048]
    out_channels = [128, 128, 128, 128]  # All deblocks output 128 channels
    upsample_strides = [0.25, 0.5, 1, 2]  # Downsample for layer1/2, keep for layer3, upsample for layer4

    # Create synthetic inputs matching ResNet50 layer outputs
    # With downsample_factor=16, feature map is input_size/16
    # For 256x704 input -> 16x44 feature map
    # But upsample_strides control the final output size
    target_h, target_w = 16, 44  # Feature map resolution at layer3 (stride 1)
    torch_layer1 = torch.randn(batch_size, 256, target_h * 4, target_w * 4)  # 64x176 (stride 4 in backbone)
    torch_layer2 = torch.randn(batch_size, 512, target_h * 2, target_w * 2)  # 32x88 (stride 8 in backbone)
    torch_layer3 = torch.randn(batch_size, 1024, target_h, target_w)  # 16x44 (stride 16 in backbone)
    torch_layer4 = torch.randn(batch_size, 2048, target_h // 2, target_w // 2)  # 8x22 (stride 32 in backbone)

    # Load reference model with correct config matching checkpoint
    from models.experimental.BevDepth.reference.bevdepth.layers.necks.second_fpn import SECONDFPN

    reference_fpn = SECONDFPN(
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2],
        use_conv_for_no_stride=False,
    )

    # Download and load weights
    weights_path = download_bevdepth_weights()
    fpn_state = extract_fpn_depthnet_state_dict(weights_path)

    filtered_state = {}
    for k, v in fpn_state.items():
        if "img_neck" in k:
            new_key = k.replace("model.backbone.img_neck.", "")
            filtered_state[new_key] = v

    # Load weights into reference
    reference_fpn.load_state_dict(filtered_state, strict=False)

    # Fuse BatchNorm into conv layers for fair comparison with TTNN
    # TTNN fuses BN, so we should too for the reference
    from models.experimental.BevDepth.tests.test_resnet50_backbone_pcc import fuse_conv_bn_weights

    # Fuse BN for each deblock (4 deblocks total)
    for i in range(4):
        deblock = reference_fpn.deblocks[i]
        conv_layer = deblock[0]
        bn_layer = deblock[1]

        if hasattr(bn_layer, "weight") and hasattr(bn_layer, "running_mean"):
            is_transposed = isinstance(conv_layer, torch.nn.ConvTranspose2d)

            conv_weight = conv_layer.weight.data
            bn_channels = bn_layer.weight.shape[0]

            if is_transposed:
                # ConvTranspose2d: weight is [in_channels, out_channels, ...]
                conv_in_channels, conv_out_channels = conv_weight.shape[0], conv_weight.shape[1]

                if conv_out_channels != bn_channels:
                    if conv_out_channels > bn_channels:
                        logger.info(
                            f"Reference deblock {i}: truncating ConvTranspose2d weight output channels from {conv_out_channels} to {bn_channels} to match BN"
                        )
                        # Truncate output channels (second dimension)
                        conv_weight = conv_weight[:, :bn_channels, :, :].clone()
                    else:
                        raise ValueError(
                            f"Reference deblock {i}: ConvTranspose2d has {conv_out_channels} output channels but BN has {bn_channels} channels"
                        )

                conv_weight_for_fusion = conv_weight.permute(1, 0, 2, 3).contiguous()

                # Fuse BN into conv (weight is now in Conv2d format: [out_channels, in_channels, ...])
                fused_weight, fused_bias = fuse_conv_bn_weights(
                    conv_weight_for_fusion,
                    bn_layer.weight.data,
                    bn_layer.bias.data,
                    bn_layer.running_mean,
                    bn_layer.running_var,
                    eps=bn_layer.eps,
                )

                # Transpose back to ConvTranspose2d format [in_channels, out_channels, ...]
                fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()
            else:
                # Conv2d: weight is [out_channels, in_channels, ...]
                conv_out_channels = conv_weight.shape[0]

                if conv_out_channels != bn_channels:
                    if conv_out_channels > bn_channels:
                        logger.info(
                            f"Reference deblock {i}: truncating Conv2d weight output channels from {conv_out_channels} to {bn_channels} to match BN"
                        )
                        # Truncate output channels (first dimension)
                        conv_weight = conv_weight[:bn_channels, :, :, :].clone()
                    else:
                        raise ValueError(
                            f"Reference deblock {i}: Conv2d has {conv_out_channels} output channels but BN has {bn_channels} channels"
                        )

                # Fuse BN into conv (already in correct format)
                fused_weight, fused_bias = fuse_conv_bn_weights(
                    conv_weight,
                    bn_layer.weight.data,
                    bn_layer.bias.data,
                    bn_layer.running_mean,
                    bn_layer.running_var,
                    eps=bn_layer.eps,
                )
            # Update conv layer
            conv_layer.weight.data = fused_weight
            if conv_layer.bias is None:
                conv_layer.bias = torch.nn.Parameter(fused_bias)
            else:
                conv_layer.bias.data = fused_bias

            # Replace BN with Identity
            deblock[1] = torch.nn.Identity()

    reference_fpn.eval()

    # Reference forward with float32 inputs
    with torch.no_grad():
        ref_outputs = reference_fpn([torch_layer1, torch_layer2, torch_layer3, torch_layer4])

    logger.info(f"Reference output shape: {ref_outputs[0].shape}")

    # Prepare TTNN parameters
    fpn_params = prepare_secondfpn_parameters(
        fpn_state,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides,
    )

    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    }

    ttnn_fpn = SECONDFPN_TTNN(
        device=device,
        parameters=fpn_params,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides,
        model_config=model_config,
        use_torch_conv_transpose=False,  # Set to False to test TTNN conv_transpose2d
    )

    # Convert inputs to TTNN format (B, H, W, C) - all 4 layers
    ttnn_inputs = []
    for torch_tensor in [torch_layer1, torch_layer2, torch_layer3, torch_layer4]:
        torch_tensor_hwc = torch_tensor.permute(0, 2, 3, 1).contiguous()
        ttnn_tensor = ttnn.from_torch(
            torch_tensor_hwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.TILE_LAYOUT)
        ttnn_inputs.append(ttnn_tensor)

    # TTNN forward
    ttnn_outputs = ttnn_fpn(ttnn_inputs, batch_size=batch_size)

    # Compare outputs
    ttnn_out_torch = ttnn.to_torch(ttnn_outputs[0])
    ttnn_out_torch = ttnn_out_torch.permute(0, 3, 1, 2).contiguous()

    pcc_result = comp_pcc(ref_outputs[0], ttnn_out_torch)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    logger.info(f"SECONDFPN: PCC = {pcc_value:.6f}")
    logger.info(f"  Reference shape: {ref_outputs[0].shape}")
    logger.info(f"  TTNN shape: {ttnn_out_torch.shape}")

    assert pcc_value > 0.90, f"SECONDFPN PCC {pcc_value:.6f} is below threshold 0.90"

    return pcc_value


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(64, 160)])
@pytest.mark.parametrize("depth_channels", [112])  # Match checkpoint: depth_channels=112
def test_depthnet_pcc(device, batch_size, height, width, depth_channels):
    """Test TTNN DepthNet against reference"""
    from models.experimental.BevDepth.tt.ttnn_depthnet import DepthNet_TTNN, prepare_depthnet_parameters

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
    depthnet_state = extract_fpn_depthnet_state_dict(weights_path)

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
    # Shapes: (B, num_sweeps, num_cameras, 4, 4) for most, (B, 4, 4) for bda_mat
    num_sweeps = 1
    num_cameras = 1  # Single camera for testing

    # Capture intermediate outputs and inputs from reference model using hooks
    ref_intermediate_outputs = {}
    ref_intermediate_inputs = {}

    def make_hook(name, capture_input=False):
        def hook(module, input, output):
            # Store output, converting to CPU and detaching
            if isinstance(output, torch.Tensor):
                ref_intermediate_outputs[name] = output.detach().cpu().clone()
            else:
                ref_intermediate_outputs[name] = output

            # Also capture input if requested (for block1, block2, block3)
            if capture_input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    ref_intermediate_inputs[name] = input[0].detach().cpu().clone()

        return hook

    # Register hooks for intermediate layers
    hooks = []

    # Hook into reduce_conv (Sequential: Conv2d -> BN -> ReLU)
    if hasattr(reference_depthnet, "reduce_conv"):
        # Hook after the Sequential (after BN and ReLU)
        hooks.append(reference_depthnet.reduce_conv.register_forward_hook(make_hook("reduce_conv")))

    # Hook into context_se (SELayer)
    if hasattr(reference_depthnet, "context_se"):
        hooks.append(reference_depthnet.context_se.register_forward_hook(make_hook("context_se")))

    # Hook into context_conv (Conv2d)
    if hasattr(reference_depthnet, "context_conv"):
        hooks.append(reference_depthnet.context_conv.register_forward_hook(make_hook("context_conv")))

    # Hook into depth_se (SELayer)
    if hasattr(reference_depthnet, "depth_se"):
        hooks.append(reference_depthnet.depth_se.register_forward_hook(make_hook("depth_se")))

    # Hook into depth_conv Sequential layers
    if hasattr(reference_depthnet, "depth_conv"):
        depth_conv = reference_depthnet.depth_conv
        # Hook into BasicBlocks (indices 0, 1, 2) - capture both input and output
        for i in range(3):
            if len(depth_conv) > i:
                block = depth_conv[i]
                hooks.append(block.register_forward_hook(make_hook(f"block{i+1}", capture_input=True)))
        # Hook into ASPP (index 3) - capture both input and output
        if len(depth_conv) > 3:
            hooks.append(depth_conv[3].register_forward_hook(make_hook("aspp", capture_input=True)))
        # Hook into DCN (index 4) - capture both input and output
        if len(depth_conv) > 4:
            hooks.append(depth_conv[4].register_forward_hook(make_hook("dcn", capture_input=True)))
        # Hook into final conv (index 5) - capture both input and output
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
            # Remove hooks
            for hook in hooks:
                hook.remove()

    # Prepare TTNN parameters
    # Use mid_channels=512 to match checkpoint (from base_exp.py: depth_net_conf)
    depthnet_params = prepare_depthnet_parameters(
        depthnet_state,
        in_channels=512,
        mid_channels=512,  # Match checkpoint: mid_channels=512
        depth_channels=depth_channels,
    )

    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "ENABLE_STEP_PCC": True,  # Enable step-by-step PCC logging
        "USE_PYTORCH_FALLBACK_BASICBLOCK": False,  # Set to False to test TTNN implementations
        "USE_PYTORCH_FALLBACK_ASPP": False,  # Disable PyTorch fallback - debug TTNN ASPP implementation
        "USE_PYTORCH_FALLBACK_ASPP_DILATED_CONV": True,  # Use PyTorch fallback for dilated convs (x2, x3, x4) -
        "USE_PYTORCH_FALLBACK_ASPP_UPSAMPLE": True,  # Use PyTorch fallback for bilinear upsampling (x5) - avoids L1 OOM
        "USE_PYTORCH_FALLBACK_DCN": False,  # Disable PyTorch fallback - debug TTNN DCN implementation
        "USE_PYTORCH_FALLBACK_FINAL_CONV": False,  # Disable PyTorch fallback - debug TTNN final_depth_conv implementation
        "DEBUG_BLOCK1": False,  # Disable detailed debugging for block1 (already working)
        "DEBUG_ASPP": False,  # Disable detailed debugging for ASPP
        "TEST_BLOCK1_ONLY": False,  # Set to True to test only block1 (skip block2, block3)
    }

    ttnn_depthnet = DepthNet_TTNN(
        device=device,
        parameters=depthnet_params,
        in_channels=512,
        mid_channels=512,  # Match checkpoint: mid_channels=512
        context_channels=80,  # Match checkpoint: context_channels=80 (from output_channels)
        depth_channels=depth_channels,  # Should be 112 from parametrize
        model_config=model_config,
    )

    # Store reference intermediate outputs and inputs for step-by-step PCC computation
    ttnn_depthnet.step_pcc_ref_outputs = ref_intermediate_outputs
    ttnn_depthnet.step_pcc_ref_inputs = ref_intermediate_inputs

    # Store reference layer instances for direct use (for debugging even when fallback is disabled)
    if hasattr(reference_depthnet, "depth_conv"):
        depth_conv = reference_depthnet.depth_conv
        # Store reference BasicBlock instances (always store for debugging)
        ttnn_depthnet.ref_block1 = depth_conv[0] if len(depth_conv) > 0 else None
        ttnn_depthnet.ref_block2 = depth_conv[1] if len(depth_conv) > 1 else None
        ttnn_depthnet.ref_block3 = depth_conv[2] if len(depth_conv) > 2 else None
        logger.info("Stored reference BasicBlock instances for debugging")

        # Store reference ASPP, DCN, and final conv instances for direct use
        ttnn_depthnet.ref_aspp = depth_conv[3] if len(depth_conv) > 3 else None
        ttnn_depthnet.ref_dcn = depth_conv[4] if len(depth_conv) > 4 else None
        ttnn_depthnet.ref_final_conv = depth_conv[5] if len(depth_conv) > 5 else None
        logger.info("Stored reference ASPP, DCN, and final conv instances for debugging")

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

    # TTNN forward (with mats_dict for consistency with reference)
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
        logger.info("Testing SECONDFPN...")
        fpn_results = test_secondfpn_pcc(device, batch_size=1, height=64, width=160)
        print("\nSECONDFPN PCC Results:")
        for level, pcc in fpn_results.items():
            print(f"  {level}: {pcc:.6f}")

        logger.info("\nTesting DepthNet...")
        depthnet_pcc = test_depthnet_pcc(device, batch_size=1, height=64, width=160, depth_channels=118)
        print(f"\nDepthNet PCC: {depthnet_pcc:.6f}")

    finally:
        ttnn.close_device(device)
