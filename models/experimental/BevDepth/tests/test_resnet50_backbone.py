# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_resnet50_backbone import ResNet50_BEVDepth


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


def load_reference_backbone():
    """Load reference ResNet50 backbone from torchvision"""
    from torchvision.models import resnet50

    model = resnet50(pretrained=False)
    return model


def extract_backbone_state_dict(checkpoint_path):
    """Extract backbone weights from BEVDepth checkpoint"""
    import gc

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        del checkpoint
        gc.collect()
    else:
        state_dict = checkpoint

    # Extract only img_backbone weights
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone.img_backbone."):
            new_key = key.replace("model.backbone.img_backbone.", "")
            backbone_state[new_key] = value

    # Free the full state_dict to save memory
    del state_dict
    gc.collect()

    logger.info(f"Extracted {len(backbone_state)} backbone parameters")

    return backbone_state


def fuse_conv_bn_weights(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse BatchNorm parameters into conv weights for inference.
    """
    # Calculate scale factor from BN
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std

    # Fuse into conv weight: multiply each output channel by its scale
    # conv_weight shape: (out_channels, in_channels, kH, kW)
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

    # Fuse into bias
    fused_bias = bn_bias - (bn_weight * bn_mean / std)

    return fused_weight, fused_bias


def fuse_batchnorm_into_conv(state_dict):
    """Fuse all BatchNorm layers into their corresponding conv layers"""
    fused_state = {}

    # Fuse bn1 into conv1
    if "conv1.weight" in state_dict and "bn1.weight" in state_dict:
        fused_weight, fused_bias = fuse_conv_bn_weights(
            state_dict["conv1.weight"],
            state_dict["bn1.weight"],
            state_dict["bn1.bias"],
            state_dict["bn1.running_mean"],
            state_dict["bn1.running_var"],
        )
        fused_state["conv1.weight"] = fused_weight
        fused_state["conv1.bias"] = fused_bias

    # Fuse BN in each bottleneck block
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        block_idx = 0
        while True:
            block_prefix = f"{layer_name}.{block_idx}."

            # Check if this block exists
            if f"{block_prefix}conv1.weight" not in state_dict:
                break

            # Fuse bn1, bn2, bn3 for each conv in the block
            for conv_idx in range(1, 4):
                conv_key = f"{block_prefix}conv{conv_idx}.weight"
                bn_key = f"{block_prefix}bn{conv_idx}.weight"

                if conv_key in state_dict and bn_key in state_dict:
                    fused_weight, fused_bias = fuse_conv_bn_weights(
                        state_dict[conv_key],
                        state_dict[bn_key],
                        state_dict[f"{block_prefix}bn{conv_idx}.bias"],
                        state_dict[f"{block_prefix}bn{conv_idx}.running_mean"],
                        state_dict[f"{block_prefix}bn{conv_idx}.running_var"],
                    )
                    fused_state[conv_key] = fused_weight
                    fused_state[f"{block_prefix}conv{conv_idx}.bias"] = fused_bias

            # Fuse downsample BN if exists
            if f"{block_prefix}downsample.0.weight" in state_dict:
                if f"{block_prefix}downsample.1.weight" in state_dict:
                    fused_weight, fused_bias = fuse_conv_bn_weights(
                        state_dict[f"{block_prefix}downsample.0.weight"],
                        state_dict[f"{block_prefix}downsample.1.weight"],
                        state_dict[f"{block_prefix}downsample.1.bias"],
                        state_dict[f"{block_prefix}downsample.1.running_mean"],
                        state_dict[f"{block_prefix}downsample.1.running_var"],
                    )
                    fused_state[f"{block_prefix}downsample.0.weight"] = fused_weight
                    fused_state[f"{block_prefix}downsample.0.bias"] = fused_bias

            block_idx += 1

    # Copy all other weights that don't need fusion
    for key, value in state_dict.items():
        if key not in fused_state and not key.startswith("bn") and "downsample.1" not in key:
            fused_state[key] = value

    logger.info(f"Fused BatchNorm into conv weights. Original keys: {len(state_dict)}, Fused keys: {len(fused_state)}")
    return fused_state


def prepare_ttnn_parameters(state_dict, device):
    """Keep weights as PyTorch tensors - convert during conv2d call"""

    class Parameters:
        pass

    params = Parameters()

    params.conv1 = Parameters()
    params.conv1.weight = state_dict["conv1.weight"].to(torch.bfloat16)
    params.conv1.bias = state_dict.get("conv1.bias", None)
    if params.conv1.bias is not None:
        params.conv1.bias = params.conv1.bias.to(torch.bfloat16)

    # Layers - keep as PyTorch tensors
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        layer_params = []

        block_idx = 0
        while True:
            block_prefix = f"{layer_name}.{block_idx}."
            if not any(k.startswith(block_prefix) for k in state_dict.keys()):
                break

            block_params = Parameters()

            # All weights as PyTorch tensors - load fused biases from state_dict
            block_params.conv1 = Parameters()
            block_params.conv1.weight = state_dict[f"{block_prefix}conv1.weight"].to(torch.bfloat16)
            # Load fused bias if it exists (from batch norm fusion)
            conv1_bias_key = f"{block_prefix}conv1.bias"
            if conv1_bias_key in state_dict:
                block_params.conv1.bias = state_dict[conv1_bias_key].to(torch.bfloat16)
            else:
                block_params.conv1.bias = None

            block_params.conv2 = Parameters()
            block_params.conv2.weight = state_dict[f"{block_prefix}conv2.weight"].to(torch.bfloat16)
            # Load fused bias if it exists
            conv2_bias_key = f"{block_prefix}conv2.bias"
            if conv2_bias_key in state_dict:
                block_params.conv2.bias = state_dict[conv2_bias_key].to(torch.bfloat16)
            else:
                block_params.conv2.bias = None

            block_params.conv3 = Parameters()
            block_params.conv3.weight = state_dict[f"{block_prefix}conv3.weight"].to(torch.bfloat16)
            # Load fused bias if it exists
            conv3_bias_key = f"{block_prefix}conv3.bias"
            if conv3_bias_key in state_dict:
                block_params.conv3.bias = state_dict[conv3_bias_key].to(torch.bfloat16)
            else:
                block_params.conv3.bias = None

            if f"{block_prefix}downsample.0.weight" in state_dict:
                block_params.downsample = [Parameters()]
                block_params.downsample[0].weight = state_dict[f"{block_prefix}downsample.0.weight"].to(torch.bfloat16)
                # Load fused bias if it exists
                downsample_bias_key = f"{block_prefix}downsample.0.bias"
                if downsample_bias_key in state_dict:
                    block_params.downsample[0].bias = state_dict[downsample_bias_key].to(torch.bfloat16)
                else:
                    block_params.downsample[0].bias = None

            layer_params.append(block_params)
            block_idx += 1

        setattr(params, layer_name, layer_params)

    return params


def replace_bn_with_identity(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, torch.nn.Identity())
        else:
            replace_bn_with_identity(child)


def enable_conv_bias(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Conv2d) and child.bias is None:
            # Replace with conv that has bias
            new_conv = torch.nn.Conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                bias=True,
            )
            # Copy weights
            new_conv.weight.data = child.weight.data.clone()
            setattr(module, name, new_conv)
        else:
            enable_conv_bias(child)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(256, 640)])
def test_resnet50_bevdepth_pcc(device, batch_size, height, width):
    """Test TTNN ResNet50 against BEVDepth reference model"""

    # Download and load weights
    weights_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(weights_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)

    # Load reference model
    reference_model = load_reference_backbone()

    # Enable bias for conv1 and all conv layers since we're loading fused weights
    # ResNet50 from torchvision has bias=False by default, but fused weights include bias
    if reference_model.conv1.bias is None:
        # Replace conv1 with a version that has bias
        reference_model.conv1 = torch.nn.Conv2d(
            reference_model.conv1.in_channels,
            reference_model.conv1.out_channels,
            reference_model.conv1.kernel_size,
            reference_model.conv1.stride,
            reference_model.conv1.padding,
            reference_model.conv1.dilation,
            reference_model.conv1.groups,
            bias=True,  # Enable bias for fused weights
        )

    # Enable bias for all conv layers in bottleneck blocks
    # def enable_conv_bias(module):
    #     for name, child in list(module.named_children()):
    #         if isinstance(child, torch.nn.Conv2d) and child.bias is None:
    #             # Replace with conv that has bias
    #             new_conv = torch.nn.Conv2d(
    #                 child.in_channels,
    #                 child.out_channels,
    #                 child.kernel_size,
    #                 child.stride,
    #                 child.padding,
    #                 child.dilation,
    #                 child.groups,
    #                 bias=True,
    #             )
    #             # Copy weights
    #             new_conv.weight.data = child.weight.data.clone()
    #             setattr(module, name, new_conv)
    #         else:
    #             enable_conv_bias(child)

    enable_conv_bias(reference_model)

    # Now load the fused weights (including biases)
    reference_model.load_state_dict(backbone_state, strict=False)

    # # Replace batch norm with identity since weights are fused
    # def replace_bn_with_identity(module):
    #     for name, child in list(module.named_children()):
    #         if isinstance(child, torch.nn.BatchNorm2d):
    #             setattr(module, name, torch.nn.Identity())
    #         else:
    #             replace_bn_with_identity(child)

    replace_bn_with_identity(reference_model)
    reference_model.eval()

    # Prepare TTNN model
    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    }

    ttnn_params = prepare_ttnn_parameters(backbone_state, device)

    ttnn_model = ResNet50_BEVDepth(
        device=device,
        parameters=ttnn_params,
        batch_size=batch_size,
        model_config=model_config,
        return_intermediate=True,
        return_block_outputs=True,  # Return block-level outputs for debugging
    )

    # Create input - TTNN conv2d expects (B, H, W, C) format
    torch_input = torch.randn(batch_size, 3, height, width)
    # Reshape to (B, H, W, C) for TTNN
    torch_input_reshaped = torch_input.permute(0, 2, 3, 1).contiguous()

    # Use DRAM for input to avoid L1 memory exhaustion
    ttnn_input = ttnn.from_torch(
        torch_input_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # Start with ROW_MAJOR for host->device transfer
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Use DRAM instead of L1 to avoid memory issues
    )
    # Convert to TILE_LAYOUT after moving to device
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    # Reference forward (batch norm is replaced with Identity, so bn1 does nothing)
    with torch.no_grad():
        x = reference_model.conv1(torch_input)
        x = reference_model.bn1(x)  # Now Identity, does nothing
        x = reference_model.relu(x)
        ref_conv1_output = x.clone()  # Save conv1 output (before maxpool) for comparison
        x = reference_model.maxpool(x)
        ref_layer1_input = x.clone()  # Save input to layer1 for comparison

        # Get layer1 block outputs for detailed comparison
        ref_layer1_blocks = []
        for i, block in enumerate(reference_model.layer1):
            x = block(x)
            ref_layer1_blocks.append(x.clone())
        ref_layer1 = x

        ref_layer2 = reference_model.layer2(ref_layer1)
        ref_layer3 = reference_model.layer3(ref_layer2)
        ref_layer4 = reference_model.layer4(ref_layer3)

    # TTNN forward
    ttnn_features = ttnn_model(ttnn_input, input_height=height, input_width=width)

    # Compare outputs
    pcc_results = {}

    layers = {
        "layer1": ref_layer1,
        "layer2": ref_layer2,
        "layer3": ref_layer3,
        "layer4": ref_layer4,
    }

    for layer_name, ref_output in layers.items():
        ttnn_output = ttnn.to_torch(ttnn_features[layer_name])

        # TTNN output is (B, H, W, C), reference is (B, C, H, W)
        # Permute TTNN output to match reference format
        ttnn_output = ttnn_output.permute(0, 3, 1, 2).contiguous()

        pcc_result = comp_pcc(ref_output, ttnn_output)
        # comp_pcc returns (bool, float), extract the float value
        pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
        pcc_results[layer_name] = pcc_value

        logger.info(f"{layer_name}: PCC = {pcc_value:.6f}")

    # Assert PCC thresholds
    for layer_name, pcc_value in pcc_results.items():
        assert pcc_value > 0.99, f"{layer_name} PCC {pcc_value:.6f} is below threshold 0.99"

    return pcc_results


if __name__ == "__main__":
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=32768,  # Increase L1_SMALL allocation
    )

    try:
        results = test_resnet50_bevdepth_pcc(device, batch_size=1, height=256, width=640)
        print("\nPCC Results:")
        for layer, pcc in results.items():
            print(f"  {layer}: {pcc:.6f}")
    finally:
        ttnn.close_device(device)
