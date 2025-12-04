# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import gc
import os
import urllib.request
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_resnet50_backbone import ResNet50_BEVDepth


def download_bevdepth_weights():
    """Download BEVDepth pretrained weights"""
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

    return resnet50(pretrained=False)


def extract_backbone_state_dict(checkpoint_path):
    """Extract backbone weights from BEVDepth checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if "state_dict" in checkpoint:
        del checkpoint
        gc.collect()

    # Extract only img_backbone weights
    backbone_state = {
        key.replace("model.backbone.img_backbone.", ""): value
        for key, value in state_dict.items()
        if key.startswith("model.backbone.img_backbone.")
    }

    del state_dict
    gc.collect()
    logger.info(f"Extracted {len(backbone_state)} backbone parameters")
    return backbone_state


def fuse_conv_bn_weights(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm parameters into conv weights for inference."""
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
    fused_bias = bn_bias - (bn_weight * bn_mean / std)
    return fused_weight, fused_bias


def _fuse_block_conv_bn(state_dict, block_prefix, conv_idx, fused_state):
    """Helper to fuse conv and BN for a single conv layer in a block."""
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
            if f"{block_prefix}conv1.weight" not in state_dict:
                break

            # Fuse bn1, bn2, bn3 for each conv in the block
            for conv_idx in range(1, 4):
                _fuse_block_conv_bn(state_dict, block_prefix, conv_idx, fused_state)

            # Fuse downsample BN if exists
            downsample_conv_key = f"{block_prefix}downsample.0.weight"
            downsample_bn_key = f"{block_prefix}downsample.1.weight"
            if downsample_conv_key in state_dict and downsample_bn_key in state_dict:
                fused_weight, fused_bias = fuse_conv_bn_weights(
                    state_dict[downsample_conv_key],
                    state_dict[downsample_bn_key],
                    state_dict[f"{block_prefix}downsample.1.bias"],
                    state_dict[f"{block_prefix}downsample.1.running_mean"],
                    state_dict[f"{block_prefix}downsample.1.running_var"],
                )
                fused_state[downsample_conv_key] = fused_weight
                fused_state[f"{block_prefix}downsample.0.bias"] = fused_bias

            block_idx += 1

    # Copy all other weights that don't need fusion
    for key, value in state_dict.items():
        if key not in fused_state and not key.startswith("bn") and "downsample.1" not in key:
            fused_state[key] = value

    logger.info(f"Fused BatchNorm into conv weights. Original keys: {len(state_dict)}, Fused keys: {len(fused_state)}")
    return fused_state


def _create_conv_params(state_dict, prefix, conv_name=""):
    """Helper to create conv parameters from state dict."""

    class Parameters:
        pass

    params = Parameters()
    if conv_name:
        weight_key = f"{prefix}{conv_name}.weight"
        bias_key = f"{prefix}{conv_name}.bias"
    else:
        # For downsample where prefix already includes the full path
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"

    params.weight = state_dict[weight_key].to(torch.bfloat16)
    params.bias = state_dict.get(bias_key, None)
    if params.bias is not None:
        params.bias = params.bias.to(torch.bfloat16)

    return params


def prepare_ttnn_parameters(state_dict, device):
    """Prepare TTNN parameters from fused state dict"""

    class Parameters:
        pass

    params = Parameters()
    params.conv1 = _create_conv_params(state_dict, "", "conv1")

    # Process layers
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        layer_params = []
        block_idx = 0

        while True:
            block_prefix = f"{layer_name}.{block_idx}."
            if not any(k.startswith(block_prefix) for k in state_dict.keys()):
                break

            block_params = Parameters()
            block_params.conv1 = _create_conv_params(state_dict, block_prefix, "conv1")
            block_params.conv2 = _create_conv_params(state_dict, block_prefix, "conv2")
            block_params.conv3 = _create_conv_params(state_dict, block_prefix, "conv3")

            # Handle downsample
            downsample_key = f"{block_prefix}downsample.0.weight"
            if downsample_key in state_dict:
                block_params.downsample = [_create_conv_params(state_dict, f"{block_prefix}downsample.0", "")]

            layer_params.append(block_params)
            block_idx += 1

        setattr(params, layer_name, layer_params)

    return params


def replace_bn_with_identity(module):
    """Replace all BatchNorm2d layers with Identity"""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, torch.nn.Identity())
        else:
            replace_bn_with_identity(child)


def enable_conv_bias(module):
    """Enable bias for all Conv2d layers that don't have it"""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Conv2d) and child.bias is None:
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
            new_conv.weight.data = child.weight.data.clone()
            setattr(module, name, new_conv)
        else:
            enable_conv_bias(child)


def _prepare_reference_model(backbone_state):
    """Prepare reference model with fused weights"""
    reference_model = load_reference_backbone()

    # Enable bias for conv1
    if reference_model.conv1.bias is None:
        reference_model.conv1 = torch.nn.Conv2d(
            reference_model.conv1.in_channels,
            reference_model.conv1.out_channels,
            reference_model.conv1.kernel_size,
            reference_model.conv1.stride,
            reference_model.conv1.padding,
            reference_model.conv1.dilation,
            reference_model.conv1.groups,
            bias=True,
        )

    enable_conv_bias(reference_model)
    reference_model.load_state_dict(backbone_state, strict=False)
    replace_bn_with_identity(reference_model)
    reference_model.eval()
    return reference_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(256, 640)])
def test_resnet50_bevdepth_pcc(device, batch_size, height, width):
    """Test TTNN ResNet50 against BEVDepth reference model"""
    # Download and load weights
    weights_path = download_bevdepth_weights()
    backbone_state = extract_backbone_state_dict(weights_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)

    # Prepare reference model
    reference_model = _prepare_reference_model(backbone_state)

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
        return_block_outputs=False,
    )

    # Create input
    torch_input = torch.randn(batch_size, 3, height, width)
    torch_input_reshaped = torch_input.permute(0, 2, 3, 1).contiguous()

    ttnn_input = ttnn.from_torch(
        torch_input_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    # Reference forward
    with torch.no_grad():
        x = reference_model.conv1(torch_input)
        x = reference_model.bn1(x)  # Identity after replacement
        x = reference_model.relu(x)
        x = reference_model.maxpool(x)
        ref_layer1 = reference_model.layer1(x)
        ref_layer2 = reference_model.layer2(ref_layer1)
        ref_layer3 = reference_model.layer3(ref_layer2)
        ref_layer4 = reference_model.layer4(ref_layer3)

    # TTNN forward
    ttnn_features = ttnn_model(ttnn_input, input_height=height, input_width=width)

    # Compare outputs
    layers = {
        "layer1": ref_layer1,
        "layer2": ref_layer2,
        "layer3": ref_layer3,
        "layer4": ref_layer4,
    }

    pcc_results = {}
    for layer_name, ref_output in layers.items():
        ttnn_output = ttnn.to_torch(ttnn_features[layer_name])
        ttnn_output = ttnn_output.permute(0, 3, 1, 2).contiguous()

        pcc_result = comp_pcc(ref_output, ttnn_output)
        pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
        pcc_results[layer_name] = pcc_value
        logger.info(f"{layer_name}: PCC = {pcc_value:.6f}")

    # Assert PCC thresholds
    for layer_name, pcc_value in pcc_results.items():
        assert pcc_value > 0.99, f"{layer_name} PCC {pcc_value:.6f} is below threshold 0.99"

    return pcc_results


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    try:
        results = test_resnet50_bevdepth_pcc(device, batch_size=1, height=256, width=640)
        print("\nPCC Results:")
        for layer, pcc in results.items():
            print(f"  {layer}: {pcc:.6f}")
    finally:
        ttnn.close_device(device)
