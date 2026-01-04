# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test TTNN SECONDFPN against reference implementation"""

import torch
import ttnn
import pytest
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters
from models.experimental.BevDepth.tests.pcc.test_bevdepth_backbone import extract_neck_state_dict, fuse_conv_bn_weights
from models.experimental.BevDepth.tests.pcc.test_bevdepth_backbone import download_bevdepth_weights
from models.experimental.BevDepth.tt.ttnn_secondfpn import SECONDFPN_TTNN
from models.experimental.BevDepth.reference.bevdepth.layers.necks.second_fpn import SECONDFPN


@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height, width", [(64, 160)])
def test_secondfpn_pcc(device, batch_size, height, width):
    """Test TTNN SECONDFPN against reference"""
    in_channels = [256, 512, 1024, 2048]
    out_channels = [128, 128, 128, 128]
    upsample_strides = [0.25, 0.5, 1, 2]

    # Create synthetic inputs matching ResNet50 layer outputs
    target_h, target_w = 16, 44
    torch_layer1 = torch.randn(batch_size, 256, target_h * 4, target_w * 4)  # 64x176
    torch_layer2 = torch.randn(batch_size, 512, target_h * 2, target_w * 2)  # 32x88
    torch_layer3 = torch.randn(batch_size, 1024, target_h, target_w)  # 16x44
    torch_layer4 = torch.randn(batch_size, 2048, target_h // 2, target_w // 2)  # 8x22

    # Load reference model
    reference_fpn = SECONDFPN(
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2],
        use_conv_for_no_stride=False,
    )

    # Download and load weights
    weights_path = download_bevdepth_weights()
    fpn_state = extract_neck_state_dict(weights_path)

    filtered_state = {}
    for k, v in fpn_state.items():
        if "img_neck" in k:
            new_key = k.replace("model.backbone.img_neck.", "")
            filtered_state[new_key] = v

    reference_fpn.load_state_dict(filtered_state, strict=False)

    # Fuse BatchNorm into conv layers for fair comparison with TTNN
    for i in range(4):
        deblock = reference_fpn.deblocks[i]
        conv_layer = deblock[0]
        bn_layer = deblock[1]

        if hasattr(bn_layer, "weight") and hasattr(bn_layer, "running_mean"):
            is_transposed = isinstance(conv_layer, torch.nn.ConvTranspose2d)

            conv_weight = conv_layer.weight.data
            bn_channels = bn_layer.weight.shape[0]

            if is_transposed:
                conv_in_channels, conv_out_channels = conv_weight.shape[0], conv_weight.shape[1]

                if conv_out_channels != bn_channels:
                    if conv_out_channels > bn_channels:
                        conv_weight = conv_weight[:, :bn_channels, :, :].clone()
                    else:
                        raise ValueError(
                            f"ConvTranspose2d has {conv_out_channels} output channels but BN has {bn_channels} channels"
                        )

                conv_weight_for_fusion = conv_weight.permute(1, 0, 2, 3).contiguous()

                fused_weight, fused_bias = fuse_conv_bn_weights(
                    conv_weight_for_fusion,
                    bn_layer.weight.data,
                    bn_layer.bias.data,
                    bn_layer.running_mean,
                    bn_layer.running_var,
                    eps=bn_layer.eps,
                )

                fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()
            else:
                conv_out_channels = conv_weight.shape[0]

                if conv_out_channels != bn_channels:
                    if conv_out_channels > bn_channels:
                        conv_weight = conv_weight[:bn_channels, :, :, :].clone()
                    else:
                        raise ValueError(
                            f"Conv2d has {conv_out_channels} output channels but BN has {bn_channels} channels"
                        )

                fused_weight, fused_bias = fuse_conv_bn_weights(
                    conv_weight,
                    bn_layer.weight.data,
                    bn_layer.bias.data,
                    bn_layer.running_mean,
                    bn_layer.running_var,
                    eps=bn_layer.eps,
                )

            conv_layer.weight.data = fused_weight
            if conv_layer.bias is None:
                conv_layer.bias = torch.nn.Parameter(fused_bias)
            else:
                conv_layer.bias.data = fused_bias

            deblock[1] = torch.nn.Identity()

    reference_fpn.eval()

    # Reference forward
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
    )

    # Convert inputs to TTNN format (B, H, W, C)
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
    PCC_THRESHOLD = 0.97

    logger.info(f"SECONDFPN: PCC = {pcc_value:.6f}")

    assert pcc_value > PCC_THRESHOLD, f"SECONDFPN PCC {pcc_value:.6f} is below threshold {PCC_THRESHOLD}"

    return pcc_value


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=98304)

    try:
        logger.info("Testing SECONDFPN...")
        fpn_pcc = test_secondfpn_pcc(device, batch_size=1, height=64, width=160)
        print(f"\nSECONDFPN PCC: {fpn_pcc:.6f}")
    finally:
        ttnn.close_device(device)
