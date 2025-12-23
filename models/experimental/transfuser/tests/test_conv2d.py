# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

DEVICE_PARAMS = {"l1_small_size": 32768}
PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_conv2d_1x1_zero_padding_stride_1_from_files(device):
    """Test 1x1 conv2d with 0 padding and stride 1 using loaded input and weights from files"""

    # Load input tensor from TTNN format
    ttnn_input_tensor = ttnn.load_tensor("se_fc2_input.tensorbin", device=device)

    # Load PyTorch input for reference
    torch_input_tensor = torch.load("se_fc2_torch_input.pt")

    # Load state dict containing weights
    state_dict = torch.load("se_fc2_state_dict.pt")
    # Get dimensions from loaded tensors
    batch_size, in_channels, input_height, input_width = torch_input_tensor.shape
    out_channels = state_dict["weight"].shape[0]

    # Create a PyTorch Conv2d model and load state dict properly
    torch_model = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, 1),
        padding=(0, 0),
        stride=(1, 1),
        bias=state_dict.get("bias") is not None,
    )

    # Load state dict into model
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.eval()

    # Extract weights and bias from the loaded model
    conv_weight = torch_model.weight.data
    conv_bias = torch_model.bias.data if torch_model.bias is not None else None

    # Convert weights to TTNN format
    ttnn_weight_tensor = ttnn.from_torch(conv_weight, device=device)

    # Reshape bias to (1, 1, 1, out_channels) before converting to TTNN
    if conv_bias is not None:
        conv_bias_reshaped = conv_bias.reshape((1, 1, 1, -1))
        ttnn_bias_tensor = ttnn.from_torch(conv_bias_reshaped, device=device)
    else:
        ttnn_bias_tensor = None

    # Create configuration with loaded weights
    configuration = Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=(1, 1),  # 1x1 convolution
        padding=(0, 0),  # 0 padding
        stride=(1, 1),  # stride 1
        weight=ttnn_weight_tensor,
        bias=ttnn_bias_tensor,
    )

    # Create and execute TT CNN layer
    layer = TtConv2d(configuration, device)
    ttnn_output_tensor = layer(ttnn_input_tensor)

    # Reference PyTorch implementation using the loaded model
    torch_output_tensor = torch_model(torch_input_tensor)

    # Compare results
    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    pcc, pcc_msg = assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )
    logger.info(f"PCC = {pcc_msg}. Threshold = {PCC_THRESHOLD}")
