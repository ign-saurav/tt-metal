# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
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
    print("Available keys in state dict:")
    for key in state_dict.keys():
        print(f"  {key}")

    conv_weight = state_dict["weight"]
    conv_bias = state_dict.get("bias", None)

    # Convert weights to TTNN format
    ttnn_weight_tensor = ttnn.from_torch(conv_weight, device=device)

    # Reshape bias to (1, 1, 1, out_channels) before converting to TTNN
    if conv_bias is not None:
        conv_bias = conv_bias.reshape((1, 1, 1, -1))
        ttnn_bias_tensor = ttnn.from_torch(conv_bias, device=device)
    else:
        ttnn_bias_tensor = None

    # Get dimensions from loaded tensors
    batch_size, in_channels, input_height, input_width = torch_input_tensor.shape
    out_channels = conv_weight.shape[0]

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

    # Reference PyTorch implementation
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        conv_weight,
        conv_bias.reshape(-1) if conv_bias is not None else None,  # Use original 1D bias for PyTorch
        padding=configuration.padding,
        stride=configuration.stride,
    )

    # Compare results
    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )
