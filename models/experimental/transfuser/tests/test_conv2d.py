# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    TtConv2d,
    WidthShardedStrategyConfiguration,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

DEVICE_PARAMS = {"l1_small_size": 32768}
PCC_THRESHOLD = 0.999

INPUT_SIZES = [(8, 8), (16, 8)]
CHANNEL_CONFIGS = [
    {"in_channels": 8, "out_channels": 16},
    {"in_channels": 16, "out_channels": 16},
]
BATCH_SIZES = [1, 2]
KERNEL_CONFIGS = [{"kernel_size": 3, "padding": 1}, {"kernel_size": 5, "padding": 2}]


def create_conv2d_input_tensor(configuration: Conv2dConfiguration):
    shape = (configuration.batch_size, configuration.in_channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1))
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("channel_config", CHANNEL_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("kernel_config", KERNEL_CONFIGS)
@pytest.mark.parametrize(
    "sharding_strategy",
    [
        AutoShardedStrategyConfiguration(),
        HeightShardedStrategyConfiguration(),
        WidthShardedStrategyConfiguration(),
        BlockShardedStrategyConfiguration(),
    ],
)
def test_conv2d(input_size, channel_config, batch_size, kernel_config, sharding_strategy, device):
    input_height, input_width = input_size
    in_channels, out_channels = channel_config["in_channels"], channel_config["out_channels"]
    kernel_size, padding = kernel_config["kernel_size"], kernel_config["padding"]

    configuration = Conv2dConfiguration.with_random_weights(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        sharding_strategy=sharding_strategy,
    )

    weight, bias = configuration.weight, configuration.bias
    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configuration)

    layer = TtConv2d(configuration, device)

    ttnn_output_tensor = layer(ttnn_input_tensor)
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        ttnn.to_torch(weight),
        ttnn.to_torch(bias).reshape(-1) if bias is not None else None,
        padding=configuration.padding,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", [(224, 224)])  # Standard input size
@pytest.mark.parametrize("channel_config", [(16, 32)])  # (in_channels, out_channels)
@pytest.mark.parametrize("batch_size", [1])
def test_conv2d_1x1_zero_padding_stride_1(input_size, channel_config, batch_size, device):
    """Test 1x1 conv2d with 0 padding and stride 1 using TT CNN Builder"""
    input_height, input_width = input_size
    in_channels, out_channels = channel_config

    # Create configuration for 1x1 conv with 0 padding and stride 1
    configuration = Conv2dConfiguration.with_random_weights(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=(1, 1),  # 1x1 convolution
        padding=(0, 0),  # 0 padding
        stride=(1, 1),  # stride 1
    )

    # Create input tensors
    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configuration)

    # Create and execute TT CNN layer
    layer = TtConv2d(configuration, device)
    ttnn_output_tensor = layer(ttnn_input_tensor)

    # Reference PyTorch implementation
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        ttnn.to_torch(configuration.weight),
        ttnn.to_torch(configuration.bias).reshape(-1) if configuration.bias is not None else None,
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


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_conv2d_1x1_zero_padding_stride_1_from_files(device):
    """Test 1x1 conv2d with 0 padding and stride 1 using loaded input and weights from files"""

    # Load input tensor from TTNN format
    ttnn_input_tensor = ttnn.load_tensor("input.tensorbin", device=device)

    # Load PyTorch input for reference
    torch_input_tensor = torch.load("torch_input.pt")

    # Load state dict containing weights
    state_dict = torch.load("state_dict.pt")

    # Extract conv2d weights and bias from state dict
    # Adjust the key names based on your actual state dict structure
    conv_weight = state_dict["conv.weight"]  # Replace with actual key
    conv_bias = state_dict.get("conv.bias", None)  # Replace with actual key (optional)

    # Convert weights to TTNN format
    ttnn_weight_tensor = ttnn.from_torch(conv_weight, device=device)
    ttnn_bias_tensor = ttnn.from_torch(conv_bias, device=device) if conv_bias is not None else None

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
        conv_bias,
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
