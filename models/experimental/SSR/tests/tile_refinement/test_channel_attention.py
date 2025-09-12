# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.tt.tile_refinement import TTChannelAttention
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.SSR.reference.SSR.model.tile_refinement import ChannelAttention


def create_channel_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Extract the sequential layers
        layers = list(torch_model.attention.children())
        conv1 = layers[1]  # First Conv2d layer
        conv2 = layers[3]  # Second Conv2d layer

        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

        # Preprocess first convolution
        params["conv1"] = {
            "weight": ttnn.prepare_conv_weights(
                weight_tensor=ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                weights_format="OIHW",
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=True,
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
            "bias": ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(
                    conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  # Reshape to 4D: [1, 1, 1, out_channels]
                ),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
        }

        # Preprocess second convolution
        params["conv2"] = {
            "weight": ttnn.prepare_conv_weights(
                weight_tensor=ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                weights_format="OIHW",
                in_channels=conv2.in_channels,
                out_channels=conv2.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=True,
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
            "bias": ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(
                    conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  # Reshape to 4D: [1, 1, 1, out_channels]
                ),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                in_channels=conv2.in_channels,
                out_channels=conv2.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
        }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_feat, height, width, squeeze_factor",
    [
        (1, 180, 64, 64, 30),  # SSR config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_channel_attention(device, batch_size, num_feat, height, width, squeeze_factor):
    torch.manual_seed(0)

    # Create reference model
    ref_model = ChannelAttention(num_feat=num_feat, squeeze_factor=squeeze_factor)
    ref_model.eval()

    # Create input tensor
    input_tensor = torch.randn(batch_size, num_feat, height, width)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    # Create TTNN model
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_channel_attention_preprocessor(device),
        device=device,
    )

    memory_config = ttnn.L1_MEMORY_CONFIG
    tt_model = TTChannelAttention(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        squeeze_factor=squeeze_factor,
        memory_config=memory_config,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=memory_config,
    )

    # TTNN forward pass
    tt_output = tt_model(tt_input)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)
    tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.98)
    logger.info(f"pcc: {pcc_message}")

    if does_pass:
        logger.info("ChannelAttention Passed!")
    else:
        logger.warning("ChannelAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
