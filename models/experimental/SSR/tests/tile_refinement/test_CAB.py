# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.tt.tile_refinement import TTCAB
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.SSR.reference.SSR.model.tile_refinement import ChannelAttention
from models.experimental.SSR.tests.tile_refinement.test_channel_attention import create_channel_attention_preprocessor


class CAB(nn.Module):
    """Reference PyTorch CAB implementation"""

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


def create_cab_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Extract the sequential layers from CAB
        cab_layers = list(torch_model.cab.children())
        conv1 = cab_layers[0]  # First Conv2d layer
        conv2 = cab_layers[2]  # Second Conv2d layer (after GELU)
        channel_attention = cab_layers[3]  # ChannelAttention module

        # Preprocess first convolution (3x3)
        params["conv1"] = {
            "weight": ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": ttnn.from_torch(conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        }

        # Preprocess second convolution (3x3)
        params["conv2"] = {
            "weight": ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": ttnn.from_torch(conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        }

        # Preprocess channel attention using existing preprocessor
        channel_attention_preprocessor = create_channel_attention_preprocessor(device)
        params["channel_attention"] = channel_attention_preprocessor(
            channel_attention, "channel_attention", ttnn_module_args
        )

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_feat, height, width, compress_ratio, squeeze_factor",
    [
        (1, 180, 64, 64, 3, 30),  # SSR config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_cab_block(device, batch_size, num_feat, height, width, compress_ratio, squeeze_factor):
    torch.manual_seed(0)

    # Create reference model
    ref_model = CAB(num_feat=num_feat, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
    ref_model.eval()

    # Create input tensor
    input_tensor = torch.randn(batch_size, num_feat, height, width)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_cab_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor),
    )

    tt_model = TTCAB(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # NCHW -> NHWC
    )

    # TTNN forward pass
    tt_output = tt_model(tt_input)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)
    tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Compare outputs
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.97)

    logger.info(f"Batch: {batch_size}, Features: {num_feat}, Size: {height}x{width}")
    logger.info(f"Compress ratio: {compress_ratio}, Squeeze factor: {squeeze_factor}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("CAB Block Passed!")
    else:
        logger.warning("CAB Block Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Verify output shapes match
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"
