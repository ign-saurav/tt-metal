# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE
from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.pointpillars.tt.utils import (
    prepare_split_conv_transpose2d_weights_bias,
    split_conv_transpose2d_and_run,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": SD_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "in_channels, input_height, input_width, out_channels, output_height, output_width, conv_in_channel_split_factor, conv_out_channel_split_factor, kernel_size",
    [
        (256, 62, 54, 128, 248, 216, 4, 2, 4),
        (128, 124, 108, 128, 248, 216, 2, 2, 2),
    ],
)
def test_split_conv(
    device,
    in_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    kernel_size,
):
    torch_input = torch.randn([1, in_channels, input_height, input_width])
    torch_weights = torch.randn([in_channels, out_channels, kernel_size, kernel_size])
    torch_biases = torch.randn([out_channels])

    torch_output = torch.nn.functional.conv_transpose2d(
        torch_input, torch_weights, bias=torch_biases, stride=(kernel_size, kernel_size), padding=(0, 0), groups=1
    )

    conv_weights, conv_bias = prepare_split_conv_transpose2d_weights_bias(
        in_channels,
        out_channels,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        torch_weights,
        torch_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        enable_act_double_buffer=False,
        reshard_if_not_optimal=True,
        activation=None,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    ttnn_output = split_conv_transpose2d_and_run(
        ttnn_input,
        conv_weights,
        conv_bias,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        compute_config,
        conv_config,
        ttnn.bfloat16,
        stride=(kernel_size, kernel_size),
        padding=0,
        output_padding=0,
        kernel_size=(kernel_size, kernel_size),
    )

    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
    ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    passing, pcc = comp_pcc(torch_output, ttnn_output, 0.97)
    logger.info(f"Neck PCC: {pcc}")
    assert passing, f"Neck PCC check failed: {pcc}"
