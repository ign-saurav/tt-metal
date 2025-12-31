# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    sharding_strategy=AutoShardedStrategyConfiguration(),
    config_tensors_in_dram=False,
    enable_act_double_buffer=None,
    enable_weights_double_buffer=None,
) -> Conv2dConfiguration:
    """Create Conv2dConfiguration from parameters dict."""
    if enable_act_double_buffer is None:
        from models.tt_cnn.tt.builder import WidthShardedStrategyConfiguration

        enable_act_double_buffer = not isinstance(sharding_strategy, WidthShardedStrategyConfiguration)

    if enable_weights_double_buffer is None:
        enable_weights_double_buffer = True

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
        config_tensors_in_dram=config_tensors_in_dram,
    )
