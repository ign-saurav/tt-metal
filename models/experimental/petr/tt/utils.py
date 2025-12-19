# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from loguru import logger

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


# Helper function to create Conv2dConfiguration from parameters
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
) -> Conv2dConfiguration:
    """
    Conv2dConfiguration from parameters dict for SqueezeExcitation.
    """

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
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
    )


def inverse_sigmoid(x, eps: float = 1e-7):
    device = x.device()

    x_torch = ttnn.to_torch(x).to(torch.float32)
    x_torch = torch.clamp(x_torch, min=eps, max=1.0 - eps)
    one_minus_x = 1.0 - x_torch

    one_minus_x = torch.clamp(one_minus_x, min=eps)
    x_torch = torch.clamp(x_torch, min=eps)
    result = torch.log(x_torch / one_minus_x)
    if torch.isnan(result).any() or torch.isinf(result).any():
        logger.warning(f"NaN/Inf in inverse_sigmoid! Clamping output")
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

    result_ttnn = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return result_ttnn


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    tmp_val = ttnn.add(ttnn.div(val, period), offset)
    tmp_val = ttnn.floor(tmp_val)
    tmp_val = ttnn.mul(tmp_val, period)

    limited_val = ttnn.sub(val, tmp_val)
    return limited_val


def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot_sine = ttnn.to_layout(rot_sine, layout=ttnn.TILE_LAYOUT)
    rot_cosine = ttnn.to_layout(rot_cosine, layout=ttnn.TILE_LAYOUT)
    rot = ttnn.atan2(rot_sine, rot_cosine)

    rot = ttnn.mul(rot, -1)
    rot = ttnn.sub(rot, np.pi / 2)

    rot = limit_period(rot, period=np.pi * 2)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = ttnn.to_layout(width, layout=ttnn.TILE_LAYOUT)
    length = ttnn.to_layout(length, layout=ttnn.TILE_LAYOUT)
    height = ttnn.to_layout(height, layout=ttnn.TILE_LAYOUT)

    width = ttnn.exp(width)
    length = ttnn.exp(length)
    height = ttnn.exp(height)
    if normalized_bboxes.shape[-1] > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        cx = ttnn.to_layout(cx, layout=ttnn.TILE_LAYOUT)
        cy = ttnn.to_layout(cy, layout=ttnn.TILE_LAYOUT)
        cz = ttnn.to_layout(cz, layout=ttnn.TILE_LAYOUT)
        vx = ttnn.to_layout(vx, layout=ttnn.TILE_LAYOUT)
        vy = ttnn.to_layout(vy, layout=ttnn.TILE_LAYOUT)
        denormalized_bboxes = ttnn.concat([cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        cx = ttnn.to_layout(cx, layout=ttnn.TILE_LAYOUT)
        cy = ttnn.to_layout(cy, layout=ttnn.TILE_LAYOUT)
        cz = ttnn.to_layout(cz, layout=ttnn.TILE_LAYOUT)
        denormalized_bboxes = torch.concat([cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes
