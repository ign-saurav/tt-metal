# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


class SECONDFPN_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels=[256, 512, 128],
        out_channels=[128, 128, 1024],
        upsample_strides=[4, 2, 1],
        model_config=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_levels = len(in_channels)

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.deblocks = parameters.deblocks
        logger.info(f"SECONDFPN init: {self.num_levels} levels")

    def __call__(self, x, batch_size=1):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        ups = []
        for i in range(self.num_levels):
            feat = x[i]
            height, width = feat.shape[1], feat.shape[2]
            stride = self.upsample_strides[i]

            # Transposed conv for upsampling
            out_height = height * stride
            out_width = width * stride

            # Use regular conv + upsample as fallback
            if stride > 1:
                feat = ttnn.upsample(feat, (batch_size, out_height, out_width, feat.shape[3]))

            # Conv + BN + ReLU
            feat = ttnn_conv2d(
                input_tensor=feat,
                weight_tensor=self.deblocks[i].conv_weight,
                bias_tensor=self.deblocks[i].conv_bias,
                device=self.device,
                batch_size=batch_size,
                input_height=out_height,
                input_width=out_width,
                in_channels=self.in_channels[i],
                out_channels=self.out_channels[i],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                **self.model_config,
            )

            if len(feat.shape) == 3:
                feat = ttnn.reshape(feat, (batch_size, out_height, out_width, self.out_channels[i]))

            ups.append(feat)

        # Concatenate along channel dimension
        out = ttnn.concat(ups, dim=-1)
        return [out]


def prepare_secondfpn_parameters(state_dict, in_channels=[256, 512, 128], out_channels=[128, 128, 1024]):
    class Parameters:
        pass

    params = Parameters()
    params.deblocks = []

    # Find the actual prefix used in this checkpoint
    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.img_neck.",
        "img_backbone.img_neck.",
        "backbone.img_neck.",
        "img_neck.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find img_neck prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No img_neck keys found in checkpoint")

    logger.info(f"Using SECONDFPN prefix: {prefix}")

    for i in range(len(in_channels)):
        deblock = Parameters()
        # deblocks.{i}.0 is the transposed conv or conv layer
        # deblocks.{i}.1 is the BatchNorm
        # deblocks.{i}.2 is ReLU (no params)
        deblock.conv_weight = state_dict[f"{prefix}deblocks.{i}.0.weight"].to(torch.bfloat16)
        deblock.conv_bias = state_dict.get(f"{prefix}deblocks.{i}.0.bias", None)
        if deblock.conv_bias is not None:
            deblock.conv_bias = deblock.conv_bias.to(torch.bfloat16)

        # Also load BN params if needed
        deblock.bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        deblock.bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        if deblock.bn_weight is not None:
            deblock.bn_weight = deblock.bn_weight.to(torch.bfloat16)
        if deblock.bn_bias is not None:
            deblock.bn_bias = deblock.bn_bias.to(torch.bfloat16)

        params.deblocks.append(deblock)

    logger.info(f"Prepared SECONDFPN parameters for {len(in_channels)} levels")
    return params
