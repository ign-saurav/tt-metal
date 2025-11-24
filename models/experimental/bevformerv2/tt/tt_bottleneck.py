# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.bevformerv2.tt.common import TtConv2D
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


class TtBottleneck:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
        *,
        model_configs: BevFormerV2ModelConfig | None = None,
        block_path: str | None = None,
    ):
        self.is_downsample = is_downsample
        self.activation_dtype = activation_dtype

        self.conv1 = TtConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            model_configs=model_configs,
            layer_path=f"{block_path}.conv1" if block_path else None,
        )
        self.conv2 = TtConv2D(
            conv_args.conv2,
            conv_pth.conv2,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=32,
            model_configs=model_configs,
            layer_path=f"{block_path}.conv2" if block_path else None,
        )
        self.conv3 = TtConv2D(
            conv_args.conv3,
            conv_pth.conv3,
            device=device,
            activation=None,
            is_blk=conv3_blk_sharded,
            model_configs=model_configs,
            layer_path=f"{block_path}.conv3" if block_path else None,
        )

        if is_downsample:
            self.downsample = TtConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation=None,
                is_blk=blk_sharded,
                activation_dtype=activation_dtype,
                model_configs=model_configs,
                layer_path=f"{block_path}.downsample" if block_path else None,
            )

    def __call__(self, x_identity):
        x, out_ht, out_wdth = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, out_ht, out_wdth = self.conv2(x)
        x, out_ht, out_wdth = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
