# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from typing import Optional

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.bevformerv2.tt.utils import create_conv2d_configuration
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


@dataclass
class BottleneckOptimizer:
    conv1: dict
    conv2: dict
    conv3: dict
    downsample: dict


bottleneck_layer_optimisations = {
    "default": BottleneckOptimizer(
        conv1={},
        conv2={"act_block_h": 32},
        conv3={},
        downsample={},
    ),
    "layer1": BottleneckOptimizer(
        conv1={},
        conv2={"act_block_h": 32},
        conv3={},
        downsample={},
    ),
    "layer2": BottleneckOptimizer(
        conv1={},
        conv2={"act_block_h": 32},
        conv3={},
        downsample={
            "is_blk": True,
            "activation_dtype": ttnn.bfloat8_b,
        },
    ),
    "layer3": BottleneckOptimizer(
        conv1={},
        conv2={"act_block_h": 32},
        conv3={},
        downsample={
            "is_blk": True,
            "activation_dtype": ttnn.bfloat8_b,
        },
    ),
    "layer4": BottleneckOptimizer(
        conv1={},
        conv2={"is_blk": True, "activation_dtype": ttnn.bfloat8_b},
        conv3={"is_blk": True},
        downsample={
            "is_blk": True,
            "activation_dtype": ttnn.bfloat8_b,
        },
    ),
}


def get_bottleneck_optimisation(layer_name: Optional[str] = None):
    """Get bottleneck optimization configuration based on layer name."""
    if not layer_name:
        return bottleneck_layer_optimisations["default"]

    for key in ["layer4", "layer3", "layer2", "layer1"]:
        if key in layer_name:
            return bottleneck_layer_optimisations[key]

    return bottleneck_layer_optimisations["default"]


class TtBottleneck:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        *,
        model_configs: BevFormerV2ModelConfig | None = None,
        block_path: str | None = None,
        layer_optimisations: Optional[BottleneckOptimizer] = None,
    ):
        self.is_downsample = is_downsample

        # Get layer optimizations
        if layer_optimisations is None:
            layer_optimisations = get_bottleneck_optimisation(block_path)

        self.layer_optimisations = layer_optimisations

        # Determine activation dtype from optimizations or defaults
        activation_dtype = layer_optimisations.downsample.get("activation_dtype", ttnn.bfloat16)
        if activation_dtype == ttnn.bfloat8_b:
            self.activation_dtype = ttnn.bfloat8_b
        else:
            self.activation_dtype = ttnn.bfloat16

        # conv1
        conv1_opts = layer_optimisations.conv1.copy()
        conv1_config = create_conv2d_configuration(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            model_configs=model_configs,
            layer_path=f"{block_path}.conv1" if block_path else None,
            **conv1_opts,
        )
        self.conv1 = TtConv2d(conv1_config, device)

        # conv2
        conv2_opts = layer_optimisations.conv2.copy()
        conv2_config = create_conv2d_configuration(
            conv_args.conv2,
            conv_pth.conv2,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            model_configs=model_configs,
            layer_path=f"{block_path}.conv2" if block_path else None,
            **conv2_opts,
        )
        self.conv2 = TtConv2d(conv2_config, device)

        # conv3
        conv3_opts = layer_optimisations.conv3.copy()
        conv3_config = create_conv2d_configuration(
            conv_args.conv3,
            conv_pth.conv3,
            device=device,
            activation=None,
            model_configs=model_configs,
            layer_path=f"{block_path}.conv3" if block_path else None,
            **conv3_opts,
        )
        self.conv3 = TtConv2d(conv3_config, device)

        if is_downsample:
            downsample_opts = layer_optimisations.downsample.copy()
            downsample_config = create_conv2d_configuration(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation=None,
                model_configs=model_configs,
                layer_path=f"{block_path}.downsample" if block_path else None,
                **downsample_opts,
            )
            self.downsample = TtConv2d(downsample_config, device)

    def __call__(self, x_identity):
        x = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv2(x)
        x = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
