# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import Conv2dNormActivation, Maxpool2DOperation, override_conv_config


@dataclass
class VGGBackboneOptimizationConfig:
    conv1: dict
    conv2: dict
    conv3: dict
    conv4: dict
    conv5: dict
    conv8: dict
    conv9: dict
    conv12: dict


vgg_backbone_optimizations = VGGBackboneOptimizationConfig(
    conv1={
        "sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "deallocate_activation": True,
    },
    conv2={
        "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "deallocate_activation": True,
    },
    conv3={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "deallocate_activation": True,
    },
    conv4={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    conv5={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    conv8={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    conv9={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    conv12={
        "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "deallocate_activation": True,
    },
)


class TtVGGBackbone:
    def __init__(self, conv_config_layer, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        layers = []

        for i, conv_config in enumerate(conv_config_layer):
            if isinstance(conv_config, Conv2dConfiguration):
                optimization_key = f"conv{i+1}"
                override_dict = getattr(vgg_backbone_optimizations, optimization_key, {})
                updated_config = override_conv_config(conv_config, override_dict)

                layers.append(
                    Conv2dNormActivation(
                        device=device,
                        conv_config=updated_config,
                        activation_layer=ttnn.relu,
                    )
                )
            elif isinstance(conv_config, MaxPool2dConfiguration):
                layers.append(
                    Maxpool2DOperation(
                        device=device,
                        conv_config=conv_config,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer configuration found: {type(conv_config)}")

        self.block = layers

    def __call__(self, device, input, return_source=False):
        tt_sources = []

        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

            if i == 12:
                tt_sources.append(result)

        if return_source:
            return result, tt_sources
        return result
