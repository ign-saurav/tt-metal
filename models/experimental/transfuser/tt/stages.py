# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


optimization_dict = {
    "layer1": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer2": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer3": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # for image width
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # for image width
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer4": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
}


class Ttstages:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        stage_name,
        device,
    ) -> None:
        self.device = device
        self.inplanes = 32
        # Define stage-specific parameters
        stage_config = {
            "layer1": {"planes": 72, "groups": 3},
            "layer2": {"planes": 216, "groups": 9},
            "layer3": {"planes": 576, "groups": 24},
            "layer4": {"planes": 1512, "groups": 63},
        }

        config = stage_config[stage_name]
        self.layer = self._make_layer(
            parameters=parameters,
            planes=config["planes"],
            blocks=len(parameters.keys()),
            stride=stride,
            groups=config["groups"],
            model_config=model_config,
            stage_name=stage_name,
        )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        layers = []
        self.inplanes = 32

        layer_config = optimization_dict[stage_name]

        # First block (may have downsample)
        downsample = stride != 1 or self.inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=parameters["b1"],
                model_config=model_config,
                layer_config=layer_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                device=self.device,
            )
        )
        self.inplanes = planes

        # Remaining blocks
        for block_num in range(1, blocks):
            block_name = f"b{block_num + 1}"
            layers.append(
                TTRegNetBottleneck(
                    parameters=parameters[block_name],
                    model_config=model_config,
                    layer_config=layer_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    device=self.device,
                )
            )

        return layers

    def __call__(self, x, device):
        # Process image input
        for block in self.layer:
            x = block(x, device)

        return x
