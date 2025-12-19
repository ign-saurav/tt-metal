# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
)
from dataclasses import dataclass
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import Conv2dNormActivation, Maxpool2DOperation

from dataclasses import replace


@dataclass
class TtVGGBackbone:
    conv1: dict
    conv2: dict
    conv3: dict
    conv4: dict
    conv5: dict
    conv6: dict
    conv7: dict
    conv8: dict
    conv9: dict
    conv10: dict
    conv11: dict
    conv12: dict
    conv13: dict
    conv14: dict
    conv15: dict
    conv16: dict
    conv17: dict
    conv18: dict
    conv19: dict


vgg_backbone_optimisations = TtVGGBackbone(
    conv1={
        "sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
        # "sharding_strategy": AutoShardedStrategyConfiguration(),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv2={
        # "sharding_strategy": AutoShardedStrategyConfiguration(),
        # "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),#act_block_h_override=15 * 32),
        "sharding_strategy": HeightShardedStrategyConfiguration(
            reshard_if_not_optimal=True, act_block_h_override=32
        ),  # act_block_h_override=15 * 32),
        # "sharding_strategy": WidthSliceStrategyConfiguration(reshard_if_not_optimal=True),#act_block_h_override=15 * 32),
        # "slice_strategy":  WidthSliceStrategyConfiguration(num_slices=4),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv3={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv4={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv5={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv6={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv7={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv8={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv9={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv10={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv11={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        # "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),#act_block_h_override=15 * 32),
        # "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),#act_block_h_override=15 * 32),
        # "sharding_strategy": WidthSliceStrategyConfiguration(reshard_if_not_optimal=True),#act_block_h_override=15 * 32),
        # "slice_strategy":  WidthSliceStrategyConfiguration(num_slices=4),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv12={
        # "sharding_strategy": AutoShardedStrategyConfiguration(),
        "sharding_strategy": BlockShardedStrategyConfiguration(
            reshard_if_not_optimal=True, act_block_h_override=32
        ),  # act_block_h_override=15 * 32),
        # "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),#act_block_h_override=15 * 32),
        # "sharding_strategy": WidthShardedStrategyConfiguration(reshard_if_not_optimal=True,act_block_w_div=32),#act_block_h_override=15 * 32),
        # "slice_strategy":  WidthSliceStrategyConfiguration(num_slices=4),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv13={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv14={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv15={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv16={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv17={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv18={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv19={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
)


def override_conv_config(config, override_dict):
    if not isinstance(config, Conv2dConfiguration):
        return config
    return replace(config, **override_dict)


class TtVGGBackbone:
    def __init__(self, conv_config_layer, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        layers = []
        conv_count = 0

        # print(conv_config_layer)
        for i, conv_config in enumerate(conv_config_layer):
            # Explicitly distinguish between Conv2dNormActivation and Maxpool2DOperation by checking type or attribute unique to each
            if isinstance(conv_config, Conv2dConfiguration):
                optimisation_key = f"conv{i+1}"
                # print("optimiasationkey", optimisation_key)
                override_dict = getattr(vgg_backbone_optimisations, optimisation_key, {})
                updated_config = override_conv_config(conv_config, override_dict)
                layers.append(
                    Conv2dNormActivation(
                        device=device,
                        conv_config=updated_config,
                        activation_layer=ttnn.relu,
                    )
                )
                # conv_count+=1
            elif isinstance(conv_config, MaxPool2dConfiguration):
                # This is a maxpool config, instantiate Maxpool2DOperation
                layers.append(
                    Maxpool2DOperation(
                        device=device,
                        conv_config=conv_config,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer configuration found: {type(conv_config)}")

            # if i > 2:
            #     break
        self.block = layers

    def __call__(self, device, input):
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)
            print("layer_done", i + 1)

        return result
