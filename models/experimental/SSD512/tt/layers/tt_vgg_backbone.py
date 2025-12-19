# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
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


class Conv2dNormActivation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        activation_layer=None,
    ):
        # if activation_layer == ttnn.relu:
        #     # self.activation_layer = None
        #     activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        # else:
        #     self.activation_layer = activation_layer
        #     activation = None

        # self.conv_config = Conv2dConfiguration.from_torch(
        #     layer, input_height=input_height, input_width=input_width, batch_size=batch_size
        # )
        self.conv_config = conv_config
        self.activation_layer = activation_layer

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        # input_tensor=input_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        # input_tensor = post_conv_reshape(input_tensor, out_height=_out_height, out_width=_out_width)
        if self.activation_layer is not None:
            # input_tensor=input_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


class Maxpool2DOperation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        # activation_layer=None,
    ):
        self.conv_config = conv_config
        # self.activation_layer = activation_layer

        # self.conv = TtConv2d(self.conv_config, device)
        self.pool = TtMaxPool2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        # input_tensor=input_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)
        input_tensor = self.pool(input_tensor)

        return input_tensor


# def override_conv_configs(config, override_dict):
#         """
#         Takes a list of Conv2dConfiguration (and/or MaxPool2dConfiguration) instances,
#         and updates all Conv2dConfiguration objects with the parameters from override_dict.

#         Returns a new list with updated configurations.
#         - Non-Conv2dConfiguration objects (e.g. MaxPool2dConfiguration) are left unchanged.
#         """
#         updated = []
#         # for config in conv_configs:
#         if isinstance(config, Conv2dConfiguration):
#             # Copy to avoid mutating original
#             cfg = config.copy() if hasattr(config, 'copy') else config
#             for k, v in override_dict.items():
#                 if hasattr(cfg, k):
#                     setattr(cfg, k, v)
#                 elif hasattr(cfg, f"_{k}"):  # private attr fallback
#                     setattr(cfg, f"_{k}", v)
#             updated.append(cfg)
#         else:
#             updated.append(config)
#         return updated


from dataclasses import replace


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
                # This is a conv config, instantiate Conv2dNormActivation
                # INSERT_YOUR_CODE
                # Helper function to override (update) parameters in all Conv2dConfiguration objects
                # updated_config= override_conv_config(conv_config,vgg_backbone_optimisations.conv1 )
                # TODO: Pass the correct dict of optimizations for each layer (e.g., conv1, conv2, ...); example below assumes conv names in order.
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
