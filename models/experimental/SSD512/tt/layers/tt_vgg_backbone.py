# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of VGG backbone for SSD512.

This module provides:
- TtVGGBackbone: Main class that builds and executes VGG backbone layers on TTNN
- Per-layer optimization configurations for memory and performance tuning
- Helper functions to override Conv2dConfiguration parameters
"""

import ttnn
from dataclasses import dataclass, replace
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import Conv2dNormActivation, Maxpool2DOperation


@dataclass
class TtVGGBackbone:
    """
    Dataclass to store per-convolution-layer optimization configurations.

    Each conv field (conv1, conv2, etc.) contains a dictionary of optimization
    parameters that will be applied to the corresponding Conv2d layer in the
    VGG backbone. This allows fine-grained tuning of memory layout, sharding,
    and buffer management for each layer.

    Attributes:
        conv1-conv19: Dictionary of optimization parameters for each convolution layer
    """

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


# Per-layer optimization configurations for VGG backbone
# These settings control memory layout, sharding strategy, and buffer management
# to optimize performance and memory usage for each specific layer
vgg_backbone_optimisations = TtVGGBackbone(
    # conv1: First convolution layer (typically 3->64 channels)
    conv1={
        "sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,  # Enable double buffering for activations
        "enable_weights_double_buffer": True,  # Enable double buffering for weights
        "deallocate_activation": True,  # Free activation memory after use
        "reallocate_halo_output": True,  # Reallocate halo regions for sharded operations
    },
    # conv2: Second convolution layer
    conv2={
        "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    # conv3-conv10: Middle convolution layers (typically 64->128->256->512 channels)
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
    # conv11: Typically the dilated convolution (512->1024 channels)
    conv11={
        "sharding_strategy": AutoShardedStrategyConfiguration(),  # Let TTNN choose optimal sharding
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    # conv12: Final 1x1 convolution (1024->1024 channels)
    conv12={
        "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    # conv13-conv19: Additional layers for extended VGG backbone
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
    """
    Create a new Conv2dConfiguration with overridden parameters.

    Since Conv2dConfiguration is a frozen dataclass, we cannot modify it in-place.
    This function uses dataclasses.replace() to create a new instance with
    the specified parameters overridden.

    Args:
        config: Conv2dConfiguration instance to override
        override_dict: Dictionary of parameter names and values to override

    Returns:
        New Conv2dConfiguration instance with overridden parameters, or the
        original config if it's not a Conv2dConfiguration

    Example:
        >>> override_dict = {"sharding_strategy": BlockShardedStrategyConfiguration()}
        >>> new_config = override_conv_config(old_config, override_dict)
    """
    if not isinstance(config, Conv2dConfiguration):
        return config
    return replace(config, **override_dict)


class TtVGGBackbone:
    """
    TTNN implementation of VGG backbone for SSD512.

    This class builds a VGG backbone network using TTNN operations, applying
    per-layer optimizations for memory and performance. It processes a sequence
    of Conv2d and MaxPool2d configurations and executes them on the TTNN device.

    The backbone consists of:
    - Multiple Conv2d layers with ReLU activation
    - MaxPool2d layers for downsampling
    - Per-layer optimization configurations for memory efficiency

    Attributes:
        batch_size: Batch size for input tensors
        device: TTNN device to execute operations on
        block: List of layer operations (Conv2dNormActivation or Maxpool2DOperation)
    """

    def __init__(self, conv_config_layer, device, batch_size: int):
        """
        Initialize VGG backbone with layer configurations.

        Args:
            conv_config_layer: List of Conv2dConfiguration and MaxPool2dConfiguration
                              objects defining the network layers
            device: TTNN device to execute operations on
            batch_size: Batch size for input tensors
        """
        self.batch_size = batch_size
        self.device = device

        layers = []

        # Process each layer configuration and build the network
        for i, conv_config in enumerate(conv_config_layer):
            if isinstance(conv_config, Conv2dConfiguration):
                # Apply per-layer optimizations from vgg_backbone_optimisations
                # Layer indices are 1-based (conv1, conv2, etc.)
                optimisation_key = f"conv{i+1}"
                override_dict = getattr(vgg_backbone_optimisations, optimisation_key, {})

                # Create new config with optimizations applied
                updated_config = override_conv_config(conv_config, override_dict)

                # Build Conv2d layer with ReLU activation
                layers.append(
                    Conv2dNormActivation(
                        device=device,
                        conv_config=updated_config,
                        activation_layer=ttnn.relu,
                    )
                )
            elif isinstance(conv_config, MaxPool2dConfiguration):
                # Build MaxPool2d layer
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
        """
        Execute the VGG backbone forward pass.

        Args:
            device: TTNN device to execute operations on
            input: Input tensor in TTNN format (BHWC layout)
            return_source: If True, return intermediate feature maps at layer 12
                         (used for SSD512 feature extraction)

        Returns:
            If return_source=False: Final output tensor
            If return_source=True: Tuple of (final_output, [source_tensor])

        Note:
            Currently extracts source features at layer 12 (index 12) for SSD512
            multi-scale feature extraction. This can be customized based on
            the specific feature extraction requirements.
        """
        tt_sources = []

        # Execute each layer sequentially
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

            # Extract intermediate features at layer 12 for SSD512
            # This corresponds to a specific feature map used by the detection heads
            if i == 12:
                tt_sources.append(result)

        if return_source:
            return result, tt_sources
        return result
