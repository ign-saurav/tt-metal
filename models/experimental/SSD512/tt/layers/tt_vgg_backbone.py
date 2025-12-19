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
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import Conv2dNormActivation, Maxpool2DOperation


@dataclass
class VGGBackboneOptimizationConfig:
    """
    Dataclass to store per-convolution-layer optimization configurations.

    Each conv field (conv1, conv2, etc.) contains a dictionary of optimization
    parameters that will be applied to the corresponding Conv2d layer in the
    VGG backbone. This allows fine-grained tuning of memory layout, sharding,
    and buffer management for each layer.

    Attributes:
        conv1-conv15: Dictionary of optimization parameters for each convolution layer.
                     Each dictionary can contain:
                     - sharding_strategy: Sharding strategy configuration
                     - deallocate_activation: Whether to free activation memory after use
                     - Other Conv2dConfiguration override parameters
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


# ============================================================================
# Per-Layer Optimization Configurations
# ============================================================================

# Per-layer optimization configurations for VGG backbone.
# These settings control memory layout, sharding strategy, and buffer management
# to optimize performance and memory usage for each specific layer.
#
# Configuration patterns:
# - Early layers (conv1-conv2): Height sharding for large spatial dimensions
# - Middle layers (conv3-conv10): Block sharding or auto sharding
# - Late layers (conv11-conv12): Auto sharding for dilated/1x1 convolutions
# - Extended layers (conv13-conv15): Auto sharding for additional backbone layers
vgg_backbone_optimizations = VGGBackboneOptimizationConfig(
    # conv1: First convolution layer (3->64 channels, large spatial dimensions)
    # Uses height sharding with large block height for efficient memory usage
    conv1={
        "sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "deallocate_activation": True,
    },
    # conv2: Second convolution layer (64->64 channels)
    # Uses height sharding with re-sharding optimization enabled
    conv2={
        "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "deallocate_activation": True,
    },
    # conv3: Third convolution layer (64->128 channels)
    # Uses block sharding with large block height for initial feature extraction
    conv3={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "deallocate_activation": True,
    },
    # conv4: Fourth convolution layer (128->128 channels)
    # Uses auto sharding to let TTNN choose optimal strategy
    conv4={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv5: Fifth convolution layer (128->256 channels)
    # Uses block sharding for consistent memory layout
    conv5={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    # conv6: Sixth convolution layer (256->256 channels)
    # Uses auto sharding for flexibility
    conv6={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv7: Seventh convolution layer (256->256 channels)
    # Uses auto sharding for flexibility
    conv7={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv8: Eighth convolution layer (256->512 channels)
    # Uses block sharding for consistent memory layout
    conv8={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    # conv9: Ninth convolution layer (512->512 channels)
    # Uses block sharding for consistent memory layout
    conv9={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "deallocate_activation": True,
    },
    # conv10: Tenth convolution layer (512->512 channels)
    # Uses auto sharding for flexibility
    conv10={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv11: Dilated convolution layer (512->1024 channels)
    # Uses auto sharding to handle dilated convolution efficiently
    conv11={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv12: Final 1x1 convolution layer (1024->1024 channels)
    # Uses block sharding with re-sharding optimization for 1x1 convolutions
    conv12={
        "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "deallocate_activation": True,
    },
    # conv13: Extended backbone layer (1024->256 channels)
    # Uses auto sharding for additional backbone layers
    conv13={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv14: Extended backbone layer (256->512 channels)
    # Uses auto sharding for additional backbone layers
    conv14={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
    # conv15: Extended backbone layer (512->512 channels)
    # Uses auto sharding for additional backbone layers
    conv15={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "deallocate_activation": True,
    },
)

# Backward compatibility alias
vgg_backbone_optimisations = vgg_backbone_optimizations


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
                # Apply per-layer optimizations from vgg_backbone_optimizations
                # Layer indices are 1-based (conv1, conv2, etc.)
                optimization_key = f"conv{i+1}"
                override_dict = getattr(vgg_backbone_optimizations, optimization_key, {})

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
