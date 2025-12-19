# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Extras backbone for SSD512.

This module provides:
- TtExtrasBackbone: Main class that builds and executes Extras backbone layers on TTNN
- Feature extraction at alternating layers for multi-scale detection in SSD512

The Extras backbone is an auxiliary network added to the VGG backbone to provide
additional feature maps at different scales for object detection. It consists of
a series of convolutional layers that progressively reduce spatial dimensions
while maintaining or adjusting channel dimensions.
"""

import ttnn
from models.experimental.SSD512.tt.utils import Conv2dNormActivation


class TtExtrasBackbone:
    """
    TTNN implementation of Extras backbone for SSD512.

    This class builds an Extras backbone network using TTNN operations. The Extras
    backbone is used in SSD512 to generate additional feature maps at different
    scales for multi-scale object detection. It processes a sequence of Conv2d
    configurations and executes them on the TTNN device.

    The backbone consists of:
    - Multiple Conv2d layers with ReLU activation
    - Alternating feature extraction for multi-scale detection
    - Progressive spatial downsampling with varying channel configurations

    Attributes:
        batch_size: Batch size for input tensors
        device: TTNN device to execute operations on
        block: List of layer operations (Conv2dNormActivation)
    """

    def __init__(self, conv_config_layer, device, batch_size: int):
        """
        Initialize Extras backbone with layer configurations.

        Args:
            conv_config_layer: List of Conv2dConfiguration objects defining the
                              network layers. These layers typically alternate
                              between different kernel sizes and stride patterns
                              to create multi-scale feature maps.
            device: TTNN device to execute operations on
            batch_size: Batch size for input tensors
        """
        self.batch_size = batch_size
        self.device = device

        # Build the network by processing each layer configuration
        layers = []
        for i, conv_config in enumerate(conv_config_layer):
            # Each layer is a Conv2d with ReLU activation
            # The Extras backbone uses ReLU activation for all layers
            layers.append(
                Conv2dNormActivation(
                    device=device,
                    conv_config=conv_config,
                    activation_layer=ttnn.relu,
                )
            )

        self.block = layers

    def __call__(self, device, input, return_source=False):
        """
        Execute the Extras backbone forward pass.

        Args:
            device: TTNN device to execute operations on
            input: Input tensor in TTNN format (BHWC layout)
            return_source: If True, return intermediate feature maps at alternating
                         layers (odd indices) for SSD512 multi-scale feature extraction

        Returns:
            If return_source=False: Final output tensor
            If return_source=True: Tuple of (final_output, [source_tensors])
                                  where source_tensors contains feature maps
                                  extracted at layers with odd indices (1, 3, 5, ...)

        Note:
            The Extras backbone extracts features at every odd-indexed layer
            (i % 2 == 1) to provide multi-scale feature maps for SSD512 detection
            heads. This pattern allows the network to capture features at different
            spatial resolutions for detecting objects of various sizes.
        """
        tt_sources = []

        # Execute each layer sequentially
        for i, layer in enumerate(self.block):
            # First layer processes the input, subsequent layers process previous output
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

            # Extract intermediate features at odd-indexed layers for multi-scale detection
            # This pattern (i % 2 == 1) extracts features at layers 1, 3, 5, 7, etc.
            # These feature maps are used by SSD512 detection heads at different scales
            if i % 2 == 1:
                tt_sources.append(result)

        if return_source:
            return result, tt_sources
        return result
