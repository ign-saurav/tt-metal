# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of Swin2SR reference implementation.

This script demonstrates how to use the PyTorch reference implementation
of Swin2SR for image super-resolution.
"""

import torch
from loguru import logger

from models.experimental.swin2sr.reference.swin2sr import Swin2SR


def create_classical_sr_model(scale: int = 2, training_patch_size: int = 64) -> Swin2SR:
    """Create a Swin2SR model for classical super-resolution.

    Args:
        scale: Upscale factor (2, 3, 4, or 8).
        training_patch_size: Patch size used during training.

    Returns:
        Swin2SR model instance.
    """
    model = Swin2SR(
        upscale=scale,
        img_size=training_patch_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    return model


def example_inference():
    """Example inference with Swin2SR model."""
    # Create model
    model = create_classical_sr_model(scale=2, training_patch_size=64)
    model.eval()

    # Create dummy input (B, C, H, W)
    # In practice, this would be a low-resolution image
    batch_size = 1
    channels = 3
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Upscale factor: {output.shape[2] / input_tensor.shape[2]}")


if __name__ == "__main__":
    example_inference()
