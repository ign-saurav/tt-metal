# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import urllib.request
from pathlib import Path

from loguru import logger
import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    sharding_strategy=AutoShardedStrategyConfiguration(),
    config_tensors_in_dram=False,
    enable_act_double_buffer=None,
    enable_weights_double_buffer=None,
) -> Conv2dConfiguration:
    """Create Conv2dConfiguration from parameters dict."""
    if enable_act_double_buffer is None:
        from models.tt_cnn.tt.builder import WidthShardedStrategyConfiguration

        enable_act_double_buffer = not isinstance(sharding_strategy, WidthShardedStrategyConfiguration)

    if enable_weights_double_buffer is None:
        enable_weights_double_buffer = True

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
        config_tensors_in_dram=config_tensors_in_dram,
    )


def _get_sharding_strategy(input_height, input_width, in_channels, out_channels):
    """Determine optimal sharding strategy based on tensor dimensions."""
    spatial_size = input_height * input_width
    channel_size = in_channels * out_channels

    if spatial_size > channel_size and spatial_size > 256:
        if spatial_size > channel_size * 4:
            act_block_h = min(256, max(32, spatial_size // 32))
            return HeightShardedStrategyConfiguration(act_block_h_override=act_block_h)
        else:
            return AutoShardedStrategyConfiguration()
    else:
        return AutoShardedStrategyConfiguration()


def ensure_checkpoint_downloaded(checkpoint_filename: str, url: str, checkpoint_dir: str = None) -> str:
    """
    Ensure a checkpoint file is downloaded. If it doesn't exist, download it from the URL.

    Args:
        checkpoint_filename: Name of the checkpoint file (e.g., "Swin2SR_ClassicalSR_X2_64.pth")
        url: URL to download the checkpoint from
        checkpoint_dir: Directory where checkpoints are stored. If None, uses default location.

    Returns:
        Path to the checkpoint file
    """
    if checkpoint_dir is None:
        # Default to swin2sr/resources/checkpoints/
        current_file = Path(__file__)
        checkpoint_dir = current_file.parent.parent / "resources" / "checkpoints"

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / checkpoint_filename

    if not checkpoint_path.exists():
        logger.info(f"Checkpoint not found at {checkpoint_path}, downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Successfully downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint from {url}: {e}")

    return str(checkpoint_path)


def get_checkpoint_path(checkpoint_filename: str, checkpoint_dir: str = None) -> str:
    """
    Get the path to a checkpoint file, downloading it if necessary.

    This function automatically downloads the X2 checkpoint if it doesn't exist.
    For other checkpoints, use ensure_checkpoint_downloaded directly.

    Args:
        checkpoint_filename: Name of the checkpoint file
        checkpoint_dir: Directory where checkpoints are stored. If None, uses default location.

    Returns:
        Path to the checkpoint file
    """
    if checkpoint_filename == "Swin2SR_ClassicalSR_X2_64.pth":
        url = "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth"
        return ensure_checkpoint_downloaded(checkpoint_filename, url, checkpoint_dir)
    elif checkpoint_filename == "Swin2SR_ClassicalSR_X4_64.pth":
        url = "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth"
        return ensure_checkpoint_downloaded(checkpoint_filename, url, checkpoint_dir)
    else:
        # For other checkpoints, just return the path (no auto-download)
        if checkpoint_dir is None:
            current_file = Path(__file__)
            checkpoint_dir = current_file.parent.parent / "resources" / "checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / checkpoint_filename
        return str(checkpoint_path)
