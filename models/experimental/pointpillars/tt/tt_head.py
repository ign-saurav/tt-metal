# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from dataclasses import dataclass
from typing import Optional


@dataclass
class PointPillarsHeadConfig:
    """
    Centralized configuration for PointPillars Head.

    This allows easy modification of datatypes and other parameters
    without changing the implementation code.
    """

    # Data types
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    input_dtype: ttnn.DataType = ttnn.bfloat16

    # Memory configuration
    input_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG

    # Convolution configuration
    shard_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    deallocate_activation: bool = False
    activation: Optional[str] = None  # No activation for head convolutions

    # Compute kernel configuration
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = False

    # Model architecture
    in_channels: int = 384  # From neck output
    n_classes: int = 3
    n_anchors: int = 6

    # Input dimensions
    input_height: int = 248
    input_width: int = 216

    def get_compute_config(self, device):
        """Get compute kernel configuration for the device."""
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.math_fidelity,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def get_conv_config(self):
        """Get Conv2D configuration."""
        return ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate_activation,
        )


def preprocess_head_parameters(checkpoint_state, device, config, batch_size=1):
    """
    Preprocess PointPillars head parameters for TTNN.

    Args:
        checkpoint_state: PyTorch checkpoint containing weights
        device: TTNN device
        config: PointPillarsHeadConfig instance
        batch_size: Batch size (default 1)

    Returns:
        parameters: Dict containing preprocessed weights and biases
    """
    parameters = {}

    # Configuration for each head convolution
    head_configs = [
        ("conv_cls", config.in_channels, config.n_anchors * config.n_classes),
        ("conv_reg", config.in_channels, config.n_anchors * 7),
        ("conv_dir_cls", config.in_channels, config.n_anchors * 2),
    ]

    compute_config = config.get_compute_config(device)
    conv_config = config.get_conv_config()

    for conv_name, in_channels, out_channels in head_configs:
        logger.info(f"Processing {conv_name}: {in_channels} -> {out_channels}")

        # Load weights and bias from checkpoint
        weight_key = f"head.{conv_name}.weight"
        bias_key = f"head.{conv_name}.bias"

        weight = checkpoint_state[weight_key]
        bias = checkpoint_state[bias_key]

        logger.info(f"  Weight shape: {weight.shape}, Bias shape: {bias.shape}")

        # Reshape bias to 4D
        bias = bias.reshape(1, 1, 1, -1)

        # Convert to TTNN tensors using config datatypes
        tt_weight_tensor = ttnn.from_torch(weight, dtype=config.weights_dtype)
        tt_bias_tensor = ttnn.from_torch(bias, dtype=config.weights_dtype)

        # Prepare weights for TTNN
        ttnn_weight = ttnn.prepare_conv_weights(
            weight_tensor=tt_weight_tensor,
            input_memory_config=config.input_memory_config,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=config.input_height,
            input_width=config.input_width,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            device=device,
            has_bias=True,
            input_dtype=config.input_dtype,
            conv_config=conv_config,
            compute_config=compute_config,
        )

        # Prepare bias
        ttnn_bias = ttnn.prepare_conv_bias(
            bias_tensor=tt_bias_tensor,
            input_memory_config=config.input_memory_config,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=config.input_height,
            input_width=config.input_width,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            device=device,
            input_dtype=config.input_dtype,
            conv_config=conv_config,
            compute_config=compute_config,
        )

        # Move to device
        ttnn_weight = ttnn.to_device(ttnn_weight, device)
        ttnn_bias = ttnn.to_device(ttnn_bias, device)

        # Store parameters
        parameters[conv_name] = {
            "weight": ttnn_weight,
            "bias": ttnn_bias,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input_height": config.input_height,
            "input_width": config.input_width,
            "conv_config": conv_config,
        }

        logger.info(f"  {conv_name} parameters prepared and moved to device")

    logger.info("Head parameters preprocessing complete")
    return parameters


class TtPointPillarsHead:
    """TTNN implementation of PointPillars detection head."""

    def __init__(self, parameters, device, config, batch_size=1):
        self.parameters = parameters
        self.device = device
        self.config = config
        self.batch_size = batch_size

        # Initialize compute config from model config
        self.compute_config = config.get_compute_config(device)

        logger.info("TtPointPillarsHead initialized")

    def _apply_conv(self, x, conv_params):
        """Apply a single 1x1 convolution."""
        weight = conv_params["weight"]
        bias = conv_params["bias"]
        in_channels = conv_params["in_channels"]
        out_channels = conv_params["out_channels"]
        input_height = conv_params["input_height"]
        input_width = conv_params["input_width"]
        conv_config = conv_params["conv_config"]

        # Apply convolution
        output = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            groups=1,
            conv_config=conv_config,
            compute_config=self.compute_config,
        )

        # Reshape if needed
        if output.shape[1] == 1 and output.shape[2] > 1:
            N = output.shape[0]
            C = output.shape[3]
            output = ttnn.reshape(output, (N, input_height, input_width, C))

        return output

    def __call__(self, x):
        """Forward pass through the head."""
        logger.debug(f"Head input shape: {x.shape}")

        bbox_cls_pred = self._apply_conv(x, self.parameters["conv_cls"])
        logger.debug(f"bbox_cls_pred shape: {bbox_cls_pred.shape}")

        bbox_pred = self._apply_conv(x, self.parameters["conv_reg"])
        logger.debug(f"bbox_pred shape: {bbox_pred.shape}")

        bbox_dir_cls_pred = self._apply_conv(x, self.parameters["conv_dir_cls"])
        logger.debug(f"bbox_dir_cls_pred shape: {bbox_dir_cls_pred.shape}")

        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
