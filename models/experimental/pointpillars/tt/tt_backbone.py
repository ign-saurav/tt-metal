# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def preprocess_backbone_parameters(torch_model, checkpoint_state, device, input_height, input_width, batch_size=1):
    """
    Preprocess PointPillars backbone parameters for TTNN.
    Handles all 3 blocks with their respective convolutions.
    """
    parameters = {}

    # Block configurations: (in_channels, out_channels, num_layers, stride)
    block_configs = [
        (64, 64, 3, 2),  # Block 0: 1 strided conv + 3 regular convs
        (64, 128, 5, 2),  # Block 1: 1 strided conv + 5 regular convs
        (128, 256, 5, 2),  # Block 2: 1 strided conv + 5 regular convs
    ]

    current_height = input_height
    current_width = input_width

    logger.info(f"=== Starting preprocessing with initial dimensions: {current_height}x{current_width} ===")

    # Create compute config with HiFi4
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    for block_idx, (in_channels, out_channels, num_layers, stride) in enumerate(block_configs):
        logger.info(f"\n{'='*80}")
        logger.info(
            f"BLOCK {block_idx}: in_channels={in_channels}, out_channels={out_channels}, num_layers={num_layers}, stride={stride}"
        )
        logger.info(f"Block {block_idx} will use input dimensions: {current_height}x{current_width}")
        logger.info(f"{'='*80}")

        block_params = {}

        # Store the dimensions used for THIS ENTIRE BLOCK
        block_input_height = current_height
        block_input_width = current_width

        # Process all convolutions in this block
        for conv_idx in range(num_layers + 1):  # +1 for the first strided conv
            layer_stride = stride if conv_idx == 0 else 1

            logger.info(f"\n--- Block {block_idx}, Conv {conv_idx} ---")
            logger.info(f"  Layer stride: {layer_stride}")
            logger.info(f"  Using dimensions for weight prep: {current_height}x{current_width}")
            logger.info(f"  in_channels={in_channels}, out_channels={out_channels}")

            # Get conv and bn layers from PyTorch model
            layer_offset = conv_idx * 3  # Each conv+bn+relu takes 3 indices
            layer = torch_model.multi_blocks[block_idx][layer_offset]
            bn_layer = torch_model.multi_blocks[block_idx][layer_offset + 1]

            # Get weights from checkpoint
            conv_key = f"multi_blocks.{block_idx}.{layer_offset}.weight"
            bn_weight_key = f"multi_blocks.{block_idx}.{layer_offset + 1}.weight"
            bn_bias_key = f"multi_blocks.{block_idx}.{layer_offset + 1}.bias"
            bn_mean_key = f"multi_blocks.{block_idx}.{layer_offset + 1}.running_mean"
            bn_var_key = f"multi_blocks.{block_idx}.{layer_offset + 1}.running_var"

            weight = checkpoint_state[conv_key]
            logger.info(f"  Conv weight shape from checkpoint: {weight.shape}")

            # Fold BatchNorm into Conv
            with torch.no_grad():
                bn_weight = checkpoint_state[bn_weight_key]
                bn_bias = checkpoint_state[bn_bias_key]
                bn_mean = checkpoint_state[bn_mean_key]
                bn_var = checkpoint_state[bn_var_key]

                logger.debug(
                    f"  BN params - weight: {bn_weight.shape}, bias: {bn_bias.shape}, mean: {bn_mean.shape}, var: {bn_var.shape}"
                )

                # Manual folding: scale = bn_weight / sqrt(bn_var + eps)
                eps = 1e-3
                scale = bn_weight / torch.sqrt(bn_var + eps)

                # Folded weight and bias
                weight = weight * scale.view(-1, 1, 1, 1)
                bias = bn_bias - bn_mean * scale

            logger.info(f"  After BN folding - weight: {weight.shape}, bias: {bias.shape}")

            # Reshape bias to 4D: [out_channels] -> [1, 1, 1, out_channels]
            bias = bias.reshape(1, 1, 1, -1)
            logger.debug(f"  Bias reshaped to: {bias.shape}")

            # Convert to TTNN tensors
            tt_weight_tensor = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
            tt_bias_tensor = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

            logger.debug(f"  Converted to TTNN tensors")

            # Create conv config with fused ReLU
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            )

            logger.info(f"  Calling prepare_conv_weights with:")
            logger.info(f"    input_height={current_height}, input_width={current_width}")
            logger.info(f"    in_channels={in_channels}, out_channels={out_channels}")
            logger.info(f"    stride=[{layer_stride}, {layer_stride}]")

            # Prepare weights using CURRENT dimensions
            ttnn_weight = ttnn.prepare_conv_weights(
                weight_tensor=tt_weight_tensor,
                input_memory_config=ttnn.L1_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch_size,
                input_height=current_height,
                input_width=current_width,
                kernel_size=[3, 3],
                stride=[layer_stride, layer_stride],
                padding=[1, 1],
                dilation=[1, 1],
                groups=1,
                has_bias=True,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            )

            logger.debug(f"  prepare_conv_weights completed")

            # Prepare bias
            ttnn_bias = ttnn.prepare_conv_bias(
                bias_tensor=tt_bias_tensor,
                input_memory_config=ttnn.L1_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch_size,
                input_height=current_height,
                input_width=current_width,
                kernel_size=[3, 3],
                stride=[layer_stride, layer_stride],
                padding=[1, 1],
                dilation=[1, 1],
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            )

            logger.debug(f"  prepare_conv_bias completed")

            # Move to device
            ttnn_weight = ttnn.to_device(ttnn_weight, device)
            ttnn_bias = ttnn.to_device(ttnn_bias, device)

            logger.debug(f"  Moved weights and bias to device")

            # Store parameters with metadata
            block_params[f"conv_{conv_idx}"] = {
                "weight": ttnn_weight,
                "bias": ttnn_bias,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "input_height": current_height,
                "input_width": current_width,
                "stride": layer_stride,
                "conv_config": conv_config,
            }

            logger.info(
                f"  Stored conv_{conv_idx} params with input_height={current_height}, input_width={current_width}"
            )

            # **CRITICAL**: Update dimensions IMMEDIATELY after strided conv
            if layer_stride > 1:
                old_height = current_height
                old_width = current_width
                current_height = current_height // layer_stride
                current_width = current_width // layer_stride
                in_channels = out_channels
                logger.warning(
                    f"   DIMENSION UPDATE: {old_height}x{old_width} -> {current_height}x{current_width} (stride={layer_stride})"
                )
            else:
                logger.info(f"  No dimension update (stride=1), staying at {current_height}x{current_width}")

        # Update in_channels for next block
        logger.info(f"\nBlock {block_idx} complete. Updating in_channels: {in_channels} -> {out_channels}")
        # in_channels = out_channels

        # Store block parameters
        parameters[f"block_{block_idx}"] = block_params

        logger.info(f"Block {block_idx} stored with {len(block_params)} convolutions")
        logger.info(f"Next block will start with dimensions: {current_height}x{current_width}\n")

    logger.info("=" * 80)
    logger.info("Parameters preprocessed and moved to device")
    logger.info(f"Final dimensions after all blocks: {current_height}x{current_width}")
    logger.info("=" * 80)
    return parameters


class TtPointPillarsBackbone:
    """
    Complete TTNN implementation of PointPillars backbone.
    Implements all 3 blocks with modular conv operations.
    """

    def __init__(self, device, parameters, batch_size=1):
        """
        Args:
            device: TTNN device
            parameters: Preprocessed parameters from preprocess_backbone_parameters
            batch_size: Batch size (default 1)
        """
        self.device = device
        self.parameters = parameters
        self.batch_size = batch_size

        # Extract block configurations from parameters
        self.num_blocks = len([k for k in parameters.keys() if k.startswith("block_")])

        logger.info(f"TtPointPillarsBackbone initialized with {self.num_blocks} blocks")

    def _apply_conv_bn_relu(self, x, conv_params):
        weight = conv_params["weight"]
        bias = conv_params["bias"]
        in_channels = conv_params["in_channels"]
        out_channels = conv_params["out_channels"]
        input_height = conv_params["input_height"]
        input_width = conv_params["input_width"]
        stride = conv_params["stride"]
        conv_config = conv_params["conv_config"]
        logger.debug(f"    Input shape: {x.shape}")
        logger.debug(f"    conv_params keys: {conv_params.keys()}")
        logger.debug(f"    in_channels: {conv_params['in_channels']}")
        logger.debug(f"    out_channels: {conv_params['out_channels']}")
        logger.debug(f"    input_height: {conv_params['input_height']}")
        logger.debug(f"    input_width: {conv_params['input_width']}")
        logger.debug(f"    stride: {conv_params['stride']}")
        # Create compute config with HiFi4
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Create conv config with fused ReLU (MUST match preprocessing)

        # Log input stats
        logger.info(
            f"      Input stats: shape={x.shape}, mean={ttnn.to_torch(x).mean():.4f}, std={ttnn.to_torch(x).std():.4f}, min={ttnn.to_torch(x).min():.4f}, max={ttnn.to_torch(x).max():.4f}"
        )

        # Call ttnn.conv2d with explicit parameters
        output = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=[3, 3],  # All your convs use 3x3
            stride=[stride, stride],  # Convert to list
            padding=[1, 1],  # All your convs use padding=1
            dilation=[1, 1],
            groups=1,
            bias_tensor=bias,
            conv_config=conv_config,
            compute_config=compute_config,
        )

        # Reshape output from [1, 1, N*H*W, C] to [N, H, W, C]
        output_height = input_height // stride
        output_width = input_width // stride

        if output.shape[1] == 1 and output.shape[2] > 1:
            output = ttnn.reshape(output, (self.batch_size, output_height, output_width, out_channels))

        return output

    def __call__(self, x):
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor in NHWC format [batch, height, width, channels]

        Returns:
            List of output tensors from each block in NHWC format
        """
        outputs = []

        for block_idx in range(self.num_blocks):
            block_params = self.parameters[f"block_{block_idx}"]

            logger.debug(f"Processing block {block_idx}")

            # Apply all convolutions in this block
            num_convs = len([k for k in block_params.keys() if k.startswith("conv_")])
            for conv_idx in range(num_convs):
                conv_params = block_params[f"conv_{conv_idx}"]
                x = self._apply_conv_bn_relu(x, conv_params)
                logger.debug(f"Block {block_idx}, Conv {conv_idx} output shape: {x.shape}")

            outputs.append(x)
            logger.debug(f"Block {block_idx} complete, output shape: {x.shape}")

        return outputs
