# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


class SECONDFPN_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels=[256, 512, 128],
        out_channels=[128, 128, 128],  # All deblocks output 128 channels
        upsample_strides=[4, 2, 1],
        model_config=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_levels = len(in_channels)

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.deblocks = parameters.deblocks
        logger.info(f"SECONDFPN init: {self.num_levels} levels")

    def __call__(self, x, batch_size=1):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        ups = []
        target_height = None
        target_width = None
        for i in range(self.num_levels):
            feat = x[i]
            height, width = feat.shape[1], feat.shape[2]
            stride = self.upsample_strides[i]

            # Calculate output dimensions
            out_height = height * stride
            out_width = width * stride

            # Upsample if stride > 1
            if stride > 1:
                if feat.layout != ttnn.ROW_MAJOR_LAYOUT:
                    # Ensure INTERLEAVED memory layout (not sharded)
                    if feat.is_sharded():
                        feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                    elif feat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                        feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

                    feat = ttnn.to_layout(feat, ttnn.ROW_MAJOR_LAYOUT)

                # Reshape to ensure proper format [B, H, W, C] if needed
                if len(feat.shape) != 4 or feat.shape[0] != batch_size:
                    feat = ttnn.reshape(feat, (batch_size, height, width, feat.shape[-1]))

                # Call upsample with scale_factor (ROW_MAJOR_LAYOUT doesn't require tile alignment)
                feat = ttnn.upsample(feat, scale_factor=stride, mode="nearest")

                # Convert back to TILE_LAYOUT for subsequent operations
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)

                # Ensure DRAM memory config after layout conversion
                # Check if buffer type is DRAM by comparing with DRAM_MEMORY_CONFIG
                if feat.memory_config().buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type:
                    feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            # Conv + BN + ReLU
            # Get kernel size from weight tensor shape
            kernel_size = self.deblocks[i].kernel_size

            if i == 0:
                # First deblock: use the upsampled dimensions as target
                target_height = out_height
                target_width = out_width
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

            conv_out_height = out_height + 2 * padding[0] - kernel_size[0] + 1
            conv_out_width = out_width + 2 * padding[1] - kernel_size[1] + 1

            if feat.is_sharded():
                feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
            else:
                # Ensure DRAM and INTERLEAVED
                mem_cfg = feat.memory_config()
                if (
                    mem_cfg.buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type
                    or mem_cfg.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                ):
                    feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            # Ensure tensor is in TILE_LAYOUT for conv2d (DRAM conv requires TILE_LAYOUT)
            if feat.layout != ttnn.TILE_LAYOUT:
                # Convert layout but keep DRAM memory config
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                # Re-ensure DRAM after layout conversion
                if feat.memory_config().buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type:
                    feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            feat = ttnn_conv2d(
                input_tensor=feat,
                weight_tensor=self.deblocks[i].conv_weight,
                bias_tensor=self.deblocks[i].conv_bias,
                device=self.device,
                batch_size=batch_size,
                input_height=out_height,
                input_width=out_width,
                in_channels=self.in_channels[i],
                out_channels=self.out_channels[i],
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,  # Use BLOCK_SHARDED like ResNet50
                packer_l1_acc=False,  # Disable L1 accumulator to avoid L1 buffer clashes
            )

            # Reshape output to [B, H, W, C] format using actual output dimensions
            # ttnn_conv2d may return flattened format [1, 1, H*W, C] or [B, 1, H*W, C]
            actual_shape = feat.shape
            total_elements = (
                feat.shape[0] * feat.shape[1] * feat.shape[2] * feat.shape[3]
                if len(feat.shape) == 4
                else feat.shape[0] * feat.shape[1] * feat.shape[2]
            )

            if len(feat.shape) == 4:
                if feat.shape[1] == 1 and feat.shape[2] == conv_out_height * conv_out_width:
                    # Flattened format: [B, 1, H*W, C] -> [B, H, W, C]
                    feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))
                elif feat.shape[1] == conv_out_height and feat.shape[2] == conv_out_width:
                    # Already in [B, H, W, C] format with correct dimensions
                    pass
                else:
                    # Try to infer from total elements
                    expected_elements = batch_size * conv_out_height * conv_out_width * self.out_channels[i]
                    if total_elements == expected_elements:
                        feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))
                    else:
                        # Total elements don't match - use actual dimensions from tensor
                        # Calculate actual spatial dimensions from total elements
                        spatial_size = total_elements // (batch_size * self.out_channels[i])
                        # Try to find factors that match
                        import math

                        # Find closest square root
                        sqrt_size = int(math.sqrt(spatial_size))
                        # Try to find dimensions that work
                        for h in range(sqrt_size, 0, -1):
                            if spatial_size % h == 0:
                                actual_h = h
                                actual_w = spatial_size // h
                                break
                        else:
                            actual_h = spatial_size
                            actual_w = 1

                        # Reshape to actual dimensions first
                        feat = ttnn.reshape(feat, (batch_size, actual_h, actual_w, self.out_channels[i]))
                        conv_out_height = actual_h
                        conv_out_width = actual_w
            elif len(feat.shape) == 3:
                # 3D tensor, reshape to 4D using calculated dimensions
                feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))

            # Ensure all deblocks output the same spatial dimensions for concatenation
            # All outputs must have the same spatial dimensions (target_height x target_width)
            if conv_out_height != target_height or conv_out_width != target_width:
                # Check if we can reshape (element count must match)
                current_elements = batch_size * conv_out_height * conv_out_width * self.out_channels[i]
                target_elements = batch_size * target_height * target_width * self.out_channels[i]

                if current_elements == target_elements:
                    # Same number of elements, safe to reshape
                    feat = ttnn.reshape(feat, (batch_size, target_height, target_width, self.out_channels[i]))
                else:
                    # Different number of elements - need to crop or pad to match target
                    # This happens because standard "same" padding with even kernels may change size
                    # We'll crop if output is larger, or pad if smaller
                    if conv_out_height > target_height or conv_out_width > target_width:
                        # Output is larger - crop to target size (center crop)
                        h_start = (conv_out_height - target_height) // 2
                        w_start = (conv_out_width - target_width) // 2
                        h_end = h_start + target_height
                        w_end = w_start + target_width
                        feat = ttnn.slice(
                            feat, [0, h_start, w_start, 0], [batch_size, h_end, w_end, self.out_channels[i]]
                        )
                    elif conv_out_height < target_height or conv_out_width < target_width:
                        # Output is smaller - pad to target size with zeros
                        # Calculate padding needed
                        pad_h = target_height - conv_out_height
                        pad_w = target_width - conv_out_width
                        # Pad symmetrically: pad before and after
                        pad_h_before = pad_h // 2
                        pad_h_after = pad_h - pad_h_before
                        pad_w_before = pad_w // 2
                        pad_w_after = pad_w - pad_w_before
                        # Use ttnn.pad to add zeros
                        feat = ttnn.pad(
                            feat, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)), value=0.0
                        )
                    else:
                        # Should not reach here
                        raise ValueError(f"Unexpected dimension mismatch case for deblock {i}")

            ups.append(feat)

        # Concatenate along channel dimension
        out = ttnn.concat(ups, dim=-1)
        return [out]


def prepare_secondfpn_parameters(state_dict, in_channels=[256, 512, 128], out_channels=[128, 128, 128]):
    class Parameters:
        pass

    params = Parameters()
    params.deblocks = []

    # Find the actual prefix used in this checkpoint
    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.img_neck.",
        "img_backbone.img_neck.",
        "backbone.img_neck.",
        "img_neck.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find img_neck prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No img_neck keys found in checkpoint")

    logger.info(f"Using SECONDFPN prefix: {prefix}")

    for i in range(len(in_channels)):
        deblock = Parameters()
        weight = state_dict[f"{prefix}deblocks.{i}.0.weight"]

        # Extract kernel size from weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        deblock.kernel_size = (kernel_h, kernel_w)

        # CRITICAL: Always truncate weight to match out_channels[i] before any processing
        # The checkpoint may have more output channels than the BN, so we must truncate
        original_channels = weight.shape[0]
        if weight.shape[0] != out_channels[i]:
            if weight.shape[0] > out_channels[i]:
                logger.info(
                    f"deblocks.{i}.0.weight: truncating from {original_channels} to {out_channels[i]} output channels to match BN"
                )
                # Use clone() to ensure we get a new tensor, not a view
                weight = weight[: out_channels[i], :, :, :].clone()
            else:
                raise ValueError(
                    f"deblocks.{i}.0.weight has {weight.shape[0]} output channels, but need {out_channels[i]}"
                )

        # Verify truncation worked
        assert (
            weight.shape[0] == out_channels[i]
        ), f"Weight truncation failed: expected {out_channels[i]} channels, got {weight.shape[0]}"

        # Load conv weight and bias
        conv_weight = weight.to(torch.bfloat16)

        # Double-check after conversion
        assert (
            conv_weight.shape[0] == out_channels[i]
        ), f"After conversion: conv_weight has {conv_weight.shape[0]} channels, expected {out_channels[i]}"
        conv_bias = state_dict.get(f"{prefix}deblocks.{i}.0.bias", None)
        if conv_bias is not None:
            conv_bias = conv_bias.to(torch.bfloat16)

        # Load BN params and fuse into conv weights/bias
        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
        bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            # Verify BN channels match conv output channels (should match out_channels[i] after truncation)
            conv_out_channels = conv_weight.shape[0]
            bn_channels = bn_weight.shape[0]

            # Both should match out_channels[i] at this point
            if conv_out_channels != out_channels[i]:
                raise ValueError(
                    f"deblocks.{i}: conv_weight has {conv_out_channels} output channels, expected {out_channels[i]}"
                )

            if bn_channels != conv_out_channels:
                if bn_channels > conv_out_channels:
                    logger.warning(
                        f"Deblock {i}: BN channels ({bn_channels}) > conv output channels ({conv_out_channels}). "
                        f"Using first {conv_out_channels} BN channels."
                    )
                    # Take only the first conv_out_channels BN parameters
                    bn_weight = bn_weight[:conv_out_channels]
                    bn_bias = bn_bias[:conv_out_channels]
                    bn_mean = bn_mean[:conv_out_channels]
                    bn_var = bn_var[:conv_out_channels]
                else:
                    raise ValueError(
                        f"Deblock {i}: BN channels ({bn_channels}) < conv output channels ({conv_out_channels}). "
                        f"Cannot fuse BN."
                    )

            # Fuse BatchNorm into conv weights and bias
            # Formula: scale = bn_weight / sqrt(bn_var + eps)
            eps = 1e-5  # Standard BN epsilon
            std = torch.sqrt(bn_var + eps)
            scale = bn_weight / std

            # Fuse into conv weight: multiply each output channel by its scale
            # conv_weight shape: (out_channels, in_channels, kH, kW)
            fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

            # Fuse into bias
            # If conv has bias: fused_bias = scale * conv_bias + bn_bias - (bn_weight * bn_mean / std)
            # If conv has no bias: fused_bias = bn_bias - (bn_weight * bn_mean / std)
            if conv_bias is not None:
                fused_bias = scale * conv_bias + bn_bias - (bn_weight * bn_mean / std)
            else:
                fused_bias = bn_bias - (bn_weight * bn_mean / std)

            deblock.conv_weight = fused_weight.to(torch.bfloat16)
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use conv weights/bias as-is
            deblock.conv_weight = conv_weight
            deblock.conv_bias = conv_bias

        params.deblocks.append(deblock)

    logger.info(f"Prepared SECONDFPN parameters for {len(in_channels)} levels")
    return params
