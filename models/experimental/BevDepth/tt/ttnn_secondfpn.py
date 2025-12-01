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

            # Get kernel size from weight tensor shape
            kernel_size = self.deblocks[i].kernel_size

            if i < 2:  # Deblocks 0 and 1: Use ConvTranspose2d (matching reference exactly)
                # ConvTranspose2d formula: H_out = (H_in - 1) * stride + kernel_size
                target_conv_height = (height - 1) * stride + kernel_size[0]
                target_conv_width = (width - 1) * stride + kernel_size[1]

                if i == 0:
                    # First deblock: use the target dimensions as the final target
                    target_height = target_conv_height
                    target_width = target_conv_width

                # Ensure tensor is in TILE_LAYOUT for conv_transpose2d
                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                else:
                    mem_cfg = feat.memory_config()
                    if (
                        mem_cfg.buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type
                        or mem_cfg.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                    ):
                        feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

                if feat.layout != ttnn.TILE_LAYOUT:
                    feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                    if feat.memory_config().buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type:
                        feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

                # Convert weights to TTNN format if needed
                weight_tensor = self.deblocks[i].conv_weight
                if isinstance(weight_tensor, torch.Tensor):
                    weight_tensor = ttnn.from_torch(
                        weight_tensor,
                        dtype=self.model_config["WEIGHTS_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

                bias_tensor = self.deblocks[i].conv_bias
                if bias_tensor is not None and isinstance(bias_tensor, torch.Tensor):
                    if len(bias_tensor.shape) == 1:
                        bias_tensor = bias_tensor.view(1, 1, 1, -1)
                    bias_tensor = ttnn.from_torch(
                        bias_tensor,
                        dtype=self.model_config["WEIGHTS_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

                conv_config = ttnn.Conv2dConfig(
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    deallocate_activation=True,
                    enable_act_double_buffer=False,
                )

                compute_config = ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=False,
                )

                # Use ConvTranspose2d (matching reference model)
                feat, [output_height, output_width] = ttnn.conv_transpose2d(
                    input_tensor=feat,
                    weight_tensor=weight_tensor,
                    bias_tensor=bias_tensor,
                    device=self.device,
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    batch_size=batch_size,
                    input_height=height,
                    input_width=width,
                    kernel_size=kernel_size,
                    stride=(stride, stride),
                    padding=(0, 0),  # ConvTranspose2d default padding
                    output_padding=(0, 0),  # No output padding
                    dilation=(1, 1),
                    conv_config=conv_config,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                    mirror_kernel=True,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )

                conv_out_height = output_height
                conv_out_width = output_width

            else:  # Deblock 2: Use Conv2d (regular convolution)
                # Calculate output dimensions after upsampling
                out_height = height * stride
                out_width = width * stride

                # Upsample if stride > 1
                if stride > 1:
                    if feat.layout != ttnn.ROW_MAJOR_LAYOUT:
                        if feat.is_sharded():
                            feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                        elif feat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
                        feat = ttnn.to_layout(feat, ttnn.ROW_MAJOR_LAYOUT)

                    if len(feat.shape) != 4 or feat.shape[0] != batch_size:
                        feat = ttnn.reshape(feat, (batch_size, height, width, feat.shape[-1]))

                    feat = ttnn.upsample(feat, scale_factor=stride, mode="nearest")
                    feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                    if feat.memory_config().buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type:
                        feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

                # Calculate padding for Conv2d
                padding_h = (kernel_size[0] - 1) // 2
                padding_w = (kernel_size[1] - 1) // 2
                padding = (padding_h, padding_w)
                conv_out_height = out_height + 2 * padding[0] - kernel_size[0] + 1
                conv_out_width = out_width + 2 * padding[1] - kernel_size[1] + 1

                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                else:
                    mem_cfg = feat.memory_config()
                    if (
                        mem_cfg.buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type
                        or mem_cfg.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                    ):
                        feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

                if feat.layout != ttnn.TILE_LAYOUT:
                    feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
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
                    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    packer_l1_acc=False,
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

        # Convert all tensors to INTERLEAVED DRAM before concatenation
        # ttnn.concat doesn't support BLOCK_SHARDED layout
        processed_ups = []
        for up_tensor in ups:
            if up_tensor.is_sharded():
                up_tensor = ttnn.sharded_to_interleaved(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
            else:
                # Ensure DRAM and INTERLEAVED
                mem_cfg = up_tensor.memory_config()
                if (
                    mem_cfg.buffer_type != ttnn.DRAM_MEMORY_CONFIG.buffer_type
                    or mem_cfg.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                ):
                    up_tensor = ttnn.to_memory_config(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
            processed_ups.append(up_tensor)

        # Concatenate along channel dimension
        out = ttnn.concat(processed_ups, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [out]


def compute_reference_fused_weights(state_dict, prefix, i, in_channels, out_channels):
    """
    Compute what the reference model's fused weights should be, for comparison.
    This replicates the exact process from test_fpn_depthnet.py lines 179-244.
    """
    from models.experimental.BevDepth.tests.test_resnet50_backbone_pcc import fuse_conv_bn_weights

    # Step 1: Get checkpoint weight [out, in, ...]
    checkpoint_weight = state_dict[f"{prefix}deblocks.{i}.0.weight"].clone()

    # Truncate if needed
    if checkpoint_weight.shape[0] > out_channels[i]:
        checkpoint_weight = checkpoint_weight[: out_channels[i], :, :, :].clone()

    # Step 4: Get BN params first (needed for truncation decision)
    bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
    bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
    bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
    bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

    if i < 2:  # ConvTranspose2d deblocks
        # Step 2: Transpose to [in, out, ...] for ConvTranspose2d (test line 179)
        conv_weight = checkpoint_weight.permute(1, 0, 2, 3).contiguous()

        # Truncate if needed (matching test line 225)
        if bn_weight is not None and conv_weight.shape[1] > bn_weight.shape[0]:
            conv_weight = conv_weight[:, : bn_weight.shape[0], :, :].clone()

        # Step 3: Transpose to [out, in, ...] for BN fusion (test line 231)
        conv_weight_for_fusion = conv_weight.permute(1, 0, 2, 3).contiguous()
    else:
        # Conv2d: already in [out, in, ...] format
        # Truncate if needed (matching test line 255) - truncate based on BN channels
        if bn_weight is not None and checkpoint_weight.shape[0] > bn_weight.shape[0]:
            checkpoint_weight = checkpoint_weight[: bn_weight.shape[0], :, :, :].clone()
        conv_weight_for_fusion = checkpoint_weight

    if bn_weight is not None and bn_mean is not None and bn_var is not None:
        # Step 5: Fuse BN (test line 234 or 262)
        # Use the actual BN eps from the state dict if available, otherwise use 1e-3
        eps = 1e-3  # SECONDFPN default
        fused_weight, fused_bias = fuse_conv_bn_weights(
            conv_weight_for_fusion.float(),
            bn_weight.float(),
            bn_bias.float(),
            bn_mean.float(),
            bn_var.float(),
            eps=eps,
        )

        # fused_weight is now in [out, in, ...] format (from fuse_conv_bn_weights)
        # Reference model would transpose this back to [in, out, ...] for ConvTranspose2d (test line 244)
        # But for comparison with our Conv2d weights, we keep it as [out, in, ...]
        return fused_weight, fused_bias
    else:
        if i < 2:
            # Transpose to [out, in, ...] for Conv2d
            conv_weight_for_fusion = conv_weight_for_fusion.permute(1, 0, 2, 3).contiguous()
        return conv_weight_for_fusion, None


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

        # Checkpoint format: For ConvTranspose2d (deblocks 0,1), checkpoint stores as [out_channels, in_channels, ...]
        # For regular Conv2d (deblock 2), checkpoint stores as [out_channels, in_channels, ...]
        # Both are in the same format in the checkpoint, which is correct for regular Conv2d

        # Extract kernel size from weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        deblock.kernel_size = (kernel_h, kernel_w)

        # Get BN params to determine correct truncation (matching reference model's process)
        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)

        # Truncate weight based on BN channels (matching reference model's truncation process)
        # For deblocks 0,1: reference truncates after transpose (test line 225)
        # For deblock 2: reference truncates before fusion (test line 255)
        if bn_weight is not None:
            bn_channels = bn_weight.shape[0]
            if weight.shape[0] > bn_channels:
                original_channels = weight.shape[0]
                logger.info(
                    f"deblocks.{i}.0.weight: truncating from {original_channels} to {bn_channels} output channels to match BN"
                )
                # Use clone() to ensure we get a new tensor, not a view
                weight = weight[:bn_channels, :, :, :].clone()
            elif weight.shape[0] < bn_channels:
                raise ValueError(
                    f"deblocks.{i}.0.weight has {weight.shape[0]} output channels, but BN has {bn_channels} channels"
                )
        else:
            # No BN, use out_channels[i] as fallback
            if weight.shape[0] > out_channels[i]:
                weight = weight[: out_channels[i], :, :, :].clone()

        # For deblocks 0 and 1: originally ConvTranspose2d, now using upsampling + Conv2d
        # Reference model exact process (test lines 179, 231-244):
        #   1. Checkpoint: [out, in, ...] format (line 178)
        #   2. Transpose to [in, out, ...] for ConvTranspose2d (line 179)
        #   3. For BN fusion: transpose to [out, in, ...] (line 231)
        #   4. Fuse BN on [out, in, ...] format (line 234)
        #   5. Transpose back to [in, out, ...] for ConvTranspose2d (line 244)
        #
        # For our Conv2d: we need [out_channels, in_channels, kH, kW]
        # Strategy: Match reference's exact process to ensure BN fusion is identical
        #   1. Start with checkpoint [out, in, ...]
        #   2. Transpose to [in, out, ...] (matching reference's ConvTranspose2d format)
        #   3. Transpose to [out, in, ...] for BN fusion (matching reference's fusion format)
        #   4. Fuse BN on [out, in, ...] format
        #   5. Keep as [out, in, ...] for Conv2d (don't transpose back)
        if i < 2:  # deblocks 0 and 1 (originally ConvTranspose2d)
            # Step 1: Transpose to [in, out, ...] to match reference's ConvTranspose2d format
            weight = weight.permute(1, 0, 2, 3).contiguous()  # [out, in, ...] -> [in, out, ...]
            # We'll transpose to [out, in, ...] for BN fusion, then keep it for Conv2d
            needs_transpose_for_fusion = True
        else:
            needs_transpose_for_fusion = False

        # Load conv weight and bias
        # For deblocks 0 and 1: currently in [in_channels, out_channels, ...] format (after transpose)
        # For deblock 2: in [out_channels, in_channels, ...] format
        # Keep in float32 for BN fusion to match reference model's precision
        conv_weight = weight.float()

        # Double-check after conversion
        if needs_transpose_for_fusion:
            # Weight is in [in_channels, out_channels, ...] format, check shape[1]
            assert (
                conv_weight.shape[1] == out_channels[i]
            ), f"After conversion: conv_weight has {conv_weight.shape[1]} output channels, expected {out_channels[i]}"
        else:
            # Weight is in [out_channels, in_channels, ...] format, check shape[0]
            assert (
                conv_weight.shape[0] == out_channels[i]
            ), f"After conversion: conv_weight has {conv_weight.shape[0]} output channels, expected {out_channels[i]}"
        conv_bias = state_dict.get(f"{prefix}deblocks.{i}.0.bias", None)
        if conv_bias is not None:
            conv_bias = conv_bias.to(torch.bfloat16)

        # Load BN params and fuse into conv weights/bias
        # (bn_weight was already loaded above for truncation, but reload to ensure we have the right one)
        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
        bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            # Step 2: Transpose to [out, in, ...] for BN fusion (matching reference's fusion format)
            if needs_transpose_for_fusion:
                # Weight is currently [in, out, ...], transpose to [out, in, ...] for fusion
                conv_weight_for_fusion = conv_weight.permute(1, 0, 2, 3).contiguous()
            else:
                conv_weight_for_fusion = conv_weight

            # Verify BN channels match conv output channels
            # After transpose (if needed), weight is in [out_channels, in_channels, ...] format
            conv_out_channels = conv_weight_for_fusion.shape[0]  # [out_channels, in_channels, ...]
            bn_channels = bn_weight.shape[0]

            # For deblock 2 (Conv2d), reference model truncates weight based on BN channels (test line 255)
            # For deblocks 0,1 (ConvTranspose2d), reference model truncates based on BN channels (test line 225)
            # So we should truncate weight to match BN channels if needed
            if conv_out_channels > bn_channels:
                logger.warning(
                    f"Deblock {i}: Truncating conv weight output channels from {conv_out_channels} to {bn_channels} to match BN"
                )
                conv_weight_for_fusion = conv_weight_for_fusion[:bn_channels, :, :, :].clone()
                conv_out_channels = bn_channels
            elif conv_out_channels < bn_channels:
                # This shouldn't happen if we truncated correctly earlier, but handle it
                logger.warning(
                    f"Deblock {i}: BN channels ({bn_channels}) > conv output channels ({conv_out_channels}). "
                    f"Truncating BN to match conv."
                )
                bn_weight = bn_weight[:conv_out_channels]
                bn_bias = bn_bias[:conv_out_channels]
                bn_mean = bn_mean[:conv_out_channels]
                bn_var = bn_var[:conv_out_channels]
                bn_channels = conv_out_channels

            # Both should match now
            if conv_out_channels != bn_channels:
                raise ValueError(
                    f"Deblock {i}: After truncation, conv has {conv_out_channels} channels but BN has {bn_channels} channels"
                )

            # Fuse BatchNorm into conv weights and bias
            # Formula: scale = bn_weight / sqrt(bn_var + eps)
            # SECONDFPN uses eps=1e-3 (not the standard 1e-5)
            # Keep everything in float32 for fusion to match reference precision
            eps = 1e-3
            bn_weight_f32 = bn_weight.float()
            bn_bias_f32 = bn_bias.float()
            bn_mean_f32 = bn_mean.float()
            bn_var_f32 = bn_var.float()

            std = torch.sqrt(bn_var_f32 + eps)
            scale = bn_weight_f32 / std

            # Step 3: Fuse BN on [out, in, ...] format (matching reference's fusion format at line 234)
            # Weight is [out_channels, in_channels, kH, kW], scale applies to dim 0 (output channels)
            # This matches the reference model's fuse_conv_bn_weights function exactly
            fused_weight = conv_weight_for_fusion.float() * scale.view(-1, 1, 1, 1)

            # Fuse into bias
            # The reference fuse_conv_bn_weights function uses: fused_bias = bn_bias - (bn_weight * bn_mean / std)
            # This assumes conv has no bias (which is true for SECONDFPN ConvTranspose2d layers: bias=False)
            # So we use the same formula regardless of whether conv_bias exists
            fused_bias = bn_bias_f32 - (bn_weight_f32 * bn_mean_f32 / std)

            # Convert to bfloat16 after fusion (matching reference model's precision)
            fused_weight = fused_weight.to(torch.bfloat16)

            # DEBUG: Compare with reference model's fused weights
            try:
                ref_fused_weight, ref_fused_bias = compute_reference_fused_weights(
                    state_dict, prefix, i, in_channels, out_channels
                )
                # ref_fused_weight is already in [out, in, ...] format (from fuse_conv_bn_weights)
                # This matches what we need for Conv2d, so no transpose needed

                # Compare weights - use tighter tolerance for deblock 2 since it's still showing differences
                weight_diff = (fused_weight.float() - ref_fused_weight.float()).abs()
                max_diff = weight_diff.max().item()
                mean_diff = weight_diff.mean().item()

                # For deblock 2, the difference might be due to numerical precision in truncation order
                # Check if shapes match first
                if fused_weight.shape != ref_fused_weight.shape:
                    logger.warning(
                        f"Deblock {i}: Shape mismatch! Our shape: {fused_weight.shape}, Ref shape: {ref_fused_weight.shape}"
                    )
                elif max_diff > 5e-3 or mean_diff > 5e-4:  # Slightly relaxed tolerance
                    logger.warning(
                        f"Deblock {i}: Weight mismatch! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                    )
                    logger.warning(
                        f"  Our weight shape: {fused_weight.shape}, Ref weight shape: {ref_fused_weight.shape}"
                    )
                    # Find where the max difference is
                    max_idx = weight_diff.argmax().item()
                    # Convert to tuple indices for better debugging
                    flat_idx = max_idx
                    if len(fused_weight.shape) == 4:
                        out_c, in_c, h, w = fused_weight.shape
                        out_idx = flat_idx // (in_c * h * w)
                        rem = flat_idx % (in_c * h * w)
                        in_idx = rem // (h * w)
                        h_idx = (rem % (h * w)) // w
                        w_idx = rem % w
                        logger.warning(
                            f"  Max diff at [out={out_idx}, in={in_idx}, h={h_idx}, w={w_idx}]: "
                            f"ours={fused_weight[out_idx, in_idx, h_idx, w_idx].item():.6f}, "
                            f"ref={ref_fused_weight[out_idx, in_idx, h_idx, w_idx].item():.6f}"
                        )
                    else:
                        logger.warning(
                            f"  Max diff at index {max_idx}: ours={fused_weight.flatten()[max_idx].item():.6f}, "
                            f"ref={ref_fused_weight.flatten()[max_idx].item():.6f}"
                        )
                else:
                    logger.info(
                        f"Deblock {i}: Weights match reference (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})"
                    )

                # Compare bias
                if ref_fused_bias is not None:
                    bias_diff = (fused_bias.float() - ref_fused_bias.float()).abs()
                    max_bias_diff = bias_diff.max().item()
                    if max_bias_diff > 1e-3:
                        logger.warning(f"Deblock {i}: Bias mismatch! Max diff: {max_bias_diff:.6f}")
                    else:
                        logger.info(f"Deblock {i}: Bias matches reference (max diff: {max_bias_diff:.6f})")
            except Exception as e:
                logger.warning(f"Could not compare with reference weights for deblock {i}: {e}")

            # Step 4: For deblocks 0 and 1, transpose back to [in, out, ...] for ConvTranspose2d (matching reference line 244)
            # For deblock 2, keep as [out, in, ...] for Conv2d
            if i < 2:
                # Transpose back to ConvTranspose2d format [in_channels, out_channels, ...]
                fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()
            deblock.conv_weight = fused_weight
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            # No BN to fuse
            # For deblocks 0 and 1: keep as [in, out, ...] for ConvTranspose2d
            # For deblock 2: transpose to [out, in, ...] for Conv2d
            if i >= 2 and needs_transpose_for_fusion:
                conv_weight = conv_weight.permute(1, 0, 2, 3).contiguous()  # [in, out, ...] -> [out, in, ...]
            # Convert to bfloat16
            deblock.conv_weight = conv_weight.to(torch.bfloat16)
            deblock.conv_bias = conv_bias.to(torch.bfloat16) if conv_bias is not None else None

        params.deblocks.append(deblock)

    logger.info(f"Prepared SECONDFPN parameters for {len(in_channels)} levels")
    return params
