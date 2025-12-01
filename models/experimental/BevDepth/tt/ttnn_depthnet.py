# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


class BasicBlock_TTNN:
    def __init__(self, device, parameters, in_channels, out_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x, batch_size, height, width):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        identity = x

        # Input x should be allocated from reduce conv, but verify
        # Ensure tensor is properly allocated BEFORE any operations
        if not x.is_allocated():
            logger.error(
                f"Input tensor to BasicBlock is not allocated - shape: {x.shape}, sharded: {x.is_sharded()}, layout: {x.layout}"
            )
            # Try to recover by converting sharded to interleaved (this allocates)
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
                if not x.is_allocated():
                    raise RuntimeError("Tensor still not allocated after sharded_to_interleaved")
            else:
                # Not sharded and not allocated - this is a critical error
                # We cannot materialize an unallocated, non-sharded tensor
                # This indicates a bug upstream (conv2d should return allocated tensors)
                raise RuntimeError(
                    f"Input tensor to BasicBlock is not allocated and not sharded. "
                    f"Shape: {x.shape}, Layout: {x.layout}. "
                    f"This should not happen - upstream operations should return allocated tensors."
                )

        # Ensure tensor is in interleaved DRAM (not sharded) for stability
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Verify allocation before to_layout (to_layout can create unallocated views)
        if not x.is_allocated():
            logger.error("Tensor is not allocated before to_layout in BasicBlock")
            raise RuntimeError("Tensor buffer is not allocated before to_layout - cannot proceed")

        # Now safe to call to_layout (tensor is allocated)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            # Verify allocation after to_layout
            if not x.is_allocated():
                logger.error("Tensor became unallocated after to_layout")
                raise RuntimeError("Tensor buffer is not allocated after to_layout - cannot proceed")

        # Final verification: tensor should be allocated, in TILE_LAYOUT, and in DRAM
        if not x.is_allocated():
            logger.error("Tensor is still not allocated after all operations")
            raise RuntimeError("Tensor buffer is not allocated - cannot proceed")

        # Conv1: 3x3 - use BLOCK_SHARDED to avoid L1 buffer overflow
        out = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.conv1_weight,
            bias_tensor=self.params.conv1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )

        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, height, width, self.out_channels))

        # Ensure out is in DRAM before conv2
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            if out.is_sharded():
                out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        # Conv2: 3x3 (no activation) - use BLOCK_SHARDED to avoid L1 buffer overflow
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.params.conv2_weight,
            bias_tensor=self.params.conv2_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )

        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, height, width, self.out_channels))

        # Add + ReLU
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out


class ASPP_TTNN:
    def __init__(self, device, parameters, in_channels, mid_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x, batch_size, height, width):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        # Ensure input is in DRAM before conv2d
        # Avoid calling memory_config() which might fail if buffer isn't allocated
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM (from previous operations)

        # Ensure TILE_LAYOUT (required for DRAM conv)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            # After layout conversion, if it becomes sharded, convert to interleaved
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Branch 1: 1x1 conv, dilation=1 - use BLOCK_SHARDED to avoid L1 buffer overflow
        x1 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp1_weight,
            bias_tensor=self.params.aspp1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        if len(x1.shape) == 3:
            x1 = ttnn.reshape(x1, (batch_size, height, width, self.mid_channels))

        # Branch 2-4: 3x3 conv with dilation (simplified - use padding instead) - use BLOCK_SHARDED
        x2 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp2_weight,
            bias_tensor=self.params.aspp2_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(6, 6),  # dilation=6
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        if len(x2.shape) == 3:
            x2 = ttnn.reshape(x2, (batch_size, height, width, self.mid_channels))

        # Global pooling branch
        x5 = ttnn.global_avg_pool2d(x)
        # Ensure x5 is in DRAM
        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if x5.layout != ttnn.TILE_LAYOUT:
            x5 = ttnn.to_layout(x5, ttnn.TILE_LAYOUT)
            if x5.is_sharded():
                x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)
        x5 = ttnn_conv2d(
            input_tensor=x5,
            weight_tensor=self.params.global_weight,
            bias_tensor=self.params.global_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=1,
            input_width=1,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        x5 = ttnn.upsample(x5, (batch_size, height, width, self.mid_channels))

        # Concatenate (simplified - use only 3 branches)
        out = ttnn.concat([x1, x2, x5], dim=-1)

        # Ensure out is in DRAM before final conv
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            if out.is_sharded():
                out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        # Final conv - use BLOCK_SHARDED to avoid L1 buffer overflow
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.params.conv1_weight,
            bias_tensor=self.params.conv1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels * 3,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )

        return out


class DepthNet_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels=512,
        mid_channels=256,
        context_channels=512,
        depth_channels=118,
        model_config=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.params = parameters

        # Initialize sub-modules
        self.block1 = BasicBlock_TTNN(device, parameters.block1, mid_channels, mid_channels, self.model_config)
        self.block2 = BasicBlock_TTNN(device, parameters.block2, mid_channels, mid_channels, self.model_config)
        self.block3 = BasicBlock_TTNN(device, parameters.block3, mid_channels, mid_channels, self.model_config)
        self.aspp = ASPP_TTNN(device, parameters.aspp, mid_channels, mid_channels, self.model_config)

        logger.info(f"DepthNet init: in={in_channels}, mid={mid_channels}, depth={depth_channels}")

    def __call__(self, x, batch_size=1):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        height, width = x.shape[1], x.shape[2]

        # Input from test should already be in TILE_LAYOUT and DRAM_MEMORY_CONFIG
        # Only convert if absolutely necessary (if sharded)
        # Avoid unnecessary memory config/layout conversions that might create unallocated tensors
        if x.is_sharded():
            # Convert sharded to interleaved DRAM
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume input is already in the correct state (DRAM, INTERLEAVED, TILE_LAYOUT)
        # and proceed directly to conv2d

        # Reduce conv - use BLOCK_SHARDED to avoid L1 buffer overflow
        # Immediately convert sharded output to interleaved DRAM to ensure allocation
        logger.debug(
            f"Before reduce_conv: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}, x.is_sharded()={x.is_sharded()}"
        )
        x = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.reduce_weight,
            bias_tensor=self.params.reduce_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        logger.debug(
            f"After reduce_conv: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}, x.is_sharded()={x.is_sharded()}"
        )

        # Verify output shape is correct (should be mid_channels, not in_channels)
        if x.shape[-1] != self.mid_channels:
            logger.error(f"reduce_conv output shape mismatch! Expected channels={self.mid_channels}, got {x.shape[-1]}")
            logger.error(
                f"Full shape: {x.shape}, Expected: [batch_size={batch_size}, height={height}, width={width}, channels={self.mid_channels}]"
            )
            raise RuntimeError(
                f"reduce_conv output has wrong shape. Expected channels={self.mid_channels}, got {x.shape[-1]}. "
                f"Full shape: {x.shape}"
            )

        # Immediately convert sharded to interleaved DRAM to ensure proper allocation
        # This must be done before any reshape or layout operations
        # The conv2d output should be allocated, but converting sharded to interleaved
        # ensures we have a stable, allocated tensor in DRAM
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            logger.debug(f"After sharded_to_interleaved: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}")

        # Verify tensor is allocated after conversion
        # If not allocated and not sharded, this indicates a bug in conv2d or ttnn
        if not x.is_allocated():
            logger.error(f"Tensor from reduce_conv is not allocated - shape: {x.shape}, sharded: {x.is_sharded()}")
            logger.error("This indicates a bug - conv2d should always return allocated tensors")
            raise RuntimeError(
                f"Tensor buffer is not allocated after conv2d. "
                f"Shape: {x.shape}, Sharded: {x.is_sharded()}, Layout: {x.layout}. "
                f"This should not happen - conv2d output should be allocated."
            )

        # Ensure tensor is allocated BEFORE any layout operations
        # to_layout can create unallocated views if called on unallocated tensors
        if not x.is_allocated():
            logger.error("Tensor is not allocated before to_layout - this should not happen")
            raise RuntimeError("Tensor buffer is not allocated before to_layout - cannot proceed")

        # Now safe to call to_layout (tensor is allocated)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            # Verify allocation after to_layout (it should remain allocated)
            if not x.is_allocated():
                logger.error("Tensor became unallocated after to_layout - this is a bug")
                raise RuntimeError("Tensor buffer is not allocated after to_layout")

        # Reshape if needed - tensor is allocated and in TILE_LAYOUT
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
            # Reshape should not affect allocation, but verify
            if not x.is_allocated():
                logger.error("Tensor became unallocated after reshape - this is a bug")
                # If reshape created an unallocated view, we cannot materialize it
                # because to_torch requires allocation
                raise RuntimeError(
                    "Tensor buffer is not allocated after reshape. "
                    "This indicates a bug - reshape should not create unallocated views on allocated tensors."
                )

        # Final check: ensure tensor is in interleaved DRAM (not sharded)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Verify final state - tensor should be allocated, in TILE_LAYOUT, and in DRAM
        if not x.is_allocated():
            logger.error(f"Tensor is not allocated after all processing - shape: {x.shape}")
            raise RuntimeError("Tensor buffer is not allocated after all processing")

        # Context branch - x is sharded in DRAM from reduce conv
        # Pass it directly to conv2d which can handle sharded inputs
        context = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.context_weight,
            bias_tensor=self.params.context_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels,
            out_channels=self.context_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )

        if len(context.shape) == 3:
            context = ttnn.reshape(context, (batch_size, height, width, self.context_channels))

        # Depth branch: verify x is allocated before passing to block1
        logger.debug(
            f"Before block1: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}, x.is_sharded()={x.is_sharded()}, expected_channels={self.mid_channels}"
        )

        # Verify shape is correct (should be mid_channels after reduce_conv)
        if x.shape[-1] != self.mid_channels:
            logger.error(
                f"Tensor x has wrong shape before block1! Expected channels={self.mid_channels}, got {x.shape[-1]}"
            )
            logger.error(f"Full shape: {x.shape}. This suggests reduce_conv did not update x correctly.")
            raise RuntimeError(
                f"Tensor x has wrong shape before block1. Expected channels={self.mid_channels}, got {x.shape[-1]}. "
                f"Full shape: {x.shape}. This indicates reduce_conv did not update the tensor correctly."
            )

        if not x.is_allocated():
            logger.error(f"Tensor x is not allocated before block1 - shape: {x.shape}, sharded: {x.is_sharded()}")
            # Try to fix by converting sharded to interleaved if sharded
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
                logger.debug(
                    f"After sharded_to_interleaved before block1: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}"
                )
            else:
                # Not sharded and not allocated - this shouldn't happen
                raise RuntimeError(
                    f"Tensor x is not allocated and not sharded before block1. "
                    f"Shape: {x.shape}, Expected channels: {self.mid_channels}. "
                    f"This indicates a bug in the processing pipeline."
                )

        # Ensure x is in TILE_LAYOUT before passing to block1
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        depth = self.block1(x, batch_size, height, width)
        depth = self.block2(depth, batch_size, height, width)
        depth = self.block3(depth, batch_size, height, width)

        # ASPP
        depth = self.aspp(depth, batch_size, height, width)

        # Ensure depth is in DRAM before DCN conv
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)
            if depth.is_sharded():
                depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)

        # DCN (Deformable Conv) - using torchvision's deform_conv2d (no compiled extensions needed)
        # Similar approach to uniad/vadv2: convert to PyTorch, run deform_conv2d, convert back
        try:
            from torchvision.ops import deform_conv2d as tv_deform_conv2d

            # Convert TTNN tensor to PyTorch (NCHW format)
            depth_torch = ttnn.to_torch(depth)
            # Handle different tensor formats
            if len(depth_torch.shape) == 4:
                if depth_torch.shape[1] == 1 and depth_torch.shape[2] == height * width:
                    # Flattened format: [B, 1, H*W, C] -> [B, H, W, C]
                    depth_torch = depth_torch.reshape(batch_size, height, width, self.mid_channels)
                elif depth_torch.shape[1] == height and depth_torch.shape[2] == width:
                    # Already in [B, H, W, C] format
                    pass
                else:
                    # Try to infer from total elements
                    total_elements = depth_torch.numel()
                    expected_elements = batch_size * height * width * self.mid_channels
                    if total_elements == expected_elements:
                        depth_torch = depth_torch.reshape(batch_size, height, width, self.mid_channels)
            # Convert from [B, H, W, C] to [B, C, H, W] for PyTorch
            depth_torch = depth_torch.permute(0, 3, 1, 2).contiguous().float()

            # Generate offset using conv_offset layer (similar to DeformConv2dPack)
            offset = self.params.dcn_conv_offset(depth_torch)

            # Run torchvision's deform_conv2d
            # torchvision uses [x, y] order for offsets, which matches our conv_offset output
            depth_torch = tv_deform_conv2d(
                input=depth_torch.float(),
                offset=offset.float(),
                weight=self.params.dcn_weight.float(),
                bias=self.params.dcn_bias.float() if self.params.dcn_bias is not None else None,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )

            # Convert back to TTNN format [B, H, W, C]
            depth_torch = depth_torch.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
            depth_torch = depth_torch.reshape(1, 1, batch_size * height * width, self.mid_channels)

            # Convert back to TTNN tensor
            depth = ttnn.from_torch(
                depth_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except ImportError:
            # Fallback to regular grouped conv if torchvision not available
            logger.warning("torchvision.ops.deform_conv2d not available, using regular Conv2d as approximation")
            depth = ttnn_conv2d(
                input_tensor=depth,
                weight_tensor=self.params.dcn_weight,
                bias_tensor=self.params.dcn_bias,
                device=self.device,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation=None,
                math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
                weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )

        if len(depth.shape) == 3:
            depth = ttnn.reshape(depth, (batch_size, height, width, self.mid_channels))

        # Ensure depth is in DRAM before final conv
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)
            if depth.is_sharded():
                depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)

        # Final depth conv - use BLOCK_SHARDED to avoid L1 buffer overflow
        depth = ttnn_conv2d(
            input_tensor=depth,
            weight_tensor=self.params.final_weight,
            bias_tensor=self.params.final_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels,
            out_channels=self.depth_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )

        if len(depth.shape) == 3:
            depth = ttnn.reshape(depth, (batch_size, height, width, self.depth_channels))

        # Concatenate depth and context
        out = ttnn.concat([depth, context], dim=-1)

        return out


def prepare_depthnet_parameters(state_dict, in_channels=512, mid_channels=256, depth_channels=118):
    class Parameters:
        pass

    params = Parameters()

    # Find the actual prefix used in this checkpoint
    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.depth_net.",
        "img_backbone.depth_net.",
        "backbone.depth_net.",
        "depth_net.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        # No depth_net found, return the full state dict for debugging
        logger.error(f"Could not find depth_net prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No depth_net keys found in checkpoint")

    logger.info(f"Using DepthNet prefix: {prefix}")

    # Reduce conv
    try:
        params.reduce_weight = state_dict[f"{prefix}reduce_conv.0.weight"].to(torch.bfloat16)
        params.reduce_bias = state_dict.get(f"{prefix}reduce_conv.0.bias", None)
        if params.reduce_bias is not None:
            params.reduce_bias = params.reduce_bias.to(torch.bfloat16)
    except KeyError as e:
        logger.error(f"Failed to load reduce_conv: {e}")
        logger.info(f"Available depth_net keys: {[k for k in all_keys if prefix in k][:20]}")
        raise

    # Context conv
    params.context_weight = state_dict[f"{prefix}context_conv.weight"].to(torch.bfloat16)
    params.context_bias = state_dict.get(f"{prefix}context_conv.bias", None)
    if params.context_bias is not None:
        params.context_bias = params.context_bias.to(torch.bfloat16)

    # BasicBlocks (depth_conv.0, depth_conv.1, depth_conv.2)
    for i in range(3):
        block = Parameters()
        block.conv1_weight = state_dict[f"{prefix}depth_conv.{i}.conv1.weight"].to(torch.bfloat16)
        block.conv1_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv1.bias", None)
        block.conv2_weight = state_dict[f"{prefix}depth_conv.{i}.conv2.weight"].to(torch.bfloat16)
        block.conv2_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv2.bias", None)
        if block.conv1_bias is not None:
            block.conv1_bias = block.conv1_bias.to(torch.bfloat16)
        if block.conv2_bias is not None:
            block.conv2_bias = block.conv2_bias.to(torch.bfloat16)
        setattr(params, f"block{i+1}", block)

    # ASPP (depth_conv.3)
    params.aspp = Parameters()
    params.aspp.aspp1_weight = state_dict[f"{prefix}depth_conv.3.aspp1.atrous_conv.weight"].to(torch.bfloat16)
    params.aspp.aspp1_bias = state_dict.get(f"{prefix}depth_conv.3.aspp1.atrous_conv.bias", None)
    params.aspp.aspp2_weight = state_dict[f"{prefix}depth_conv.3.aspp2.atrous_conv.weight"].to(torch.bfloat16)
    params.aspp.aspp2_bias = state_dict.get(f"{prefix}depth_conv.3.aspp2.atrous_conv.bias", None)
    params.aspp.global_weight = state_dict[f"{prefix}depth_conv.3.global_avg_pool.1.weight"].to(torch.bfloat16)
    params.aspp.global_bias = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.1.bias", None)
    params.aspp.conv1_weight = state_dict[f"{prefix}depth_conv.3.conv1.weight"].to(torch.bfloat16)
    params.aspp.conv1_bias = state_dict.get(f"{prefix}depth_conv.3.conv1.bias", None)

    # DCN layer (depth_conv.4) - DeformConv2dPack has both weight and conv_offset
    params.dcn_weight = state_dict[f"{prefix}depth_conv.4.weight"].to(torch.bfloat16)
    params.dcn_bias = state_dict.get(f"{prefix}depth_conv.4.bias", None)
    if params.dcn_bias is not None:
        params.dcn_bias = params.dcn_bias.to(torch.bfloat16)

    # Load conv_offset layer (for DeformConv2dPack)
    # Offset shape: (deform_groups * 2 * kernel_size[0] * kernel_size[1], in_channels, kernel_size[0], kernel_size[1])
    # For DCN with groups=4, kernel=3: offset_channels = 1 * 2 * 3 * 3 = 18
    # But DeformConv2dPack uses deform_groups=1 by default, so offset_channels = 1 * 2 * 3 * 3 = 18
    try:
        conv_offset_weight = state_dict[f"{prefix}depth_conv.4.conv_offset.weight"]
        conv_offset_bias = state_dict.get(f"{prefix}depth_conv.4.conv_offset.bias", None)

        # Create a PyTorch Conv2d layer for offset generation
        # This will be used to generate offsets from input features
        offset_channels = conv_offset_weight.shape[0]  # Should be 18 for kernel=3, deform_groups=1
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_offset_bias is not None,
        )
        params.dcn_conv_offset.weight.data = conv_offset_weight
        if conv_offset_bias is not None:
            params.dcn_conv_offset.bias.data = conv_offset_bias
        params.dcn_conv_offset.eval()  # Set to eval mode
        logger.info(f"Loaded DCN conv_offset layer: {offset_channels} offset channels")
    except KeyError:
        logger.warning(f"conv_offset not found for depth_conv.4, DCN will use zero offsets (may reduce accuracy)")
        # Create a dummy conv_offset that outputs zeros
        offset_channels = 18  # 2 * 3 * 3 for kernel=3
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        params.dcn_conv_offset.weight.data.zero_()
        params.dcn_conv_offset.bias.data.zero_()
        params.dcn_conv_offset.eval()

    # Final conv (depth_conv.5)
    params.final_weight = state_dict[f"{prefix}depth_conv.5.weight"].to(torch.bfloat16)
    params.final_bias = state_dict.get(f"{prefix}depth_conv.5.bias", None)
    if params.final_bias is not None:
        params.final_bias = params.final_bias.to(torch.bfloat16)

    logger.info("Successfully prepared DepthNet parameters")
    return params
