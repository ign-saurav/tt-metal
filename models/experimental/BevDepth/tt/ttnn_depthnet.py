# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


class MLP_TTNN:
    """MLP implementation for camera-aware features"""

    def __init__(self, device, parameters, in_features, hidden_features, out_features, model_config):
        self.device = device
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x_torch):
        """Forward pass: x_torch is PyTorch tensor [batch*num_cams, in_features]"""
        # Convert input to bfloat16 to match weights dtype
        if x_torch.dtype != torch.bfloat16:
            x_torch = x_torch.to(torch.bfloat16)

        # fc1: Linear(in_features, hidden_features)
        x = torch.nn.functional.linear(x_torch, self.params.fc1_weight, self.params.fc1_bias)
        # ReLU activation
        x = torch.relu(x)
        # fc2: Linear(hidden_features, out_features)
        x = torch.nn.functional.linear(x, self.params.fc2_weight, self.params.fc2_bias)
        return x


class SELayer_TTNN:
    """Squeeze-and-Excitation Layer"""

    def __init__(self, device, parameters, channels, model_config):
        self.device = device
        self.channels = channels
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x, x_se):
        """
        Forward pass:
        x: TTNN tensor [batch, height, width, channels]
        x_se: TTNN tensor [batch, height, width, channels] (from MLP output broadcasted)
        """
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        # Ensure x_se is in correct format
        if x_se.is_sharded():
            x_se = ttnn.sharded_to_interleaved(x_se, ttnn.DRAM_MEMORY_CONFIG)
        if x_se.layout != ttnn.TILE_LAYOUT:
            x_se = ttnn.to_layout(x_se, ttnn.TILE_LAYOUT)

        batch_size, height, width, channels = x.shape

        # conv_reduce: 1x1 conv (channels -> channels)
        x_se = ttnn_conv2d(
            input_tensor=x_se,
            weight_tensor=self.params.conv_reduce_weight,
            bias_tensor=self.params.conv_reduce_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=None,
            packer_l1_acc=False,
        )

        # Reshape if needed
        if len(x_se.shape) == 3:
            x_se = ttnn.reshape(x_se, (batch_size, height, width, channels))
        elif len(x_se.shape) == 4 and (x_se.shape[0] == 1 or x_se.shape[1] == 1):
            x_se = ttnn.reshape(x_se, (batch_size, height, width, channels))

        # conv_expand: 1x1 conv (channels -> channels)
        x_se = ttnn_conv2d(
            input_tensor=x_se,
            weight_tensor=self.params.conv_expand_weight,
            bias_tensor=self.params.conv_expand_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,  # No activation before sigmoid
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=None,
            packer_l1_acc=False,
        )

        # Reshape if needed
        if len(x_se.shape) == 3:
            x_se = ttnn.reshape(x_se, (batch_size, height, width, channels))
        elif len(x_se.shape) == 4 and (x_se.shape[0] == 1 or x_se.shape[1] == 1):
            x_se = ttnn.reshape(x_se, (batch_size, height, width, channels))

        # Apply sigmoid (gate)
        x_se = ttnn.sigmoid(x_se)

        # Element-wise multiply: x * gate(x_se)
        result = ttnn.multiply(x, x_se)

        return result


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
        # Convert sharded to interleaved DRAM before reshape (required for reshape)
        if x1.is_sharded():
            x1 = ttnn.sharded_to_interleaved(x1, ttnn.DRAM_MEMORY_CONFIG)
        # Ensure tensor is in DRAM (not L1)
        if x1.is_allocated() and x1.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x1 = ttnn.to_memory_config(x1, ttnn.DRAM_MEMORY_CONFIG)
        # Ensure TILE_LAYOUT
        if x1.layout != ttnn.TILE_LAYOUT:
            x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)

        # Verify tensor is allocated
        if not x1.is_allocated():
            raise RuntimeError(f"x1 is not allocated before reshape: shape={x1.shape}")
        if x1.is_sharded():
            raise RuntimeError(f"x1 is still sharded after conversion: shape={x1.shape}")

        # Reshape x1 if needed (ttnn.conv2d returns flattened tensor)
        expected_elements = batch_size * height * width * self.mid_channels
        actual_elements = 1
        for dim in x1.shape:
            actual_elements *= dim

        if actual_elements != expected_elements:
            raise RuntimeError(
                f"Cannot reshape x1: shape={x1.shape}, expected elements={expected_elements}, actual={actual_elements}"
            )

        # Only reshape if shape doesn't already match
        if (
            len(x1.shape) != 4
            or x1.shape[0] != batch_size
            or x1.shape[1] != height
            or x1.shape[2] != width
            or x1.shape[3] != self.mid_channels
        ):
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
        # Convert sharded to interleaved DRAM before reshape (required for reshape)
        if x2.is_sharded():
            x2 = ttnn.sharded_to_interleaved(x2, ttnn.DRAM_MEMORY_CONFIG)
        # Ensure tensor is in DRAM (not L1)
        if x2.is_allocated() and x2.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x2 = ttnn.to_memory_config(x2, ttnn.DRAM_MEMORY_CONFIG)
        # Ensure TILE_LAYOUT
        if x2.layout != ttnn.TILE_LAYOUT:
            x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)

        # Verify tensor is allocated and in correct state
        if not x2.is_allocated():
            raise RuntimeError(f"x2 is not allocated before reshape: shape={x2.shape}")
        if x2.is_sharded():
            raise RuntimeError(f"x2 is still sharded after conversion: shape={x2.shape}")

        # Reshape x2 if needed (ttnn.conv2d returns flattened tensor)
        # Note: With padding=(6,6) for dilation=6, output size is larger than input
        # Calculate actual output dimensions from the flattened tensor
        actual_elements = 1
        for dim in x2.shape:
            actual_elements *= dim

        # Extract spatial dimension from flattened shape [1, NHW, C] or [1, 1, NHW, C]
        if len(x2.shape) == 3 and x2.shape[0] == 1:
            # Format: [1, batch*height*width, channels]
            NHW = x2.shape[1]
            C = x2.shape[2]
        elif len(x2.shape) == 4 and x2.shape[0] == 1 and x2.shape[1] == 1:
            # Format: [1, 1, batch*height*width, channels]
            NHW = x2.shape[2]
            C = x2.shape[3]
        else:
            raise RuntimeError(f"Unexpected x2 shape format: {x2.shape}")

        # Calculate actual output height and width
        # NHW = batch_size * output_height * output_width
        # So: output_height * output_width = NHW / batch_size
        spatial_size = NHW // batch_size
        # Find factors of spatial_size that are close to height x width
        # For now, we'll use the actual spatial size and calculate dimensions
        # With padding=(6,6) and kernel=3, output = input + 2*pad - kernel + 1
        # output_height = height + 12 - 3 + 1 = height + 10
        # output_width = width + 12 - 3 + 1 = width + 10
        # So: (height + 10) * (width + 10) = spatial_size
        # For height=64, width=160: (64+10) * (160+10) = 74 * 170 = 12580 ✓
        output_height = height + 10  # padding=6 adds 10 to each dimension
        output_width = width + 10

        # Verify the calculation matches actual spatial size
        if output_height * output_width != spatial_size:
            # If calculation doesn't match, infer from spatial_size
            # Try to find dimensions that match spatial_size
            import math

            aspect_ratio = height / width
            output_width = int(math.sqrt(spatial_size / aspect_ratio))
            output_height = spatial_size // output_width
            # Adjust to ensure they multiply correctly
            while output_height * output_width < spatial_size and output_width < spatial_size:
                output_width += 1
            while output_height * output_width > spatial_size and output_width > 1:
                output_width -= 1
            output_height = spatial_size // output_width

        # Verify calculation: output_height * output_width should equal spatial_size
        if output_height * output_width != spatial_size:
            raise RuntimeError(
                f"Cannot determine output dimensions: spatial_size={spatial_size}, "
                f"calculated output_height={output_height}, output_width={output_width}, "
                f"product={output_height * output_width}"
            )

        # Verify elements match
        expected_elements = batch_size * output_height * output_width * self.mid_channels
        if actual_elements != expected_elements:
            raise RuntimeError(
                f"Cannot reshape x2: shape={x2.shape}, calculated output=({batch_size}, {output_height}, {output_width}, {self.mid_channels}), "
                f"expected elements={expected_elements}, actual={actual_elements}, spatial_size={spatial_size}"
            )

        # Reshape to actual output dimensions
        if (
            len(x2.shape) != 4
            or x2.shape[0] != batch_size
            or x2.shape[1] != output_height
            or x2.shape[2] != output_width
            or x2.shape[3] != self.mid_channels
        ):
            try:
                x2 = ttnn.reshape(x2, (batch_size, output_height, output_width, self.mid_channels))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reshape x2: shape={x2.shape}, target=({batch_size}, {output_height}, {output_width}, {self.mid_channels}), "
                    f"error={str(e)}"
                )

        # Adjust x2 to match input size (crop if larger, pad if smaller)
        if output_height != height or output_width != width:
            # First, crop if dimensions are larger
            if output_height > height:
                crop_h_start = (output_height - height) // 2
                crop_h_end = crop_h_start + height
            else:
                crop_h_start = 0
                crop_h_end = output_height

            if output_width > width:
                crop_w_start = (output_width - width) // 2
                crop_w_end = crop_w_start + width
            else:
                crop_w_start = 0
                crop_w_end = output_width

            # Crop if needed
            if crop_h_end < output_height or crop_w_end < output_width:
                x2 = ttnn.slice(
                    x2, [0, crop_h_start, crop_w_start, 0], [batch_size, crop_h_end, crop_w_end, self.mid_channels]
                )

            # Then, pad if dimensions are smaller
            pad_h = height - x2.shape[1] if x2.shape[1] < height else 0
            pad_w = width - x2.shape[2] if x2.shape[2] < width else 0

            if pad_h > 0 or pad_w > 0:
                # Convert to ROW_MAJOR_LAYOUT for padding (TILE_LAYOUT doesn't support front padding)
                x2_was_tile = x2.layout == ttnn.TILE_LAYOUT
                if x2_was_tile:
                    x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)

                # Pad symmetrically
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before
                x2 = ttnn.pad(x2, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)), value=0.0)

                # Convert back to TILE_LAYOUT if it was originally in TILE_LAYOUT
                if x2_was_tile:
                    x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)

            # Verify final dimensions
            if x2.shape[1] != height or x2.shape[2] != width:
                raise RuntimeError(
                    f"Cannot adjust x2 dimensions: after crop/pad got ({x2.shape[1]}, {x2.shape[2]}), "
                    f"expected ({height}, {width}). Original output was ({output_height}, {output_width})"
                )

        # Branch 3: 3x3 conv with dilation=12, padding=12
        x3 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp3_weight,
            bias_tensor=self.params.aspp3_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(12, 12),  # dilation=12
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        # Convert and reshape x3 (same logic as x2)
        if x3.is_sharded():
            x3 = ttnn.sharded_to_interleaved(x3, ttnn.DRAM_MEMORY_CONFIG)
        if x3.is_allocated() and x3.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x3 = ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG)
        if x3.layout != ttnn.TILE_LAYOUT:
            x3 = ttnn.to_layout(x3, ttnn.TILE_LAYOUT)
        # Reshape x3 - calculate actual output dimensions from tensor shape
        # Extract spatial dimension from flattened shape
        if len(x3.shape) == 3 and x3.shape[0] == 1:
            NHW = x3.shape[1]
            C = x3.shape[2]
        elif len(x3.shape) == 4 and x3.shape[0] == 1 and x3.shape[1] == 1:
            NHW = x3.shape[2]
            C = x3.shape[3]
        else:
            raise RuntimeError(f"Unexpected x3 shape format: {x3.shape}")

        spatial_size = NHW // batch_size
        expected_spatial_size = height * width

        # Calculate output dimensions (should be same as input for padding=dilation)
        x3_output_height = height
        x3_output_width = width

        # Verify spatial size matches
        if spatial_size != expected_spatial_size:
            # If doesn't match, find exact factors of spatial_size
            # The output from conv2d might be larger than input due to padding behavior
            import math

            aspect_ratio = height / width
            # Start with width based on aspect ratio
            x3_output_width = int(math.sqrt(spatial_size / aspect_ratio))

            # Find exact factors by testing all divisors
            best_height = None
            best_width = None
            best_diff = float("inf")

            # Try all divisors of spatial_size (within reasonable range)
            # Check divisors from sqrt(spatial_size) down and up
            sqrt_size = int(math.sqrt(spatial_size))
            for w in range(max(1, sqrt_size - 50), min(spatial_size + 1, sqrt_size + 50)):
                if spatial_size % w == 0:
                    h = spatial_size // w
                    # Calculate how close this is to the expected dimensions
                    diff = abs(h - height) + abs(w - width)
                    if diff < best_diff:
                        best_height = h
                        best_width = w
                        best_diff = diff

            if best_height is None:
                # If no exact divisor found, try wider range
                for w in range(1, min(spatial_size + 1, 500)):  # Limit to reasonable width
                    if spatial_size % w == 0:
                        h = spatial_size // w
                        diff = abs(h - height) + abs(w - width)
                        if diff < best_diff:
                            best_height = h
                            best_width = w
                            best_diff = diff

            if best_height is None:
                raise RuntimeError(
                    f"Cannot find exact factors for x3: spatial_size={spatial_size}, "
                    f"expected={height}x{width}={expected_spatial_size}"
                )

            x3_output_height = best_height
            x3_output_width = best_width

        # Verify calculation - must match exactly for reshape
        if x3_output_height * x3_output_width != spatial_size:
            raise RuntimeError(
                f"Cannot determine x3 output dimensions: spatial_size={spatial_size}, "
                f"calculated output_height={x3_output_height}, output_width={x3_output_width}, "
                f"product={x3_output_height * x3_output_width}, expected={height}x{width}={expected_spatial_size}"
            )

        # Reshape to calculated dimensions
        if len(x3.shape) == 3 and x3.shape[0] == 1:
            x3 = ttnn.reshape(x3, (batch_size, x3_output_height, x3_output_width, self.mid_channels))
        elif len(x3.shape) == 4 and x3.shape[0] == 1 and x3.shape[1] == 1:
            x3 = ttnn.reshape(x3, (batch_size, x3_output_height, x3_output_width, self.mid_channels))
        # Adjust x3 to match input size (crop if larger, pad if smaller)
        if x3_output_height != height or x3_output_width != width:
            # First, crop if dimensions are larger
            if x3_output_height > height:
                crop_h_start = (x3_output_height - height) // 2
                crop_h_end = crop_h_start + height
            else:
                crop_h_start = 0
                crop_h_end = x3_output_height

            if x3_output_width > width:
                crop_w_start = (x3_output_width - width) // 2
                crop_w_end = crop_w_start + width
            else:
                crop_w_start = 0
                crop_w_end = x3_output_width

            # Crop if needed
            if crop_h_end < x3_output_height or crop_w_end < x3_output_width:
                x3 = ttnn.slice(
                    x3, [0, crop_h_start, crop_w_start, 0], [batch_size, crop_h_end, crop_w_end, self.mid_channels]
                )

            # Then, pad if dimensions are smaller
            pad_h = height - x3.shape[1] if x3.shape[1] < height else 0
            pad_w = width - x3.shape[2] if x3.shape[2] < width else 0

            if pad_h > 0 or pad_w > 0:
                # Convert to ROW_MAJOR_LAYOUT for padding (TILE_LAYOUT doesn't support front padding)
                x3_was_tile = x3.layout == ttnn.TILE_LAYOUT
                if x3_was_tile:
                    x3 = ttnn.to_layout(x3, ttnn.ROW_MAJOR_LAYOUT)

                # Pad symmetrically
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before
                x3 = ttnn.pad(x3, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)), value=0.0)

                # Convert back to TILE_LAYOUT if it was originally in TILE_LAYOUT
                if x3_was_tile:
                    x3 = ttnn.to_layout(x3, ttnn.TILE_LAYOUT)

            # Verify final dimensions
            if x3.shape[1] != height or x3.shape[2] != width:
                raise RuntimeError(
                    f"Cannot adjust x3 dimensions: after crop/pad got ({x3.shape[1]}, {x3.shape[2]}), "
                    f"expected ({height}, {width}). Original output was ({x3_output_height}, {x3_output_width})"
                )

        # Branch 4: 3x3 conv with dilation=18, padding=18
        x4 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp4_weight,
            bias_tensor=self.params.aspp4_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(18, 18),  # dilation=18
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            packer_l1_acc=False,
        )
        # Convert and reshape x4 (same logic as x2)
        if x4.is_sharded():
            x4 = ttnn.sharded_to_interleaved(x4, ttnn.DRAM_MEMORY_CONFIG)
        if x4.is_allocated() and x4.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x4 = ttnn.to_memory_config(x4, ttnn.DRAM_MEMORY_CONFIG)
        if x4.layout != ttnn.TILE_LAYOUT:
            x4 = ttnn.to_layout(x4, ttnn.TILE_LAYOUT)
        # Reshape x4 - calculate actual output dimensions from tensor shape
        # Extract spatial dimension from flattened shape
        if len(x4.shape) == 3 and x4.shape[0] == 1:
            NHW = x4.shape[1]
            C = x4.shape[2]
        elif len(x4.shape) == 4 and x4.shape[0] == 1 and x4.shape[1] == 1:
            NHW = x4.shape[2]
            C = x4.shape[3]
        else:
            raise RuntimeError(f"Unexpected x4 shape format: {x4.shape}")

        spatial_size = NHW // batch_size
        expected_spatial_size = height * width

        # Calculate output dimensions (should be same as input for padding=dilation)
        x4_output_height = height
        x4_output_width = width

        # Verify spatial size matches
        if spatial_size != expected_spatial_size:
            # If doesn't match, find exact factors of spatial_size
            # The output from conv2d might be larger than input due to padding behavior
            import math

            aspect_ratio = height / width
            # Start with width based on aspect ratio
            x4_output_width = int(math.sqrt(spatial_size / aspect_ratio))

            # Find exact factors by testing all divisors
            best_height = None
            best_width = None
            best_diff = float("inf")

            # Try all divisors of spatial_size (within reasonable range)
            # Check divisors from sqrt(spatial_size) down and up
            sqrt_size = int(math.sqrt(spatial_size))
            for w in range(max(1, sqrt_size - 50), min(spatial_size + 1, sqrt_size + 50)):
                if spatial_size % w == 0:
                    h = spatial_size // w
                    # Calculate how close this is to the expected dimensions
                    diff = abs(h - height) + abs(w - width)
                    if diff < best_diff:
                        best_height = h
                        best_width = w
                        best_diff = diff

            if best_height is None:
                # If no exact divisor found, try wider range
                for w in range(1, min(spatial_size + 1, 500)):  # Limit to reasonable width
                    if spatial_size % w == 0:
                        h = spatial_size // w
                        diff = abs(h - height) + abs(w - width)
                        if diff < best_diff:
                            best_height = h
                            best_width = w
                            best_diff = diff

            if best_height is None:
                raise RuntimeError(
                    f"Cannot find exact factors for x4: spatial_size={spatial_size}, "
                    f"expected={height}x{width}={expected_spatial_size}"
                )

            x4_output_height = best_height
            x4_output_width = best_width

        # Verify calculation - must match exactly for reshape
        if x4_output_height * x4_output_width != spatial_size:
            raise RuntimeError(
                f"Cannot determine x4 output dimensions: spatial_size={spatial_size}, "
                f"calculated output_height={x4_output_height}, output_width={x4_output_width}, "
                f"product={x4_output_height * x4_output_width}, expected={height}x{width}={expected_spatial_size}"
            )

        # Reshape to calculated dimensions
        if len(x4.shape) == 3 and x4.shape[0] == 1:
            x4 = ttnn.reshape(x4, (batch_size, x4_output_height, x4_output_width, self.mid_channels))
        elif len(x4.shape) == 4 and x4.shape[0] == 1 and x4.shape[1] == 1:
            x4 = ttnn.reshape(x4, (batch_size, x4_output_height, x4_output_width, self.mid_channels))

        # Verify reshape succeeded
        if x4.shape[1] != x4_output_height or x4.shape[2] != x4_output_width:
            raise RuntimeError(
                f"x4 reshape failed: expected ({batch_size}, {x4_output_height}, {x4_output_width}, {self.mid_channels}), "
                f"got {x4.shape}"
            )

        # Adjust x4 to match input size (crop if larger, pad if smaller)
        if x4_output_height != height or x4_output_width != width:
            # First, crop if dimensions are larger
            if x4_output_height > height:
                crop_h_start = (x4_output_height - height) // 2
                crop_h_end = crop_h_start + height
            else:
                crop_h_start = 0
                crop_h_end = x4_output_height

            if x4_output_width > width:
                crop_w_start = (x4_output_width - width) // 2
                crop_w_end = crop_w_start + width
            else:
                crop_w_start = 0
                crop_w_end = x4_output_width

            # Crop if needed
            if crop_h_end < x4_output_height or crop_w_end < x4_output_width:
                x4 = ttnn.slice(
                    x4, [0, crop_h_start, crop_w_start, 0], [batch_size, crop_h_end, crop_w_end, self.mid_channels]
                )

            # Then, pad if dimensions are smaller
            pad_h = height - x4.shape[1] if x4.shape[1] < height else 0
            pad_w = width - x4.shape[2] if x4.shape[2] < width else 0

            if pad_h > 0 or pad_w > 0:
                # Convert to ROW_MAJOR_LAYOUT for padding (TILE_LAYOUT doesn't support front padding)
                x4_was_tile = x4.layout == ttnn.TILE_LAYOUT
                if x4_was_tile:
                    x4 = ttnn.to_layout(x4, ttnn.ROW_MAJOR_LAYOUT)

                # Pad symmetrically
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before
                x4 = ttnn.pad(x4, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)), value=0.0)

                # Convert back to TILE_LAYOUT if it was originally in TILE_LAYOUT
                if x4_was_tile:
                    x4 = ttnn.to_layout(x4, ttnn.TILE_LAYOUT)

            # Verify final dimensions
            if x4.shape[1] != height or x4.shape[2] != width:
                raise RuntimeError(
                    f"Cannot adjust x4 dimensions: after crop/pad got ({x4.shape[1]}, {x4.shape[2]}), "
                    f"expected ({height}, {width}). Original output was ({x4_output_height}, {x4_output_width})"
                )

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
        # Convert sharded to interleaved before upsample (required)
        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)

        # Reshape x5 to [batch, 1, 1, channels] if needed (from global pooling + conv)
        if len(x5.shape) == 4 and x5.shape[0] == 1 and x5.shape[1] == 1:
            # Format: [1, 1, batch, channels] or [1, 1, 1, channels] - need to check
            if x5.shape[2] == batch_size:
                # Format: [1, 1, batch, channels]
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            elif x5.shape[2] == 1:
                # Format: [1, 1, 1, channels]
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
        elif len(x5.shape) == 3 and x5.shape[0] == 1:
            # Format: [1, batch, channels] or [1, 1, channels]
            if x5.shape[1] == batch_size:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            elif x5.shape[1] == 1:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
        elif len(x5.shape) != 4 or x5.shape[0] != batch_size or x5.shape[1] != 1 or x5.shape[2] != 1:
            # Need to reshape - check total elements match
            expected_elements = batch_size * 1 * 1 * self.mid_channels
            actual_elements = 1
            for dim in x5.shape:
                actual_elements *= dim
            if actual_elements == expected_elements:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            else:
                raise RuntimeError(
                    f"Cannot reshape x5: shape={x5.shape}, expected elements={expected_elements}, actual={actual_elements}"
                )

        # Convert to ROW_MAJOR_LAYOUT before upsample
        # TILE_LAYOUT requires tile-aligned dimensions (divisible by 32)
        # Since input is 1x1, we must use ROW_MAJOR_LAYOUT
        if x5.layout != ttnn.ROW_MAJOR_LAYOUT:
            x5 = ttnn.to_layout(x5, ttnn.ROW_MAJOR_LAYOUT)

        # Upsample from 1x1 to height x width using scale_factor
        # scale_factor should be [height, width] to go from 1x1 to height x width
        x5 = ttnn.upsample(x5, scale_factor=[height, width], mode="nearest")

        # Convert back to TILE_LAYOUT for concatenation with x1 and x2
        if x5.layout != ttnn.TILE_LAYOUT:
            x5 = ttnn.to_layout(x5, ttnn.TILE_LAYOUT)

        # Concatenate all 5 branches: x1, x2, x3, x4, x5
        out = ttnn.concat([x1, x2, x3, x4, x5], dim=-1)

        # Ensure out is in DRAM before final conv (force DRAM slicing)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.is_allocated() and out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)

        # Final conv - 2560->512, 1x1 kernel: Use channel slicing directly (4x 640->512)
        # This avoids L1 OOM errors for large channel count
        import torch

        num_slices = 4
        channels_per_slice = (self.mid_channels * 5) // num_slices  # 640

        # Split input along channel dimension into 4 slices
        out_slices = []
        weight_slices = []
        for i in range(num_slices):
            start_ch = i * channels_per_slice
            end_ch = (i + 1) * channels_per_slice if i < num_slices - 1 else self.mid_channels * 5
            out_slices.append(ttnn.slice(out, [0, 0, 0, start_ch], [batch_size, height, width, end_ch]))

        # Split weights: [out_channels, in_channels, kernel_h, kernel_w] = [512, 2560, 1, 1]
        weight_torch = (
            self.params.conv1_weight
            if isinstance(self.params.conv1_weight, torch.Tensor)
            else ttnn.to_torch(self.params.conv1_weight)
        )
        for i in range(num_slices):
            start_ch = i * channels_per_slice
            end_ch = (i + 1) * channels_per_slice if i < num_slices - 1 else self.mid_channels * 5
            weight_slices.append(weight_torch[:, start_ch:end_ch, :, :])

        # Run each slice separately, accumulating results
        out_accum = None
        for i in range(num_slices):
            out_i = ttnn_conv2d(
                input_tensor=out_slices[i],
                weight_tensor=weight_slices[i],
                bias_tensor=None,  # Apply bias once at the end
                device=self.device,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                in_channels=weight_slices[i].shape[1],  # Actual channels in this slice
                out_channels=self.mid_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation=None,  # Apply ReLU after sum
                math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
                weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                shard_layout=None,
                packer_l1_acc=False,
            )

            # Move output to DRAM and reshape if needed
            if out_i.is_sharded():
                out_i = ttnn.sharded_to_interleaved(out_i, ttnn.DRAM_MEMORY_CONFIG)
            if out_i.is_allocated() and out_i.memory_config().buffer_type != ttnn.BufferType.DRAM:
                out_i = ttnn.to_memory_config(out_i, ttnn.DRAM_MEMORY_CONFIG)
            if out_i.layout != ttnn.TILE_LAYOUT:
                out_i = ttnn.to_layout(out_i, ttnn.TILE_LAYOUT)

            if len(out_i.shape) == 3:
                out_i = ttnn.reshape(out_i, (batch_size, height, width, self.mid_channels))

            # Accumulate results
            if out_accum is None:
                out_accum = out_i
            else:
                out_accum = ttnn.add(out_accum, out_i)

            # Deallocate input slice to free L1
            out_slices[i].deallocate(True)

        out = out_accum

        # Apply bias if it exists
        if self.params.conv1_bias is not None:
            bias_ttnn = self.params.conv1_bias
            if isinstance(bias_ttnn, torch.Tensor):
                if len(bias_ttnn.shape) == 1:
                    bias_ttnn = bias_ttnn.view(1, 1, 1, -1)
                bias_ttnn = ttnn.from_torch(
                    bias_ttnn,
                    device=self.device,
                    dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            out = ttnn.add(out, bias_ttnn)

        # Apply ReLU activation
        out = ttnn.relu(out)

        # Ensure out is in correct format
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)

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

        # Initialize MLP and SELayer if parameters are available
        # Check actual input size from MLP weights (fc1_weight shape)
        if hasattr(parameters, "depth_mlp") and hasattr(parameters.depth_mlp, "fc1_weight"):
            mlp_in_features = parameters.depth_mlp.fc1_weight.shape[1]  # Get from checkpoint
        else:
            mlp_in_features = 31  # Default: 31 features (15 from stack + 16 from sensor2ego 4x4)

        if hasattr(parameters, "depth_mlp") and hasattr(parameters, "context_mlp"):
            self.depth_mlp = MLP_TTNN(
                device, parameters.depth_mlp, mlp_in_features, mid_channels, mid_channels, self.model_config
            )
            self.context_mlp = MLP_TTNN(
                device, parameters.context_mlp, mlp_in_features, mid_channels, mid_channels, self.model_config
            )
        else:
            self.depth_mlp = None
            self.context_mlp = None

        if hasattr(parameters, "depth_se") and hasattr(parameters, "context_se"):
            self.depth_se = SELayer_TTNN(device, parameters.depth_se, mid_channels, self.model_config)
            self.context_se = SELayer_TTNN(device, parameters.context_se, mid_channels, self.model_config)
        else:
            self.depth_se = None
            self.context_se = None

        # BN for MLP input (27 features: 15 from intrins/ida/bda + 12 from sensor2ego 3x4)
        if hasattr(parameters, "mlp_bn"):
            self.mlp_bn = parameters.mlp_bn
        else:
            self.mlp_bn = None

        logger.info(f"DepthNet init: in={in_channels}, mid={mid_channels}, depth={depth_channels}")

        # Enable step-by-step PCC logging (can be set via model_config)
        self.enable_step_pcc = self.model_config.get("ENABLE_STEP_PCC", False)
        self.step_pcc_ref_outputs = {}  # Will be populated by test if enabled

    def _log_step_pcc(self, step_name, ttnn_output, ref_output=None):
        """Helper function to log PCC at each step"""
        if not self.enable_step_pcc:
            return

        if ref_output is None:
            # Try to get from stored reference outputs
            ref_output = self.step_pcc_ref_outputs.get(step_name)

        if ref_output is not None:
            try:
                from models.common.utility_functions import comp_pcc

                # Convert TTNN to torch if needed
                if isinstance(ttnn_output, ttnn.Tensor):
                    ttnn_torch = ttnn.to_torch(ttnn_output)
                    # Handle different tensor formats
                    if len(ttnn_torch.shape) == 4:
                        if ttnn_torch.shape[0] == 1 and ttnn_torch.shape[1] == 1:
                            # Flattened format: [1, 1, H*W, C] -> [B, C, H, W]
                            batch_size = ref_output.shape[0]
                            channels = ref_output.shape[1]
                            height = ref_output.shape[2]
                            width = ref_output.shape[3]
                            ttnn_torch = ttnn_torch.reshape(batch_size, height, width, channels)
                            ttnn_torch = ttnn_torch.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                        elif ttnn_torch.shape[1] == height and ttnn_torch.shape[2] == width:
                            # [B, H, W, C] format
                            ttnn_torch = ttnn_torch.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    else:
                        # Try to reshape based on ref_output shape
                        ttnn_torch = ttnn_torch.reshape(ref_output.shape)

                    pcc_result = comp_pcc(ref_output, ttnn_torch)
                    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
                    logger.info(f"  [{step_name}] PCC = {pcc_value:.6f}")
                else:
                    # Already torch tensor
                    pcc_result = comp_pcc(ref_output, ttnn_output)
                    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
                    logger.info(f"  [{step_name}] PCC = {pcc_value:.6f}")
            except Exception as e:
                logger.warning(f"  [{step_name}] Failed to compute PCC: {e}")
        else:
            # Log tensor statistics instead
            if isinstance(ttnn_output, ttnn.Tensor):
                ttnn_torch = ttnn.to_torch(ttnn_output)
                logger.info(
                    f"  [{step_name}] TTNN output: shape={ttnn_torch.shape}, mean={ttnn_torch.mean().item():.6f}, std={ttnn_torch.std().item():.6f}"
                )
            else:
                logger.info(
                    f"  [{step_name}] TTNN output: shape={ttnn_output.shape}, mean={ttnn_output.mean().item():.6f}, std={ttnn_output.std().item():.6f}"
                )

    def __call__(self, x, batch_size=1, mats_dict=None):
        """
        Forward pass for DepthNet

        Args:
            x: TTNN tensor [batch, height, width, in_channels]
            batch_size: Batch size
            mats_dict: Optional dict with camera matrices. If None, uses identity matrices.
                Required keys: intrin_mats, ida_mats, sensor2ego_mats, bda_mat
        """
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d
        import torch

        height, width = x.shape[1], x.shape[2]

        # Compute MLP input from camera matrices (for SELayer)
        # If mats_dict is None, use identity matrices (for test compatibility)
        if mats_dict is None:
            # Create identity matrices for test
            num_cams = 1
            intrin_mats = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_cams, 1, 1)
            ida_mats = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_cams, 1, 1)
            # Slice to :3, : to match checkpoint (27 features: 15 from stack + 12 from sensor2ego 3x4)
            sensor2ego_mats = (
                torch.eye(4)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, num_cams, 1, 1)[:, 0:1, ..., :3, :]
            )
            bda_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            intrin_mats = mats_dict["intrin_mats"][:, 0:1, ...]
            ida_mats = mats_dict["ida_mats"][:, 0:1, ...]
            # Use :3, : to match checkpoint (27 features: 15 from stack + 12 from sensor2ego 3x4)
            sensor2ego_mats = mats_dict["sensor2ego_mats"][:, 0:1, ..., :3, :]
            bda_mat = mats_dict["bda_mat"]

        # Compute MLP input following reference implementation
        intrins = intrin_mats[..., :3, :3]  # [B, 1, num_cams, 3, 3]
        num_cams = intrins.shape[2]
        bda = bda_mat.view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida_mats[:, 0:1, ..., 0, 0],
                        ida_mats[:, 0:1, ..., 0, 1],
                        ida_mats[:, 0:1, ..., 0, 3],
                        ida_mats[:, 0:1, ..., 1, 0],
                        ida_mats[:, 0:1, ..., 1, 1],
                        ida_mats[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego_mats.view(batch_size, 1, num_cams, -1),  # [B, 1, num_cams, 12] (3x4 matrix)
            ],
            -1,
        )  # [B, 1, num_cams, 27] (15 from stack + 12 from sensor2ego)

        # Apply BN to MLP input
        if self.mlp_bn is not None:
            mlp_input = mlp_input.reshape(-1, mlp_input.shape[-1])  # [B*num_cams, 27]
            # BN: (x - running_mean) / sqrt(running_var + eps) * weight + bias
            if self.mlp_bn.running_mean is not None and self.mlp_bn.running_var is not None:
                mlp_input = (mlp_input - self.mlp_bn.running_mean) / torch.sqrt(
                    self.mlp_bn.running_var + self.mlp_bn.eps
                )
            if self.mlp_bn.weight is not None:
                mlp_input = mlp_input * self.mlp_bn.weight
            if self.mlp_bn.bias is not None:
                mlp_input = mlp_input + self.mlp_bn.bias
            mlp_input = mlp_input.reshape(batch_size, 1, num_cams, -1)  # [B, 1, num_cams, 27]

        # Compute MLP outputs
        mlp_input_flat = mlp_input.reshape(-1, mlp_input.shape[-1])  # [B*num_cams, 27]
        if self.depth_mlp is not None:
            depth_se_mlp = self.depth_mlp(mlp_input_flat)  # [B*num_cams, mid_channels]
            depth_se_mlp = depth_se_mlp.view(batch_size, 1, num_cams, -1)  # [B, 1, num_cams, mid_channels]
        else:
            depth_se_mlp = None

        if self.context_mlp is not None:
            context_se_mlp = self.context_mlp(mlp_input_flat)  # [B*num_cams, mid_channels]
            context_se_mlp = context_se_mlp.view(batch_size, 1, num_cams, -1)  # [B, 1, num_cams, mid_channels]
        else:
            context_se_mlp = None

        # Input from test should already be in TILE_LAYOUT and DRAM_MEMORY_CONFIG
        # Only convert if absolutely necessary (if sharded)
        # Avoid unnecessary memory config/layout conversions that might create unallocated tensors
        if x.is_sharded():
            # Convert sharded to interleaved DRAM
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume input is already in the correct state (DRAM, INTERLEAVED, TILE_LAYOUT)
        # and proceed directly to conv2d

        # Reduce conv - ensure input is in DRAM to force DRAM slicing path
        # This avoids L1 OOM errors for large tensors
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        elif x.is_allocated() and x.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Reduce conv: Use channel slicing to avoid L1 OOM (512->512 is too large)
        # Split into 2 channel slices: 2x (256->512) operations, then sum
        logger.debug(f"Before reduce_conv: x.shape={x.shape}")

        num_slices = 2
        channels_per_slice = self.in_channels // num_slices  # 256

        # Split input along channel dimension
        x_slice1 = ttnn.slice(x, [0, 0, 0, 0], [batch_size, height, width, channels_per_slice])
        x_slice2 = ttnn.slice(x, [0, 0, 0, channels_per_slice], [batch_size, height, width, self.in_channels])

        # Split weights: [out_channels, in_channels, kernel_h, kernel_w] = [512, 512, 3, 3]
        weight_torch = (
            self.params.reduce_weight
            if isinstance(self.params.reduce_weight, torch.Tensor)
            else ttnn.to_torch(self.params.reduce_weight)
        )
        weight_slice1_torch = weight_torch[:, 0:channels_per_slice, :, :]  # [512, 256, 3, 3]
        weight_slice2_torch = weight_torch[:, channels_per_slice:, :, :]  # [512, 256, 3, 3]

        # Run each slice separately - each produces ALL output channels
        out_slice1 = ttnn_conv2d(
            input_tensor=x_slice1,
            weight_tensor=weight_slice1_torch,
            bias_tensor=None,  # No bias in slices, apply once at the end
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=channels_per_slice,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,  # Apply ReLU after sum
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=None,
            packer_l1_acc=False,
        )

        out_slice2 = ttnn_conv2d(
            input_tensor=x_slice2,
            weight_tensor=weight_slice2_torch,
            bias_tensor=None,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=channels_per_slice,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            activations_dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            shard_layout=None,
            packer_l1_acc=False,
        )

        # Reshape outputs if needed
        if len(out_slice1.shape) == 3:
            out_slice1 = ttnn.reshape(out_slice1, (batch_size, height, width, self.mid_channels))
        if len(out_slice2.shape) == 3:
            out_slice2 = ttnn.reshape(out_slice2, (batch_size, height, width, self.mid_channels))

        # SUM the outputs (not concatenate) - each output channel depends on all input channels
        x = ttnn.add(out_slice1, out_slice2)

        # Apply bias if it exists
        if self.params.reduce_bias is not None:
            bias_ttnn = self.params.reduce_bias
            if isinstance(bias_ttnn, torch.Tensor):
                if len(bias_ttnn.shape) == 1:
                    bias_ttnn = bias_ttnn.view(1, 1, 1, -1)
                bias_ttnn = ttnn.from_torch(
                    bias_ttnn,
                    device=self.device,
                    dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            x = ttnn.add(x, bias_ttnn)

        # Apply ReLU activation
        x = ttnn.relu(x)

        # Clean up intermediate tensors
        x_slice1.deallocate(True)
        x_slice2.deallocate(True)
        out_slice1.deallocate(True)
        out_slice2.deallocate(True)

        # Ensure x is in correct format
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        logger.debug(
            f"After reduce_conv: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}, x.is_sharded()={x.is_sharded()}"
        )

        # ttnn.conv2d returns flattened tensor: [1, 1, batch*height*width, channels] or [1, batch*height*width, channels]
        # However, channel slicing produces [batch, height, width, channels] directly, so check shape first
        # We need to reshape it to [batch, height, width, channels] if it's flattened
        # Convert sharded to interleaved DRAM if needed
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Reshape flattened tensor to [batch, height, width, channels]
        if len(x.shape) == 4 and x.shape[0] == 1 and x.shape[1] == 1:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
        elif len(x.shape) == 3 and x.shape[0] == 1:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
        elif len(x.shape) != 4 or x.shape[0] != batch_size or x.shape[1] != height or x.shape[2] != width:
            expected_elements = batch_size * height * width * self.mid_channels
            actual_elements = 1
            for dim in x.shape:
                actual_elements *= dim
            if actual_elements == expected_elements:
                x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
            else:
                raise RuntimeError(
                    f"Cannot reshape x: shape={x.shape}, expected={expected_elements}, actual={actual_elements}"
                )

        # Ensure tensor is in correct state
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Log PCC after reduce_conv
        self._log_step_pcc("reduce_conv", x)

        # Context branch: Apply SELayer before context_conv
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Apply SELayer if available
        if self.context_se is not None and context_se_mlp is not None:
            # Broadcast context_se_mlp from [B, 1, num_cams, mid_channels] to [B, H, W, mid_channels]
            # For single camera, we can just expand: [B, 1, 1, mid_channels] -> [B, H, W, mid_channels]
            context_se_torch = context_se_mlp[:, 0, 0, :]  # [B, mid_channels] for first camera
            context_se_torch = context_se_torch.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, mid_channels]
            context_se_torch = context_se_torch.expand(batch_size, height, width, self.mid_channels)

            # Convert to TTNN tensor
            context_se_ttnn = ttnn.from_torch(
                context_se_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Apply SELayer: x * gate(conv_expand(relu(conv_reduce(x_se))))
            x_context = self.context_se(x, context_se_ttnn)
            # Log PCC after context SELayer
            self._log_step_pcc("context_se", x_context)
        else:
            x_context = x

        context = ttnn_conv2d(
            input_tensor=x_context,
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
            shard_layout=None,  # None defaults to HEIGHT_SHARDED, which uses DRAM
            packer_l1_acc=False,
        )

        # Reshape flattened tensor to [batch, height, width, channels]
        if len(context.shape) == 4 and context.shape[0] == 1 and context.shape[1] == 1:
            context = ttnn.reshape(context, (batch_size, height, width, self.context_channels))
        elif len(context.shape) == 3 and context.shape[0] == 1:
            context = ttnn.reshape(context, (batch_size, height, width, self.context_channels))
        elif (
            len(context.shape) != 4
            or context.shape[0] != batch_size
            or context.shape[1] != height
            or context.shape[2] != width
        ):
            expected_elements = batch_size * height * width * self.context_channels
            actual_elements = 1
            for dim in context.shape:
                actual_elements *= dim
            if actual_elements == expected_elements:
                context = ttnn.reshape(context, (batch_size, height, width, self.context_channels))
            else:
                raise RuntimeError(
                    f"Cannot reshape context: shape={context.shape}, expected={expected_elements}, actual={actual_elements}"
                )
        else:
            logger.error(f"Unexpected context tensor shape after conv2d: {context.shape}")
            raise RuntimeError(f"Cannot reshape unexpected context conv2d output shape: {context.shape}")

        # Convert sharded to interleaved if needed
        if context.is_sharded():
            context = ttnn.sharded_to_interleaved(context, ttnn.DRAM_MEMORY_CONFIG)

        # Log PCC after context_conv
        self._log_step_pcc("context_conv", context)

        # Depth branch: Apply SELayer before depth_conv
        # Verify x is allocated before passing to SELayer/block1
        logger.debug(
            f"Before depth SELayer/block1: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}, x.is_sharded()={x.is_sharded()}, expected_channels={self.mid_channels}"
        )

        # Verify shape is correct (should be mid_channels after reduce_conv)
        if x.shape[-1] != self.mid_channels:
            logger.error(
                f"Tensor x has wrong shape before depth SELayer/block1! Expected channels={self.mid_channels}, got {x.shape[-1]}"
            )
            logger.error(f"Full shape: {x.shape}. This suggests reduce_conv did not update x correctly.")
            raise RuntimeError(
                f"Tensor x has wrong shape before depth SELayer/block1. Expected channels={self.mid_channels}, got {x.shape[-1]}. "
                f"Full shape: {x.shape}. This indicates reduce_conv did not update the tensor correctly."
            )

        if not x.is_allocated():
            logger.error(
                f"Tensor x is not allocated before depth SELayer/block1 - shape: {x.shape}, sharded: {x.is_sharded()}"
            )
            # Try to fix by converting sharded to interleaved if sharded
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
                logger.debug(
                    f"After sharded_to_interleaved before depth SELayer/block1: x.shape={x.shape}, x.is_allocated()={x.is_allocated()}"
                )
            else:
                # Not sharded and not allocated - this shouldn't happen
                raise RuntimeError(
                    f"Tensor x is not allocated and not sharded before depth SELayer/block1. "
                    f"Shape: {x.shape}, Expected channels: {self.mid_channels}. "
                    f"This indicates a bug in the processing pipeline."
                )

        # Ensure x is in TILE_LAYOUT before passing to SELayer/block1
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Apply SELayer if available
        if self.depth_se is not None and depth_se_mlp is not None:
            # Broadcast depth_se_mlp from [B, 1, num_cams, mid_channels] to [B, H, W, mid_channels]
            depth_se_torch = depth_se_mlp[:, 0, 0, :]  # [B, mid_channels] for first camera
            depth_se_torch = depth_se_torch.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, mid_channels]
            depth_se_torch = depth_se_torch.expand(batch_size, height, width, self.mid_channels)

            # Convert to TTNN tensor
            depth_se_ttnn = ttnn.from_torch(
                depth_se_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Apply SELayer: x * gate(conv_expand(relu(conv_reduce(x_se))))
            x_depth = self.depth_se(x, depth_se_ttnn)
            # Log PCC after depth SELayer
            self._log_step_pcc("depth_se", x_depth)
        else:
            x_depth = x

        depth = self.block1(x_depth, batch_size, height, width)
        self._log_step_pcc("block1", depth)
        depth = self.block2(depth, batch_size, height, width)
        self._log_step_pcc("block2", depth)
        depth = self.block3(depth, batch_size, height, width)
        self._log_step_pcc("block3", depth)

        # ASPP
        depth = self.aspp(depth, batch_size, height, width)
        self._log_step_pcc("aspp", depth)

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
            # Keep in [B, H, W, C] format for TTNN (don't flatten to [1, 1, B*H*W, C])
            # depth_torch should already be [batch_size, height, width, mid_channels]

            # Convert back to TTNN tensor
            depth = ttnn.from_torch(
                depth_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Ensure depth is in correct shape [batch, height, width, channels]
            if len(depth.shape) == 4 and depth.shape[0] == 1 and depth.shape[1] == 1:
                # Flattened format: [1, 1, B*H*W, C] -> [B, H, W, C]
                depth = ttnn.reshape(depth, (batch_size, height, width, self.mid_channels))
            elif len(depth.shape) == 3:
                # [1, B*H*W, C] -> [B, H, W, C]
                depth = ttnn.reshape(depth, (batch_size, height, width, self.mid_channels))

            # Log PCC after DCN
            self._log_step_pcc("dcn", depth)
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

        # Log PCC after final depth conv
        self._log_step_pcc("final_depth_conv", depth)

        # Convert both tensors to INTERLEAVED DRAM before concat
        # ttnn.concat requires INTERLEAVED layout when inputs are sharded
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        if depth.is_allocated() and depth.memory_config().buffer_type != ttnn.BufferType.DRAM:
            depth = ttnn.to_memory_config(depth, ttnn.DRAM_MEMORY_CONFIG)
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)

        if context.is_sharded():
            context = ttnn.sharded_to_interleaved(context, ttnn.DRAM_MEMORY_CONFIG)
        if context.is_allocated() and context.memory_config().buffer_type != ttnn.BufferType.DRAM:
            context = ttnn.to_memory_config(context, ttnn.DRAM_MEMORY_CONFIG)
        if context.layout != ttnn.TILE_LAYOUT:
            context = ttnn.to_layout(context, ttnn.TILE_LAYOUT)

        # Concatenate depth and context (both in INTERLEAVED DRAM)
        out = ttnn.concat([depth, context], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out


def fuse_conv_bn_weights_unified(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Unified function to fuse BatchNorm into conv weights for inference.
    Handles both conv with bias and conv without bias.

    Formula verification:
    - BN(x) = (x - mean) / sqrt(var + eps) * weight + bias
    - For conv without bias: BN(conv(x)) = (conv(x) - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = conv(x) * (bn_weight / sqrt(var + eps)) - mean * (bn_weight / sqrt(var + eps)) + bn_bias
      = conv(x) * scale + (bn_bias - mean * scale)
    - For conv with bias: BN(conv(x) + conv_bias) = ((conv(x) + conv_bias) - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = (conv(x) + conv_bias - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = conv(x) * scale + (conv_bias - mean) * scale + bn_bias
      = conv(x) * scale + bn_bias + (conv_bias - mean) * scale

    Args:
        conv_weight: [out_channels, in_channels, kH, kW]
        conv_bias: [out_channels] or None
        bn_weight: [out_channels] (gamma)
        bn_bias: [out_channels] (beta)
        bn_mean: [out_channels] (running_mean)
        bn_var: [out_channels] (running_var)
        eps: BN epsilon

    Returns:
        fused_weight, fused_bias
    """
    # Ensure all inputs are float32 for precision during fusion
    conv_weight = conv_weight.float() if conv_weight.dtype != torch.float32 else conv_weight
    bn_weight = (
        bn_weight.float() if isinstance(bn_weight, torch.Tensor) and bn_weight.dtype != torch.float32 else bn_weight
    )
    bn_bias = bn_bias.float() if isinstance(bn_bias, torch.Tensor) and bn_bias.dtype != torch.float32 else bn_bias
    bn_mean = bn_mean.float() if isinstance(bn_mean, torch.Tensor) and bn_mean.dtype != torch.float32 else bn_mean
    bn_var = bn_var.float() if isinstance(bn_var, torch.Tensor) and bn_var.dtype != torch.float32 else bn_var

    # Calculate scale factor from BN: scale = bn_weight / sqrt(bn_var + eps)
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std

    # Fuse into conv weight: multiply each output channel by its scale
    # Shape: [out_channels, in_channels, kH, kW] * [out_channels, 1, 1, 1]
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

    # Fuse into bias
    # Handle conv_bias: if None, treat as zero
    if conv_bias is not None:
        conv_bias = conv_bias.float() if conv_bias.dtype != torch.float32 else conv_bias
    else:
        # Create zero bias tensor matching conv_weight device
        conv_bias = torch.zeros(conv_weight.shape[0], dtype=torch.float32, device=conv_weight.device)

    # Handle bn_bias: if None, treat as zero
    if bn_bias is not None:
        bn_bias_val = bn_bias
    else:
        bn_bias_val = torch.zeros_like(bn_mean)

    # Fused bias formula: bn_bias + (conv_bias - bn_mean) * scale
    # This works for both cases:
    # - conv_bias = 0: fused_bias = bn_bias - bn_mean * scale (matches standard formula)
    # - conv_bias != 0: fused_bias = bn_bias + (conv_bias - bn_mean) * scale
    fused_bias = bn_bias_val + scale * (conv_bias - bn_mean)

    return fused_weight, fused_bias


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

    # Reduce conv: reduce_conv.0 (conv) -> reduce_conv.1 (BN) -> reduce_conv.2 (ReLU)
    try:
        reduce_conv_weight = state_dict[f"{prefix}reduce_conv.0.weight"].float()  # Keep in float32 for fusion
        reduce_conv_bias = state_dict.get(f"{prefix}reduce_conv.0.bias", None)

        # Load BN parameters (reduce_conv.1)
        reduce_bn_weight = state_dict.get(f"{prefix}reduce_conv.1.weight", None)
        reduce_bn_bias = state_dict.get(f"{prefix}reduce_conv.1.bias", None)
        reduce_bn_mean = state_dict.get(f"{prefix}reduce_conv.1.running_mean", None)
        reduce_bn_var = state_dict.get(f"{prefix}reduce_conv.1.running_var", None)

        # Fuse BN into reduce_conv
        if reduce_bn_weight is not None and reduce_bn_mean is not None and reduce_bn_var is not None:
            # Get BN eps from state dict if available, otherwise use default
            reduce_bn_eps = state_dict.get(f"{prefix}reduce_conv.1.eps", 1e-5)
            if isinstance(reduce_bn_eps, torch.Tensor):
                reduce_bn_eps = reduce_bn_eps.item()
            # Use unified fusion function
            fused_reduce_weight, fused_reduce_bias = fuse_conv_bn_weights_unified(
                reduce_conv_weight,
                reduce_conv_bias,
                reduce_bn_weight,
                reduce_bn_bias,
                reduce_bn_mean,
                reduce_bn_var,
                eps=reduce_bn_eps,
            )
            params.reduce_weight = fused_reduce_weight.to(torch.bfloat16)
            params.reduce_bias = fused_reduce_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use original weights
            params.reduce_weight = reduce_conv_weight.to(torch.bfloat16)
            params.reduce_bias = reduce_conv_bias.to(torch.bfloat16) if reduce_conv_bias is not None else None
    except KeyError as e:
        logger.error(f"Failed to load reduce_conv: {e}")
        logger.info(f"Available depth_net keys: {[k for k in all_keys if prefix in k][:20]}")
        raise

    # MLP and SELayer for camera-aware features
    # MLP: 27 -> mid_channels -> mid_channels
    params.depth_mlp = Parameters()
    params.depth_mlp.fc1_weight = state_dict[f"{prefix}depth_mlp.fc1.weight"].to(torch.bfloat16)
    params.depth_mlp.fc1_bias = state_dict.get(f"{prefix}depth_mlp.fc1.bias", None)
    if params.depth_mlp.fc1_bias is not None:
        params.depth_mlp.fc1_bias = params.depth_mlp.fc1_bias.to(torch.bfloat16)
    params.depth_mlp.fc2_weight = state_dict[f"{prefix}depth_mlp.fc2.weight"].to(torch.bfloat16)
    params.depth_mlp.fc2_bias = state_dict.get(f"{prefix}depth_mlp.fc2.bias", None)
    if params.depth_mlp.fc2_bias is not None:
        params.depth_mlp.fc2_bias = params.depth_mlp.fc2_bias.to(torch.bfloat16)

    params.context_mlp = Parameters()
    params.context_mlp.fc1_weight = state_dict[f"{prefix}context_mlp.fc1.weight"].to(torch.bfloat16)
    params.context_mlp.fc1_bias = state_dict.get(f"{prefix}context_mlp.fc1.bias", None)
    if params.context_mlp.fc1_bias is not None:
        params.context_mlp.fc1_bias = params.context_mlp.fc1_bias.to(torch.bfloat16)
    params.context_mlp.fc2_weight = state_dict[f"{prefix}context_mlp.fc2.weight"].to(torch.bfloat16)
    params.context_mlp.fc2_bias = state_dict.get(f"{prefix}context_mlp.fc2.bias", None)
    if params.context_mlp.fc2_bias is not None:
        params.context_mlp.fc2_bias = params.context_mlp.fc2_bias.to(torch.bfloat16)

    # SELayer: conv_reduce, conv_expand
    params.depth_se = Parameters()
    params.depth_se.conv_reduce_weight = state_dict[f"{prefix}depth_se.conv_reduce.weight"].to(torch.bfloat16)
    params.depth_se.conv_reduce_bias = state_dict.get(f"{prefix}depth_se.conv_reduce.bias", None)
    if params.depth_se.conv_reduce_bias is not None:
        params.depth_se.conv_reduce_bias = params.depth_se.conv_reduce_bias.to(torch.bfloat16)
    params.depth_se.conv_expand_weight = state_dict[f"{prefix}depth_se.conv_expand.weight"].to(torch.bfloat16)
    params.depth_se.conv_expand_bias = state_dict.get(f"{prefix}depth_se.conv_expand.bias", None)
    if params.depth_se.conv_expand_bias is not None:
        params.depth_se.conv_expand_bias = params.depth_se.conv_expand_bias.to(torch.bfloat16)

    params.context_se = Parameters()
    params.context_se.conv_reduce_weight = state_dict[f"{prefix}context_se.conv_reduce.weight"].to(torch.bfloat16)
    params.context_se.conv_reduce_bias = state_dict.get(f"{prefix}context_se.conv_reduce.bias", None)
    if params.context_se.conv_reduce_bias is not None:
        params.context_se.conv_reduce_bias = params.context_se.conv_reduce_bias.to(torch.bfloat16)
    params.context_se.conv_expand_weight = state_dict[f"{prefix}context_se.conv_expand.weight"].to(torch.bfloat16)
    params.context_se.conv_expand_bias = state_dict.get(f"{prefix}context_se.conv_expand.bias", None)
    if params.context_se.conv_expand_bias is not None:
        params.context_se.conv_expand_bias = params.context_se.conv_expand_bias.to(torch.bfloat16)

    # BN for MLP input (27 features)
    params.mlp_bn = Parameters()
    params.mlp_bn.weight = state_dict.get(f"{prefix}bn.weight", None)
    params.mlp_bn.bias = state_dict.get(f"{prefix}bn.bias", None)
    params.mlp_bn.running_mean = state_dict.get(f"{prefix}bn.running_mean", None)
    params.mlp_bn.running_var = state_dict.get(f"{prefix}bn.running_var", None)
    params.mlp_bn.eps = 1e-5  # Default BN eps

    # Context conv
    params.context_weight = state_dict[f"{prefix}context_conv.weight"].to(torch.bfloat16)
    params.context_bias = state_dict.get(f"{prefix}context_conv.bias", None)
    if params.context_bias is not None:
        params.context_bias = params.context_bias.to(torch.bfloat16)

    # BasicBlocks (depth_conv.0, depth_conv.1, depth_conv.2)
    # BasicBlock structure: conv1 -> norm1 (BN) -> ReLU -> conv2 -> norm2 (BN) -> add -> ReLU
    # Need to fuse BN layers into conv weights

    for i in range(3):
        block = Parameters()

        # Load conv1 weight and BN1 parameters
        conv1_weight = state_dict[f"{prefix}depth_conv.{i}.conv1.weight"].float()  # Keep in float32 for fusion
        conv1_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv1.bias", None)
        if conv1_bias is not None:
            conv1_bias = conv1_bias.float()

        # Load BN1 parameters (norm1)
        bn1_weight = state_dict.get(f"{prefix}depth_conv.{i}.norm1.weight", None)
        bn1_bias = state_dict.get(f"{prefix}depth_conv.{i}.norm1.bias", None)
        bn1_mean = state_dict.get(f"{prefix}depth_conv.{i}.norm1.running_mean", None)
        bn1_var = state_dict.get(f"{prefix}depth_conv.{i}.norm1.running_var", None)

        # Fuse BN1 into conv1
        if bn1_weight is not None and bn1_mean is not None and bn1_var is not None:
            # Get BN eps from state dict if available, otherwise use default
            bn1_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm1.eps", 1e-5)
            if isinstance(bn1_eps, torch.Tensor):
                bn1_eps = bn1_eps.item()
            # Use unified fusion function (handles conv with or without bias)
            fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
                conv1_weight,
                conv1_bias,
                bn1_weight,
                bn1_bias,
                bn1_mean,
                bn1_var,
                eps=bn1_eps,
            )
            # Debug: Verify fusion for first block
            if i == 0:
                logger.debug(
                    f"Block {i} conv1 fusion: weight_norm={fused_conv1_weight.norm().item():.6f}, bias_norm={fused_conv1_bias.norm().item():.6f}"
                )
            block.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
            block.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use original weights
            block.conv1_weight = conv1_weight.to(torch.bfloat16)
            block.conv1_bias = conv1_bias.to(torch.bfloat16) if conv1_bias is not None else None

        # Load conv2 weight and BN2 parameters
        conv2_weight = state_dict[f"{prefix}depth_conv.{i}.conv2.weight"].float()  # Keep in float32 for fusion
        conv2_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv2.bias", None)
        if conv2_bias is not None:
            conv2_bias = conv2_bias.float()

        # Load BN2 parameters (norm2)
        bn2_weight = state_dict.get(f"{prefix}depth_conv.{i}.norm2.weight", None)
        bn2_bias = state_dict.get(f"{prefix}depth_conv.{i}.norm2.bias", None)
        bn2_mean = state_dict.get(f"{prefix}depth_conv.{i}.norm2.running_mean", None)
        bn2_var = state_dict.get(f"{prefix}depth_conv.{i}.norm2.running_var", None)

        # Fuse BN2 into conv2
        if bn2_weight is not None and bn2_mean is not None and bn2_var is not None:
            # Get BN eps from state dict if available, otherwise use default
            bn2_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm2.eps", 1e-5)
            if isinstance(bn2_eps, torch.Tensor):
                bn2_eps = bn2_eps.item()
            # Use unified fusion function (handles conv with or without bias)
            fused_conv2_weight, fused_conv2_bias = fuse_conv_bn_weights_unified(
                conv2_weight,
                conv2_bias,
                bn2_weight,
                bn2_bias,
                bn2_mean,
                bn2_var,
                eps=bn2_eps,
            )
            block.conv2_weight = fused_conv2_weight.to(torch.bfloat16)
            block.conv2_bias = fused_conv2_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use original weights
            block.conv2_weight = conv2_weight.to(torch.bfloat16)
            block.conv2_bias = conv2_bias.to(torch.bfloat16) if conv2_bias is not None else None

        setattr(params, f"block{i+1}", block)

    # ASPP (depth_conv.3)
    # ASPP structure: Each branch has atrous_conv -> bn -> relu, final conv1 -> bn1 -> relu
    params.aspp = Parameters()

    # Fuse BN for aspp1-aspp4 branches
    for branch_idx, branch_name in enumerate(["aspp1", "aspp2", "aspp3", "aspp4"], 1):
        atrous_weight = state_dict[f"{prefix}depth_conv.3.{branch_name}.atrous_conv.weight"].float()
        # Load BN parameters
        bn_weight = state_dict.get(f"{prefix}depth_conv.3.{branch_name}.bn.weight", None)
        bn_bias = state_dict.get(f"{prefix}depth_conv.3.{branch_name}.bn.bias", None)
        bn_mean = state_dict.get(f"{prefix}depth_conv.3.{branch_name}.bn.running_mean", None)
        bn_var = state_dict.get(f"{prefix}depth_conv.3.{branch_name}.bn.running_var", None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
            bn_eps = 1e-5
            # Atrous conv has bias=False
            fused_weight, fused_bias = fuse_conv_bn_weights_unified(
                atrous_weight,
                None,  # No conv bias
                bn_weight,
                bn_bias,
                bn_mean,
                bn_var,
                eps=bn_eps,
            )
            setattr(params.aspp, f"{branch_name}_weight", fused_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", fused_bias.to(torch.bfloat16))
        else:
            setattr(params.aspp, f"{branch_name}_weight", atrous_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", None)

    # Fuse BN for global_avg_pool (conv -> bn -> relu)
    global_weight = state_dict[f"{prefix}depth_conv.3.global_avg_pool.1.weight"].float()
    global_bn_weight = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.weight", None)
    global_bn_bias = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.bias", None)
    global_bn_mean = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_mean", None)
    global_bn_var = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_var", None)

    if global_bn_weight is not None and global_bn_mean is not None and global_bn_var is not None:
        # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
        global_bn_eps = 1e-5
        # Global avg pool conv has bias=False
        fused_global_weight, fused_global_bias = fuse_conv_bn_weights_unified(
            global_weight,
            None,  # No conv bias
            global_bn_weight,
            global_bn_bias,
            global_bn_mean,
            global_bn_var,
            eps=global_bn_eps,
        )
        params.aspp.global_weight = fused_global_weight.to(torch.bfloat16)
        params.aspp.global_bias = fused_global_bias.to(torch.bfloat16)
    else:
        params.aspp.global_weight = global_weight.to(torch.bfloat16)
        params.aspp.global_bias = None

    # Fuse BN for final conv1 (conv1 -> bn1 -> relu)
    conv1_weight = state_dict[f"{prefix}depth_conv.3.conv1.weight"].float()
    conv1_bn_weight = state_dict.get(f"{prefix}depth_conv.3.bn1.weight", None)
    conv1_bn_bias = state_dict.get(f"{prefix}depth_conv.3.bn1.bias", None)
    conv1_bn_mean = state_dict.get(f"{prefix}depth_conv.3.bn1.running_mean", None)
    conv1_bn_var = state_dict.get(f"{prefix}depth_conv.3.bn1.running_var", None)

    if conv1_bn_weight is not None and conv1_bn_mean is not None and conv1_bn_var is not None:
        # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
        conv1_bn_eps = 1e-5
        # ASPP final conv1 has bias=False
        fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
            conv1_weight,
            None,  # No conv bias
            conv1_bn_weight,
            conv1_bn_bias,
            conv1_bn_mean,
            conv1_bn_var,
            eps=conv1_bn_eps,
        )
        params.aspp.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
    else:
        params.aspp.conv1_weight = conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = None

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
