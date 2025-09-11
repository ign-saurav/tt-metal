# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.common.lightweightmodule import LightweightModule
import torch


class TTUpsample(LightweightModule):
    def __init__(self, scale, num_feat, device):
        self.scale = scale
        self.num_feat = num_feat
        self.device = device
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Pre-calculate operation parameters
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            self.num_ops = int(math.log(scale, 2))
            self.out_channels = 4 * num_feat
            self.scale_factor = 2
        elif scale == 3:
            self.num_ops = 1
            self.out_channels = 9 * num_feat
            self.scale_factor = 3
        else:
            raise ValueError(f"Unsupported scale: {scale}")
            # Initialize compute config for the device
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Initialize conv config with no activation and default output layout
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,  # Free input memory after use
            reallocate_halo_output=False,  # Reduce memory fragmentation
            act_block_h_override=32,  # Use smaller activation blocks
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # Use height sharding
        )

    def pixel_shuffle(self, x, upscale_factor):
        """Implement PixelShuffle operation in TTNN"""
        # Convert to interleaved layout before reshape to avoid sharding issues
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Ensure we're in ROW_MAJOR layout for reshape operations
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # x shape: [B, H, W, C] where C = input_channels * upscale_factor^2
        batch_size, height, width, channels = x.shape
        out_channels = channels // (upscale_factor * upscale_factor)

        # Reshape to separate the upscale dimensions - CORRECT NHWC ORDER
        # [B, H, W, out_channels, upscale_factor, upscale_factor]
        x = ttnn.reshape(
            x,
            (batch_size, height, width, out_channels, upscale_factor, upscale_factor),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute to rearrange dimensions for upsampling
        # [B, H, upscale_factor, W, upscale_factor, out_channels]
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape to final upsampled size
        # [B, H*upscale_factor, W*upscale_factor, out_channels]
        output_height = height * upscale_factor
        output_width = width * upscale_factor
        x = ttnn.reshape(
            x, (batch_size, output_height, output_width, out_channels), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return x

    def pixel_shuffle_torch(self, x, upscale_factor):
        """Implement PixelShuffle operation using PyTorch for better performance"""
        # PyTorch pixel_shuffle expects NCHW format, but our tensor is NHWC
        # Convert from NHWC to NCHW
        torch_tensor = x.permute(0, 3, 1, 2)

        # Apply PyTorch pixel shuffle
        torch_output = torch.nn.functional.pixel_shuffle(torch_tensor, upscale_factor)

        # Convert back from NCHW to NHWC
        torch_output = torch_output.permute(0, 2, 3, 1)

        # Convert back to TTNN tensor
        ttnn_output = ttnn.from_torch(
            torch_output,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.memory_config,
        )

        return ttnn_output

    def forward(self, x, parameters):
        current = x
        current_channels = self.num_feat
        slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4)
        for i in range(self.num_ops):
            # Calculate output channels for this specific convolution
            out_channels = current_channels * (self.scale_factor * self.scale_factor)
            batch_size = current.shape[0]
            height = current.shape[1]
            width = current.shape[2]
            current = ttnn.conv2d(
                input_tensor=current,
                weight_tensor=parameters[f"conv_{i}"]["weight"],
                bias_tensor=parameters[f"conv_{i}"]["bias"] if parameters[f"conv_{i}"]["bias"] else None,
                in_channels=current_channels,
                out_channels=out_channels,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                dtype=ttnn.bfloat16,
                return_output_dim=False,
                return_weights_and_bias=False,
                slice_config=slice_config,
            )

            current = ttnn.to_torch(current)
            # reshape B,1,H*W, C to B, H, W, C
            current = current.reshape(
                batch_size,
                current.shape[2] // (height * batch_size),
                current.shape[2] // (height * batch_size),
                out_channels,
            )
            current = self.pixel_shuffle_torch(current, self.scale_factor)
            # After pixel shuffle, channels return to original count
            current_channels = self.num_feat

        return current
