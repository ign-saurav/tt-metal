# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.SSR.tt.tile_refinement.tile_refinement import TTTileRefinement
from models.experimental.SSR.tt.tile_selection.tile_selection import TTTileSelection
from models.experimental.SSR.tt.tile_refinement.upsample import TTUpsample


def window_partition_ttnn(x, window_size):
    """TTNN implementation of window partitioning"""
    b, h, w, c = x.shape

    # Reshape: (b, h, w, c) -> (b, h//ws, ws, w//ws, ws, c)
    x = ttnn.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))

    # Permute: (0, 1, 3, 2, 4, 5) -> group windows together
    x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))

    # Final reshape to get windows
    x = ttnn.reshape(x, (-1, window_size, window_size, c))

    return x


def window_reverse_ttnn(windows, window_size, h, w):
    """TTNN implementation of window reverse"""
    b = int(windows.shape[0] / (h * w / window_size / window_size))

    # Reshape windows back to grid
    windows = ttnn.reshape(
        windows,
        (b, h // window_size, w // window_size, window_size, window_size, -1),
    )

    # Permute back to original order
    windows = ttnn.permute(windows, (0, 1, 3, 2, 4, 5))

    # Final reshape to original spatial dimensions
    windows = ttnn.reshape(windows, (b, h, w, -1))

    return windows


class TTSSR(LightweightModule):
    """TTNN Super-Resolution Module

    Feeds positive tiles to TR Module, negative tiles to conv layers,
    then reconstructs them together.
    """

    def __init__(self, device, parameters, args, num_cls, memory_config=None):
        super().__init__()

        self.device = device
        self.parameters = parameters
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Initialize sub-modules using existing TTNN implementations
        self.select_model = TTTileSelection(
            device=device,
            parameters=parameters.select_model,
            args=args,
            num_cls=num_cls,
            memory_config=self.memory_config,
        )

        self.sr_model = TTTileRefinement(
            device=device,
            parameters=parameters.sr_model,
            upscale=4,
            img_size=64,
            window_size=16,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            memory_config=self.memory_config,
        )

        # Initialize upsample module
        self.upsample = TTUpsample(scale=4, num_feat=64, device=device)

        # Store conv configuration for memory optimization
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            reallocate_halo_output=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.conv_before_upsample_conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            reallocate_halo_output=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def forward(self, x):
        """Forward pass through SSR module"""
        B, C, H, W = x.shape

        # Get tile selection features
        patch_fea3 = self.select_model(x)

        # Calculate selection threshold (top 25%)
        patch_fea3_flat = ttnn.reshape(patch_fea3, (-1,))
        # Convert to torch for quantile calculation
        patch_fea3_torch = ttnn.to_torch(patch_fea3_flat)
        threshold = torch.quantile(patch_fea3_torch.to(torch.float32), 0.75)

        # Create selection mask
        pi_prime = patch_fea3_torch > threshold
        pi_prime = pi_prime.view(-1)

        # Window partition the input image
        x_torch = x
        x_torch = ttnn.permute(x, (0, 2, 3, 1))
        patch_x_torch = window_partition_ttnn(
            x_torch,
            window_size=H // 4,
        )

        # Feature extraction for each patch
        lr_fea_list = []

        for i in range(B * 16):
            patch_input = ttnn.unsqueeze(patch_x_torch[i], 0)  # 1, 3, H/4, W/4
            if pi_prime[i] == 1:
                posX, fea = self.sr_model(ttnn.permute(patch_input, (0, 3, 1, 2)))
                fea = ttnn.from_device(fea)  # Move to host
                fea = ttnn.to_dtype(fea, ttnn.bfloat16)  # Convert dtype
                fea = ttnn.to_device(fea, device=self.device)  # Move back to device
                lr_fea_list.append(fea)
            else:
                # Use simple conv for negative tiles
                fea = ttnn.conv2d(
                    input_tensor=patch_input,
                    weight_tensor=self.parameters.conv_first.weight,
                    bias_tensor=self.parameters.conv_first.bias,
                    in_channels=3,
                    out_channels=180,
                    device=self.device,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    batch_size=patch_input.shape[0],
                    input_height=patch_input.shape[1],
                    input_width=patch_input.shape[2],
                    memory_config=self.memory_config,
                    conv_config=self.conv_config,
                )
                fea = ttnn.reshape(fea, [1, 64, 64, 180])
                fea = ttnn.from_device(fea)  # Move to host
                fea = ttnn.to_dtype(fea, ttnn.bfloat16)  # Convert dtype
                fea = ttnn.to_device(fea, device=self.device)  # Move back to device
                lr_fea_list.append(fea)

        # Concatenate features
        lr_fea = ttnn.concat(lr_fea_list, dim=0)

        # Window reverse to reconstruct full feature map
        lr_fea = window_reverse_ttnn(
            lr_fea,
            window_size=H // 4,
            h=H,
            w=W,
        )

        slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4)
        sr_fea = ttnn.conv2d(
            input_tensor=lr_fea,
            weight_tensor=self.parameters.conv_before_upsample.weight,
            bias_tensor=self.parameters.conv_before_upsample.bias,
            in_channels=180,
            out_channels=64,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=lr_fea.shape[0],
            input_height=lr_fea.shape[1],
            input_width=lr_fea.shape[2],
            dtype=ttnn.bfloat16,
            return_output_dim=False,
            return_weights_and_bias=False,
            slice_config=slice_config,
        )
        sr_fea = ttnn.reshape(sr_fea, [B, 256, 256, 64])

        # LeakyReLU activation
        sr_fea = ttnn.leaky_relu(sr_fea, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Upsample
        sr_fea = self.upsample(sr_fea, self.parameters.upsample)

        # Final convolution
        slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4)
        sr = ttnn.conv2d(
            input_tensor=sr_fea,
            weight_tensor=self.parameters.conv_last.weight,
            bias_tensor=self.parameters.conv_last.bias,
            in_channels=64,
            out_channels=3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            device=self.device,
            batch_size=sr_fea.shape[0],
            input_height=sr_fea.shape[1],
            input_width=sr_fea.shape[2],
            dtype=ttnn.bfloat16,
            return_output_dim=False,
            return_weights_and_bias=False,
            slice_config=slice_config,
        )

        sr = ttnn.reshape(sr, [B, 1024, 1024, 3])

        return sr, patch_fea3


class TTSSR_wo_conv(LightweightModule):
    def __init__(self, device, parameters, args, num_cls, memory_config=None, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.dtype = dtype

        # Only need select_model and sr_model - no conv layers
        self.select_model = TTTileSelection(
            device=device,
            parameters=parameters.select_model,
            args=args,
            num_cls=num_cls,
            memory_config=self.memory_config,
            dtype=dtype,
        )

        self.sr_model = TTTileRefinement(
            device=device,
            parameters=parameters.sr_model,
            upscale=4,
            img_size=64,
            window_size=16,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            memory_config=self.memory_config,
            dtype=dtype,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Same tile selection logic
        patch_fea3 = self.select_model(x)

        # Calculate selection threshold (top 25%)
        patch_fea3_flat = ttnn.reshape(patch_fea3, (-1,))
        # Convert to torch for quantile calculation
        patch_fea3_torch = ttnn.to_torch(patch_fea3_flat)
        threshold = torch.quantile(patch_fea3_torch.to(torch.float32), 0.75)
        pi_prime = patch_fea3_torch > threshold

        # Window partition
        x_torch = x
        x_torch = ttnn.permute(x, (0, 2, 3, 1))
        patch_x = window_partition_ttnn(x_torch, window_size=H // 4)

        # Process each patch
        sr_patches = []
        for i in range(B * 16):
            patch_input = ttnn.unsqueeze(patch_x[i], 0)

            if pi_prime[i] == 1:
                # Use SR model for positive tiles
                posX, _ = self.sr_model(ttnn.permute(patch_input, (0, 3, 1, 2)))
                sr_patches.append(posX)
            else:
                # Move tensor to host and convert to torch
                patch_host = ttnn.to_torch(patch_input)  # Shape: (1, H/4, W/4, C)

                # Convert to NCHW format for PyTorch upsample
                patch_nchw = patch_host.permute(0, 3, 1, 2)  # (1, H/4, W/4, C) -> (1, C, H/4, W/4)

                # Use PyTorch's upsample (bicubic like the reference)
                negX_torch = torch.nn.functional.upsample(patch_nchw, scale_factor=4, mode="bicubic")

                # Convert back to NHWC and to TTNN tensor
                negX_torch = negX_torch.permute(0, 2, 3, 1)  # Back to (1, H, W, C)
                negX = ttnn.from_torch(negX_torch, device=self.device, dtype=ttnn.bfloat16)
                if negX.layout == ttnn.ROW_MAJOR_LAYOUT:
                    negX = ttnn.to_layout(negX, ttnn.TILE_LAYOUT, dtype=self.dtype)
                sr_patches.append(negX)

        ttnn.deallocate(patch_x)

        # Concatenate and reconstruct
        sr = ttnn.concat(sr_patches, dim=0)
        return sr, patch_fea3
