import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.SSR.tt.tile_refinement import TTTileRefinement
from models.experimental.SSR.tt.tile_selection import TTTileSelection
from models.experimental.SSR.tt.upsample import TTUpsample


def ttnn_window_reverse(windows, window_size, H, W, device, memory_config=None):
    """
    TTNN equivalent of window_reverse

    Args:
        windows: TTNN tensor with shape (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        device: TTNN device
        memory_config: memory configuration for operations

    Returns:
        x: TTNN tensor with shape (B, H, W, C)
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Calculate batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]

    # Reshape to separate batch and window dimensions: (B, H//window_size, W//window_size, window_size, window_size, C)
    intermediate_shape = (B, H // window_size, W // window_size, window_size, window_size, C)
    x_reshaped = ttnn.reshape(windows, intermediate_shape)

    # Permute to reconstruct spatial dimensions: (B, H//window_size, window_size, W//window_size, window_size, C)
    # This corresponds to permute(0, 1, 3, 2, 4, 5) from the original
    x_permuted = ttnn.permute(x_reshaped, (0, 1, 3, 2, 4, 5))

    # Flatten to final shape: (B, H, W, C)
    final_shape = (B, H, W, C)
    x = ttnn.reshape(x_permuted, final_shape)

    return x


def ttnn_window_partition(x, window_size, device, memory_config=None):
    """
    TTNN equivalent of window_partition

    Args:
        x: TTNN tensor with shape (B, H, W, C)
        window_size (int): window size
        device: TTNN device
        memory_config: memory configuration for operations

    Returns:
        windows: TTNN tensor with shape (num_windows*B, window_size, window_size, C)
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Get tensor dimensions
    B, H, W, C = x.shape

    # Reshape to separate window dimensions: (B, H//window_size, window_size, W//window_size, window_size, C)
    intermediate_shape = (B, H // window_size, window_size, W // window_size, window_size, C)
    x_reshaped = ttnn.reshape(x, intermediate_shape)

    # Permute to group windows together: (B, H//window_size, W//window_size, window_size, window_size, C)
    # This corresponds to permute(0, 1, 3, 2, 4, 5) from the original
    x_permuted = ttnn.permute(x_reshaped, (0, 1, 3, 2, 4, 5))

    # Flatten to final shape: (num_windows*B, window_size, window_size, C)
    num_windows = (H // window_size) * (W // window_size)
    final_shape = (num_windows * B, window_size, window_size, C)
    windows = ttnn.reshape(x_permuted, final_shape)

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
            # act_block_h_override=32,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.conv_before_upsample_conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            reallocate_halo_output=True,
            # act_block_h_override=32,
            # act_block_h_override=16,
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
        patch_fea3, patch_fea2, patch_fea1 = self.select_model(x)

        # Calculate selection threshold (top 25%)
        patch_fea3_flat = ttnn.reshape(patch_fea3, (-1,))
        # Convert to torch for quantile calculation (TTNN doesn't have quantile yet)
        patch_fea3_torch = ttnn.to_torch(patch_fea3_flat)
        # import pdb; pdb.set_trace()
        threshold = torch.quantile(patch_fea3_torch.to(torch.float32), 0.75)

        # Create selection mask
        pi_prime = patch_fea3_torch > threshold
        pi_prime = pi_prime.view(-1)

        # Window partition the input image
        # Convert to torch for window partitioning (using existing utility)
        # x_torch = ttnn.to_torch(x)
        x_torch = x
        x_torch = ttnn.permute(x, (0, 2, 3, 1))
        patch_x_torch = ttnn_window_partition(
            x_torch,
            window_size=H // 4,
            device=self.device,
        )  # B*4*4, H/4, W/4, 3
        # import pdb; pdb.set_trace()
        # patch_x_torch = patch_x_torch.permute(0, 3, 1, 2)  # B*4*4, 3, H/4, W/4

        # Feature extraction for each patch
        lr_fea_list = []

        for i in range(B * 16):
            patch_input = ttnn.unsqueeze(patch_x_torch[i], 0)  # 1, 3, H/4, W/4

            # Convert patch to TTNN tensor
            # tt_patch = ttnn.from_torch(
            #     patch_input,
            #     device=self.device,
            #     layout=ttnn.TILE_LAYOUT,
            #     memory_config=self.memory_config
            # )

            if pi_prime[i] == 1:
                # Use SR model for positive tiles
                posX, fea = self.sr_model(ttnn.permute(patch_input, (0, 3, 1, 2)))
                # fea = ttnn.to_dtype(fea, ttnn.bfloat16)
                # lr_fea_list.append(fea)
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
                fea = ttnn.reshape(fea, [1, 64, 64, 180])  # TODO
                # fea = ttnn.to_dtype(fea, ttnn.bfloat16)
                # lr_fea_list.append(fea)
                fea = ttnn.from_device(fea)  # Move to host
                fea = ttnn.to_dtype(fea, ttnn.bfloat16)  # Convert dtype
                fea = ttnn.to_device(fea, device=self.device)  # Move back to device
                lr_fea_list.append(fea)

        # Concatenate features
        lr_fea = ttnn.concat(lr_fea_list, dim=0)

        # Window reverse to reconstruct full feature map
        # Convert to torch for window reverse operation
        # lr_fea_torch = ttnn.to_torch(lr_fea)
        # import pdb; pdb.set_trace()
        lr_fea = ttnn_window_reverse(
            lr_fea,
            window_size=H // 4,
            H=H,
            W=W,
            device=self.device,
        )

        # Convert back to TTNN
        # lr_fea = ttnn.from_torch(
        #     lr_fea_torch,
        #     device=self.device,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=self.memory_config
        # )

        # Image reconstruction
        # Conv before upsample
        # lr_fea = ttnn.to_layout(lr_fea, ttnn.ROW_MAJOR_LAYOUT, memory_config=self.memory_config)
        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dSliceHeight, num_slices=4  # Adjust based on memory constraints
        )
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
            # conv_config=self.conv_before_upsample_conv_config,
            # compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=False,
            return_weights_and_bias=False,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            slice_config=slice_config,
        )
        # import pdb; pdb.set_trace()
        sr_fea = ttnn.reshape(sr_fea, [B, 256, 256, 64])  # TODO

        # LeakyReLU activation
        # import pdb; pdb.set_trace()
        sr_fea = ttnn.leaky_relu(sr_fea, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Upsample
        sr_fea = self.upsample(sr_fea, self.parameters.upsample)
        # return sr_fea, patch_fea3, patch_fea2, patch_fea1

        # Final convolution
        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dSliceHeight, num_slices=4  # Adjust based on memory constraints
        )
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
            # memory_config=self.memory_config,
            # conv_config=self.conv_config,
            slice_config=slice_config,
        )

        sr = ttnn.reshape(sr, [B, 1024, 1024, 3])  # TODO

        return sr, patch_fea3, patch_fea2, patch_fea1
