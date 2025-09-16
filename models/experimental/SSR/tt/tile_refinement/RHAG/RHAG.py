# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from .ATTEN_BLK import TTAttenBlocks

# from models.experimental.SSR.tt.patch_embed import TTPatchEmbed
from .patch_embed_tile_refinement import TTPatchEmbed
from .patch_unembed import TTPatchUnEmbed


class TTRHAG(LightweightModule):
    """TTNN Residual Hybrid Attention Group (RHAG).

    Args:
        device: TTNN device
        parameters: Preprocessed parameters dictionary
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        compress_ratio (int): Compression ratio for CAB
        squeeze_factor (int): Squeeze factor for channel attention
        conv_scale (float): Scale factor for conv branch
        overlap_ratio (float): Overlap ratio for OCAB
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): If True, add a learnable bias to query, key, value
        qk_scale (float | None): Override default qk scale
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
        drop_path (float | tuple[float]): Stochastic depth rate
        downsample: Downsample layer at the end of the layer
        img_size (int): Input image size
        patch_size (int): Patch size
        resi_connection (str): The convolutional block before residual connection
        memory_config: TTNN memory configuration
    """

    def __init__(
        self,
        device,
        parameters,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        compress_ratio,
        squeeze_factor,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        memory_config=None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dim = dim
        self.input_resolution = input_resolution
        self.resi_connection = resi_connection
        self.dtype = dtype

        # Initialize AttenBlocks (residual_group)
        self.residual_group = TTAttenBlocks(
            device=device,
            parameters=parameters["residual_group"],
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            downsample=downsample,
            memory_config=memory_config,
            dtype=dtype,
        )

        # Initialize convolutional layer for residual connection
        if resi_connection == "1conv":
            # Extract conv parameters
            self.conv_weight = parameters["conv"]["weight"]
            self.conv_bias = parameters["conv"]["bias"]

            # Conv2d configuration
            self.conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                activation="",
                output_layout=ttnn.TILE_LAYOUT,
                deallocate_activation=True,
                reallocate_halo_output=True,
            )

            # Compute configuration
            self.compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        elif resi_connection == "identity":
            # Identity connection - no conv layer needed
            self.conv_weight = None
            self.conv_bias = None

        # Initialize PatchEmbed
        self.patch_embed = TTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,  # Set to 0 as in original
            embed_dim=dim,
            norm_layer=None,
            device=device,
            parameters=parameters["patch_embed"],
            memory_config=memory_config,
        )

        # Initialize PatchUnEmbed
        self.patch_unembed = TTPatchUnEmbed(
            mesh_device=device,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,  # Set to 0 as in original
            embed_dim=dim,
        )

    def forward(self, x, x_size, params):
        """
        Forward pass through RHAG

        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            x_size: Tuple of (height, width) for spatial dimensions
            params: Dictionary containing:
                - "rpi_sa": Relative position index for self-attention
                - "attn_mask": Attention mask for shifted windows
                - "rpi_oca": Relative position index for overlapping cross attention

        Returns:
            Output tensor with residual connection
        """
        # Store input for residual connection
        shortcut = x

        # Pass through residual group (AttenBlocks)
        x = self.residual_group(x, x_size, params)

        # Patch unembed: convert from sequence to spatial format
        x = self.patch_unembed(x, x_size)

        # Apply convolutional layer
        if self.resi_connection == "1conv":
            batch_size, embed_dim, height, width = x.shape
            x = ttnn.permute(x, (0, 2, 3, 1))  # (batch_size, embed_dim, num_patches)

            # Apply 3x3 convolution with padding=1
            x, [out_height, out_width] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.conv_weight,
                bias_tensor=self.conv_bias,
                in_channels=self.dim,
                out_channels=self.dim,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                dtype=self.dtype,
                return_output_dim=True,
            )

            x = ttnn.reshape(x, (batch_size, out_height, out_width, self.dim))
        elif self.resi_connection == "identity":
            x = ttnn.permute(x, (0, 2, 3, 1))  # (batch_size, embed_dim, num_patches)

        # Patch embed: convert back to sequence format
        x = self.patch_embed(x)

        x = ttnn.reshape(x, (x.shape[0], self.input_resolution[0] * self.input_resolution[1], self.dim))

        # Add residual connection
        x = ttnn.add(x, shortcut, dtype=self.dtype)

        return x
