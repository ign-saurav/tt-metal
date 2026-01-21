# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.swin2sr.tt.tt_patch_embed import TtSwin2SRPatchEmbed, TtSwin2SRPatchUnEmbed
from models.experimental.swin2sr.tt.tt_rstb import TtRSTB, to_2tuple
from models.experimental.swin2sr.tt.tt_upsample import TtUpsample
from models.experimental.swin2sr.tt.utils import _create_conv_config_from_params, _get_sharding_strategy
from models.tt_cnn.tt.builder import TtConv2d


class TtSwin2SR:
    """Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration.

    TTNN implementation of Swin2SR model for image super-resolution.

    Args:
        device: TTNN device.
        parameters: Dictionary of model parameters.
        img_size (int | tuple[int]): Input image size. Default: 64
        patch_size (int | tuple[int]): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple[int]): Depth of each Swin Transformer layer.
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        ape (bool): If True, add absolute position embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        memory_config: Memory configuration. Default: DRAM_MEMORY_CONFIG.
    """

    def __init__(
        self,
        device,
        parameters,
        img_size: int | tuple[int, int] = 64,
        patch_size: int | tuple[int, int] = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        ape: bool = False,
        patch_norm: bool = True,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        self.memory_config = memory_config

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            mean_values = rgb_mean
            mean_shape = (1, 3, 1, 1)
        else:
            mean_values = (0.0,)
            mean_shape = (1, 1, 1, 1)

        if in_chans == 3:
            mean_list = []
            for val in mean_values:
                mean_list.append(
                    ttnn.full(
                        (1, 1, 1, 1),
                        fill_value=val,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=memory_config,
                    )
                )
            self.mean = ttnn.concat(mean_list, dim=1)
        else:
            self.mean = ttnn.zeros(
                mean_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory_config,
            )

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.conv_first = self._create_conv_layer(
            parameters["conv_first"],
            in_channels=num_in_ch,
            out_channels=embed_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            input_height=img_size[0],
            input_width=img_size[1],
        )

        self.patch_embed = TtSwin2SRPatchEmbed(
            device=device,
            parameters=parameters["patch_embed"],
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=None if not self.patch_norm else "LayerNorm",
            memory_config=memory_config,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = TtSwin2SRPatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            memory_config=memory_config,
        )

        if self.ape:
            self.absolute_pos_embed = parameters.get("absolute_pos_embed", None)

        self.num_layers = len(depths)
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = TtRSTB(
                device=device,
                parameters=parameters["layers"][i_layer],
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                memory_config=memory_config,
            )
            self.layers.append(layer)

        self.norm_weight = parameters.get("norm", {}).get("weight", None)
        self.norm_bias = parameters.get("norm", {}).get("bias", None)

        if resi_connection == "1conv":
            self.conv_after_body = self._create_conv_layer(
                parameters["conv_after_body"],
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                input_height=img_size[0],
                input_width=img_size[1],
            )
        elif resi_connection == "3conv":
            self.conv_after_body = self._create_3conv_layer(
                parameters["conv_after_body"],
                in_channels=embed_dim,
                out_channels=embed_dim,
                input_height=img_size[0],
                input_width=img_size[1],
            )
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        if self.upsampler == "pixelshuffle":
            conv_before_params = parameters["conv_before_upsample"][0]
            sharding_strategy = _get_sharding_strategy(img_size[0], img_size[1], embed_dim, num_feat)
            conv_before_config = _create_conv_config_from_params(
                input_height=img_size[0],
                input_width=img_size[1],
                in_channels=embed_dim,
                out_channels=num_feat,
                batch_size=1,
                parameters=conv_before_params,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation=None,
                deallocate_activation=True,
                sharding_strategy=sharding_strategy,
                config_tensors_in_dram=True,
            )
            self.conv_before_upsample = TtConv2d(conv_before_config, self.device)

            self.upsample = TtUpsample(
                device=device,
                parameters=parameters["upsample"],
                scale=upscale,
                num_feat=num_feat,
                input_height=img_size[0],
                input_width=img_size[1],
                memory_config=memory_config,
            )

            self.conv_last = self._create_conv_layer(
                parameters["conv_last"],
                in_channels=num_feat,
                out_channels=num_out_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                input_height=img_size[0] * upscale,
                input_width=img_size[1] * upscale,
            )
        else:
            raise ValueError(f"Unsupported upsampler: {upsampler}")

    def _create_conv_layer(
        self,
        parameters,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        input_height,
        input_width,
        activation=None,
    ):
        sharding_strategy = _get_sharding_strategy(input_height, input_width, in_channels, out_channels)
        conv_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=1,
            parameters=parameters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
            deallocate_activation=True,
            sharding_strategy=sharding_strategy,
            config_tensors_in_dram=True,
        )
        return TtConv2d(conv_config, self.device)

    def _create_3conv_layer(self, parameters, in_channels, out_channels, input_height, input_width):
        conv_layers = []

        conv1_params = parameters[0]
        sharding_strategy1 = _get_sharding_strategy(input_height, input_width, in_channels, in_channels // 4)
        conv1_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=in_channels // 4,
            batch_size=1,
            parameters=conv1_params,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            deallocate_activation=True,
            sharding_strategy=sharding_strategy1,
            config_tensors_in_dram=True,
        )
        conv_layers.append(TtConv2d(conv1_config, self.device))

        conv2_params = parameters[1]
        sharding_strategy2 = _get_sharding_strategy(input_height, input_width, in_channels // 4, in_channels // 4)
        conv2_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            batch_size=1,
            parameters=conv2_params,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,
            deallocate_activation=True,
            sharding_strategy=sharding_strategy2,
            config_tensors_in_dram=True,
        )
        conv_layers.append(TtConv2d(conv2_config, self.device))

        conv3_params = parameters[2]
        sharding_strategy3 = _get_sharding_strategy(input_height, input_width, in_channels // 4, out_channels)
        conv3_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels // 4,
            out_channels=out_channels,
            batch_size=1,
            parameters=conv3_params,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            deallocate_activation=True,
            sharding_strategy=sharding_strategy3,
            config_tensors_in_dram=True,
        )
        conv_layers.append(TtConv2d(conv3_config, self.device))

        return conv_layers

    def check_image_size(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Check and pad image size to be a multiple of window_size.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Padded tensor.
        """
        B, C, H, W = x.shape

        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size

        if mod_pad_h > 0 or mod_pad_w > 0:
            padding = ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w))
            x = ttnn.pad(x, padding, value=0.0, memory_config=self.memory_config)

        return x

    def forward_features(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, C, H, W).
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        if self.ape and self.absolute_pos_embed is not None:
            x = ttnn.add(x, self.absolute_pos_embed, memory_config=self.memory_config)

        for layer in self.layers:
            x = layer(x, x_size)

        if self.norm_weight is not None:
            B, L, C = x.shape
            x = ttnn.reshape(x, (1, B, L, C))
            x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias)
            x = ttnn.reshape(x, (B, L, C))

        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H*upscale, W*upscale).
        """
        B, C, H, W = x.shape

        x = self.check_image_size(x)

        x = ttnn.subtract(x, self.mean, memory_config=self.memory_config)
        x = ttnn.multiply(x, self.img_range, memory_config=self.memory_config)

        if self.upsampler == "pixelshuffle":
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = self.conv_first(x)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            # Reshape to recover spatial dimensions (conv_first may flatten output)
            # Get actual batch size from tensor after conv_first (may differ from input B in multi-device setups)
            x_shape = x.shape
            actual_B = x_shape[0]
            x = ttnn.reshape(x, (actual_B, H, W, self.embed_dim), memory_config=self.memory_config)
            x = ttnn.permute(x, (0, 3, 1, 2))  # Now [B, C, H, W]

            # Save shortcut AFTER reshape to ensure correct shape
            shortcut = ttnn.reallocate(x, memory_config=self.memory_config)
            x = self.forward_features(x)

            if isinstance(self.conv_after_body, list):
                x = ttnn.permute(x, (0, 2, 3, 1))
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                for i, conv_layer in enumerate(self.conv_after_body):
                    x = conv_layer(x)
                    x = ttnn.sharded_to_interleaved(x, self.memory_config)
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                    if i < len(self.conv_after_body) - 1:
                        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                        x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=self.memory_config)
                        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                # Reshape to recover spatial dimensions after 3conv
                # Use actual batch size from tensor (may differ from input B in multi-device setups)
                actual_B_conv = x.shape[0]
                x = ttnn.reshape(x, (actual_B_conv, H, W, self.embed_dim), memory_config=self.memory_config)
                x = ttnn.permute(x, (0, 3, 1, 2))
            else:
                x = ttnn.permute(x, (0, 2, 3, 1))
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = self.conv_after_body(x)
                x = ttnn.sharded_to_interleaved(x, self.memory_config)
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                # Reshape to recover spatial dimensions after 1conv
                # Use actual batch size from tensor (may differ from input B in multi-device setups)
                actual_B_conv = x.shape[0]
                x = ttnn.reshape(x, (actual_B_conv, H, W, self.embed_dim), memory_config=self.memory_config)
                x = ttnn.permute(x, (0, 3, 1, 2))

            x = ttnn.add(shortcut, x, memory_config=self.memory_config, dtype=ttnn.bfloat16)
            ttnn.deallocate(shortcut)

            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = self.conv_before_upsample(x)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.leaky_relu(x, negative_slope=0.01, memory_config=self.memory_config)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = self.upsample(x)
            x = ttnn.to_memory_config(x, self.memory_config)

            B_up, H_up, W_up, C_up = x.shape

            x = self.conv_last(x)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            if len(x.shape) == 4 and x.shape[1] == 1:
                out_channels = x.shape[-1]
                x = ttnn.reshape(x, (B_up, H_up, W_up, out_channels), memory_config=self.memory_config)

            x = ttnn.permute(x, (0, 3, 1, 2))

            x = ttnn.divide(x, self.img_range, memory_config=self.memory_config)
            x = ttnn.add(x, self.mean, memory_config=self.memory_config)

            output_h = H * self.upscale
            output_w = W * self.upscale

            B_out, C_out, H_out, W_out = x.shape
            if H_out > output_h or W_out > output_w:
                x = ttnn.slice(
                    x,
                    [0, 0, 0, 0],
                    [B_out, C_out, output_h, output_w],
                    memory_config=self.memory_config,
                )

            return x
        else:
            raise ValueError(f"Unsupported upsampler: {self.upsampler}")
