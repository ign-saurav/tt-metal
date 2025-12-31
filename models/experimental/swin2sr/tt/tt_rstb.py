# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.swin2sr.tt.tt_basic_layer import TtBasicLayer
from models.experimental.swin2sr.tt.tt_patch_embed import TtSwin2SRPatchEmbed, TtSwin2SRPatchUnEmbed
from models.tt_cnn.tt.builder import TtConv2d, AutoShardedStrategyConfiguration
from models.experimental.swin2sr.tt.utils import _create_conv_config_from_params


class TtRSTB:
    """Residual Swin Transformer Block (RSTB).

    Args:
        device: TTNN device.
        parameters: Model parameters dictionary.
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection. '1conv' or '3conv'.
        memory_config: Memory configuration for tensors.
    """

    def __init__(
        self,
        device,
        parameters,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        resi_connection: str = "1conv",
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.input_resolution = input_resolution
        self.memory_config = memory_config
        self.l1 = ttnn.L1_MEMORY_CONFIG
        self.dram = ttnn.DRAM_MEMORY_CONFIG

        # Residual group (BasicLayer)
        self.residual_group = TtBasicLayer(
            device=device,
            parameters=parameters["residual_group"],
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            memory_config=memory_config,
        )

        # Convolutional block
        if resi_connection == "1conv":
            conv_params = parameters["conv"]
            conv_config = _create_conv_config_from_params(
                input_height=input_resolution[0],
                input_width=input_resolution[1],
                in_channels=dim,
                out_channels=dim,
                batch_size=1,
                parameters=conv_params,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
                config_tensors_in_dram=True,
            )
            self.conv = TtConv2d(conv_config, device)
        elif resi_connection == "3conv":
            # Sequential: Conv2d(dim, dim//4, 3, 1, 1) -> LeakyReLU -> Conv2d(dim//4, dim//4, 1, 1, 0) -> LeakyReLU -> Conv2d(dim//4, dim, 3, 1, 1)
            conv1_params = parameters["conv"][0]
            conv2_params = parameters["conv"][2]
            conv3_params = parameters["conv"][4]

            # First conv: dim -> dim//4
            conv1_config = _create_conv_config_from_params(
                input_height=input_resolution[0],
                input_width=input_resolution[1],
                in_channels=dim,
                out_channels=dim // 4,
                batch_size=1,
                parameters=conv1_params,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
            )
            self.conv1 = TtConv2d(conv1_config, device)

            # Second conv: dim//4 -> dim//4 (1x1 conv)
            conv2_config = _create_conv_config_from_params(
                input_height=input_resolution[0],
                input_width=input_resolution[1],
                in_channels=dim // 4,
                out_channels=dim // 4,
                batch_size=1,
                parameters=conv2_params,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
            )
            self.conv2 = TtConv2d(conv2_config, device)

            # Third conv: dim//4 -> dim
            conv3_config = _create_conv_config_from_params(
                input_height=input_resolution[0],
                input_width=input_resolution[1],
                in_channels=dim // 4,
                out_channels=dim,
                batch_size=1,
                parameters=conv3_params,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
            )
            self.conv3 = TtConv2d(conv3_config, device)
            self.conv = None  # Mark that we're using 3conv
            self.leaky_relu_negative_slope = 0.2  # Default negative slope for LeakyReLU
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        self.resi_connection = resi_connection

        # Patch embed and unembed
        self.patch_embed = TtSwin2SRPatchEmbed(
            device=device,
            parameters=parameters["patch_embed"],
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=None,
            memory_config=memory_config,
        )

        self.patch_unembed = TtSwin2SRPatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            memory_config=memory_config,
        )

    def __call__(self, x: ttnn.Tensor, x_size: tuple[int, int]) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H*W, C).
            x_size: Spatial size (H, W).

        Returns:
            Output tensor of shape (B, H*W, C).
        """
        shortcut = ttnn.reallocate(x, memory_config=self.dram)

        # Residual group
        x = self.residual_group(x, x_size)

        # Patch unembed: (B, H*W, C) -> (B, C, H, W)
        x = self.patch_unembed(x, x_size)

        # Convolution
        if self.resi_connection == "1conv":
            # Convert to (B, H, W, C) for conv2d
            x = ttnn.permute(x, (0, 2, 3, 1), memory_config=self.dram)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)
            x, _ = self.conv(x, return_output_dim=True)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)
            # Reshape back to (B, H, W, C) then to (B, C, H, W)
            B, H, W, C = x.shape
            x = ttnn.permute(x, (0, 3, 1, 2), memory_config=self.dram)
        else:  # 3conv
            # Convert to (B, H, W, C) for conv2d
            x = ttnn.permute(x, (0, 2, 3, 1), memory_config=self.dram)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)

            # First conv
            x, _ = self.conv1(x, return_output_dim=True)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)
            # Apply LeakyReLU
            x = ttnn.leaky_relu(x, negative_slope=self.leaky_relu_negative_slope, memory_config=self.dram)

            # Second conv (1x1)
            x, _ = self.conv2(x, return_output_dim=True)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)
            # Apply LeakyReLU
            x = ttnn.leaky_relu(x, negative_slope=self.leaky_relu_negative_slope, memory_config=self.dram)

            # Third conv
            x, _ = self.conv3(x, return_output_dim=True)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram)

            # Reshape back to (B, C, H, W)
            B, H, W, C = x.shape
            x = ttnn.permute(x, (0, 3, 1, 2), memory_config=self.dram)

        # Patch embed: (B, C, H, W) -> (B, H*W, C)
        x = self.patch_embed(x)

        # Residual connection
        x = ttnn.add(shortcut, x, memory_config=self.l1, dtype=ttnn.bfloat16)
        ttnn.deallocate(shortcut)

        return x
