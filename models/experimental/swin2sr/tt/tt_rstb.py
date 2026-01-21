# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.swin2sr.tt.tt_basic_layer import TtBasicLayer
from models.experimental.swin2sr.tt.tt_patch_embed import TtSwin2SRPatchEmbed, TtSwin2SRPatchUnEmbed
from models.experimental.swin2sr.tt.utils import _create_conv_config_from_params, _get_sharding_strategy
from models.tt_cnn.tt.builder import TtConv2d


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (x, x)
    return x


class TtRSTB:
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

        img_size = to_2tuple(img_size)
        H, W = input_resolution

        if resi_connection == "1conv":
            conv_params = parameters["conv"]
            sharding_strategy = _get_sharding_strategy(H, W, dim, dim)
            conv_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
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
                deallocate_activation=True,
                sharding_strategy=sharding_strategy,
                config_tensors_in_dram=True,
            )
            self.conv = TtConv2d(conv_config, device)
            self.conv_layers = [self.conv]
        elif resi_connection == "3conv":
            self.conv_layers = []

            conv1_params = parameters["conv"][0]
            sharding_strategy1 = _get_sharding_strategy(H, W, dim, dim // 4)
            conv1_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
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
                activation_dtype=ttnn.bfloat16,
                deallocate_activation=True,
                sharding_strategy=sharding_strategy1,
                config_tensors_in_dram=True,
            )
            self.conv1 = TtConv2d(conv1_config, device)
            self.conv_layers.append(self.conv1)

            conv2_params = parameters["conv"][1]
            sharding_strategy2 = _get_sharding_strategy(H, W, dim // 4, dim // 4)
            conv2_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
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
                activation_dtype=ttnn.bfloat16,
                deallocate_activation=True,
                sharding_strategy=sharding_strategy2,
                config_tensors_in_dram=True,
            )
            self.conv2 = TtConv2d(conv2_config, device)
            self.conv_layers.append(self.conv2)

            conv3_params = parameters["conv"][2]
            sharding_strategy3 = _get_sharding_strategy(H, W, dim // 4, dim)
            conv3_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
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
                deallocate_activation=True,
                sharding_strategy=sharding_strategy3,
                config_tensors_in_dram=True,
            )
            self.conv3 = TtConv2d(conv3_config, device)
            self.conv_layers.append(self.conv3)
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

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
        shortcut = ttnn.reallocate(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = self.residual_group(x, x_size=x_size)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.patch_unembed(x, x_size)

        x = ttnn.permute(x, (0, 2, 3, 1), memory_config=self.memory_config)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)

            is_last_conv = i == len(self.conv_layers) - 1
            needs_leaky_relu = len(self.conv_layers) == 3 and i < 2

            if needs_leaky_relu or is_last_conv:
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            if needs_leaky_relu:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            if not is_last_conv:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.permute(x, (0, 3, 1, 2), memory_config=self.memory_config)
        x = self.patch_embed(x)

        x = ttnn.add(shortcut, x, memory_config=self.memory_config, dtype=ttnn.bfloat16)
        ttnn.deallocate(shortcut)

        return x
