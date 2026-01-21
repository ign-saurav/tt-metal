# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
from models.experimental.swin2sr.tt.utils import _create_conv_config_from_params, _get_sharding_strategy
from models.tt_cnn.tt.builder import TtConv2d


class TtUpsample:
    """Upsample module using PixelShuffle.

    Args:
        device: TTNN device.
        parameters: Dictionary of parameters for conv layers.
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        input_height (int): Input height.
        input_width (int): Input width.
        memory_config: Memory configuration. Default: DRAM_MEMORY_CONFIG.
    """

    def __init__(
        self,
        device,
        parameters,
        scale: int,
        num_feat: int,
        input_height: int,
        input_width: int,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.scale = scale
        self.num_feat = num_feat
        self.input_height = input_height
        self.input_width = input_width
        self.memory_config = memory_config

        if (scale & (scale - 1)) == 0:
            self.num_ops = int(math.log(scale, 2))
            self.scale_factor = 2
        elif scale == 3:
            self.num_ops = 1
            self.scale_factor = 3
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")

        self.conv_layers = []
        current_height = input_height
        current_width = input_width
        current_channels = num_feat

        for i in range(self.num_ops):
            out_channels = current_channels * (self.scale_factor * self.scale_factor)
            conv_params = parameters[i] if isinstance(parameters, list) else parameters[f"{i}"]

            sharding_strategy = _get_sharding_strategy(current_height, current_width, current_channels, out_channels)
            conv_config = _create_conv_config_from_params(
                input_height=current_height,
                input_width=current_width,
                in_channels=current_channels,
                out_channels=out_channels,
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
            conv_layer = TtConv2d(conv_config, device)
            self.conv_layers.append(conv_layer)

            current_height = current_height * self.scale_factor
            current_width = current_width * self.scale_factor
            current_channels = num_feat

    def pixel_shuffle(self, x: ttnn.Tensor, upscale_factor: int) -> ttnn.Tensor:
        """Apply PixelShuffle operation using pure TTNN operations.

        Pixel shuffle rearranges channels into spatial dimensions:
        - Input: [B, H, W, C*r^2] where r is upscale_factor
        - Output: [B, H*r, W*r, C]

        Args:
            x: Input tensor in NHWC format with shape [B, H, W, C*r^2].
            upscale_factor: Upscale factor for pixel shuffle (r).

        Returns:
            Output tensor in NHWC format with shape [B, H*r, W*r, C].
        """
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        B, H, W, C = x.shape

        out_channels = C // (upscale_factor * upscale_factor)

        x = ttnn.reshape(x, (B, H, W, out_channels, upscale_factor, upscale_factor), memory_config=self.memory_config)
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3), memory_config=self.memory_config)
        x = ttnn.reshape(x, (B, H * upscale_factor, W * upscale_factor, out_channels), memory_config=self.memory_config)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C) in NHWC format.

        Returns:
            Output tensor of shape (B, H*scale, W*scale, C).
        """
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_memory_config(x, self.memory_config)

        current_height = self.input_height
        current_width = self.input_width

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (1, current_height, current_width, -1), memory_config=self.memory_config)

            x = self.pixel_shuffle(x, self.scale_factor)
            x = ttnn.to_memory_config(x, self.memory_config)

            current_height = current_height * self.scale_factor
            current_width = current_width * self.scale_factor

        return x
