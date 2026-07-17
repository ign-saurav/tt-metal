# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import AutoShardedStrategyConfiguration
from models.experimental.swin2sr.tt.utils import _create_conv_config_from_params


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (x, x)
    return x


class TtSwin2SRPatchEmbed:
    def __init__(
        self,
        device,
        parameters,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.memory_config = memory_config

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.has_norm = norm_layer is not None

        if self.has_norm:
            self.norm_memory_config = memory_config

        conv_config = _create_conv_config_from_params(
            input_height=img_size[0],
            input_width=img_size[1],
            in_channels=in_chans,
            out_channels=embed_dim,
            batch_size=1,
            parameters=parameters["proj"],
            kernel_size=patch_size,
            stride=patch_size,
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv = TtConv2d(conv_config, device)

        if self.has_norm:
            self.norm_weight = self.parameters["norm"].get("weight", None)
            self.norm_bias = self.parameters["norm"].get("bias", None)

    def __call__(self, x):
        B, C, H, W = x.shape

        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        [x, [out_height, out_width]] = self.conv(x, return_output_dim=True)
        x = ttnn.sharded_to_interleaved(x, self.memory_config)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, out_height, out_width, self.embed_dim))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

        num_patches = out_height * out_width
        x = ttnn.reshape(x, (B, num_patches, self.embed_dim))

        if self.has_norm:
            x = ttnn.reshape(x, (1, B, num_patches, self.embed_dim))
            x = ttnn.layer_norm(
                x,
                weight=self.norm_weight,
                bias=self.norm_bias,
            )
            x = ttnn.reshape(x, (B, num_patches, self.embed_dim))

        return x


class TtSwin2SRPatchUnEmbed:
    """Patch to Image Unembedding.

    Args:
        img_size (int | tuple[int]): Image size. Default: 224.
        patch_size (int | tuple[int]): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.memory_config = memory_config

    def __call__(self, x: ttnn.Tensor, x_size: tuple[int, int]) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, num_patches, embed_dim).
            x_size: Spatial size (H, W) of the output.

        Returns:
            Output tensor of shape (B, embed_dim, H, W).
        """
        B, HW, C = x.shape
        H, W = x_size

        # Convert to ROW_MAJOR for proper reshaping (TTNN reshape requires ROW_MAJOR)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=self.memory_config)
        x = ttnn.reshape(x, (B, H, W, C), memory_config=self.memory_config)
        x = ttnn.permute(x, (0, 3, 1, 2), memory_config=self.memory_config)

        return x
