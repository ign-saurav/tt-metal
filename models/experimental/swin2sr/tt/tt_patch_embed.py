# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.mobileNetV3.tt.utils import _create_conv_config_from_params
from models.tt_cnn.tt.builder import AutoShardedStrategyConfiguration


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

    def __call__(self, x):
        B, C, H, W = x.shape

        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        [x, [out_height, out_width]] = self.conv(x, return_output_dim=True)
        x = ttnn.sharded_to_interleaved(x, self.memory_config)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, out_height, out_width, self.embed_dim))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

        x = ttnn.reshape(x, (B, self.num_patches, self.embed_dim))

        if self.has_norm:
            norm_weight = self.parameters["norm"].get("weight", None)
            norm_bias = self.parameters["norm"].get("bias", None)
            x = ttnn.layer_norm(
                x,
                weight=norm_weight,
                bias=norm_bias,
                epsilon=1e-5,
                memory_config=self.norm_memory_config,
            )

        return x
