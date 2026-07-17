# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

###########################################################
# Adapted from https://github.com/mv-lab/swin2sr/tree/main
# licensed under Apache-2.0 license.
###########################################################


import torch
import torch.nn as nn

from models.experimental.swin2sr.reference.basic_layer import BasicLayer
from models.experimental.swin2sr.reference.patch_embed import PatchEmbed, PatchUnEmbed


def to_2tuple(x):
    """Convert input to a tuple of 2 elements."""
    if isinstance(x, (int, float)):
        return (x, x)
    return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection. '1conv' or '3conv'.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | tuple[float] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = None,
        use_checkpoint: bool = False,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        resi_connection: str = "1conv",
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # To save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H*W, C).
            x_size: Spatial size (H, W).

        Returns:
            Output tensor of shape (B, H*W, C).
        """
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
