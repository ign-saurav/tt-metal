# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

###########################################################
# Adapted from https://github.com/mv-lab/swin2sr/tree/main
# licensed under Apache-2.0 license.
###########################################################

import torch
import torch.nn as nn


def to_2tuple(x):
    """Convert input to a tuple of 2 elements."""
    if isinstance(x, (int, float)):
        return (x, x)
    return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple[int]): Image size. Default: 224.
        patch_size (int | tuple[int]): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: nn.Module = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding.

    Args:
        img_size (int | tuple[int]): Image size. Default: 224.
        patch_size (int | tuple[int]): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: nn.Module = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, num_patches, embed_dim).
            x_size: Spatial size (H, W) of the output.

        Returns:
            Output tensor of shape (B, embed_dim, H, W).
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B C H W
        return x
