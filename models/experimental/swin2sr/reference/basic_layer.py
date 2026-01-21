# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

###########################################################
# Adapted from https://github.com/mv-lab/swin2sr/tree/main
# licensed under Apache-2.0 license.
###########################################################

import torch
import torch.nn as nn

from models.experimental.swin2sr.reference.swin_transformer_block import SwinTransformerBlock


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

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
        pretrained_window_size (int): Local window size in pre-training.
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
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        if isinstance(drop_path, (list, tuple)):
            drop_path_list = drop_path
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_list[i],
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H*W, C).
            x_size: Spatial size (H, W).

        Returns:
            Output tensor.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
