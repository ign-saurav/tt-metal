# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV queries.

    Args:
        num_feats (int): Feature dimension for each position.
        row_num_embed (int): Number of row embeddings. Default: 50.
        col_num_embed (int): Number of column embeddings. Default: 50.
    """

    def __init__(
        self,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
    ):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            mask: Mask tensor with shape (B, H, W).

        Returns:
            Positional encoding with shape (B, num_feats*2, H, W).
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


class SinePositionalEncoding(nn.Module):
    """Sine positional encoding for BEV queries.

    Args:
        num_feats (int): Feature dimension for each position along x/y axis.
            The final output dimension is 2 * num_feats.
        temperature (int): Temperature for scaling. Default: 10000.
        normalize (bool): Whether to normalize. Default: False.
        scale (float): Scale factor for normalization. Default: 2*pi.
        eps (float): Small value for numerical stability. Default: 1e-6.
        offset (float): Offset for normalization. Default: 0.0.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ):
        super().__init__()
        if normalize:
            assert isinstance(
                scale, (float, int)
            ), "when normalize is set, scale should be provided and in float or int type"
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            mask: Mask tensor with shape (B, H, W).

        Returns:
            Positional encoding with shape (B, num_feats*2, H, W).
        """
        # Convert mask to int for cumsum
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
