# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtLearnedPositionalEncoding:
    """TTNN implementation of LearnedPositionalEncoding.

    Args:
        params: Preprocessed parameters containing row_embed and col_embed weights.
        device: TTNN device.
        num_feats (int): Feature dimension for each position.
        row_num_embed (int): Number of row embeddings. Default: 50.
        col_num_embed (int): Number of column embeddings. Default: 50.
    """

    def __init__(
        self,
        params,
        device,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        self.row_embed = ttnn.embedding
        self.col_embed = ttnn.embedding
        self.params = params
        self.device = device
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def __call__(self, mask):
        """Forward function.

        Args:
            mask: Mask tensor with shape (B, H, W).

        Returns:
            Positional encoding with shape (B, num_feats*2, H, W).
        """
        # Extract height and width from last two dimensions (compatible with PyTorch version)
        # mask.shape is (B, H, W) for TT tensors
        if len(mask.shape) == 3:
            batch_size, h, w = mask.shape
        else:
            # Handle case where mask might have different shape
            h, w = mask.shape[-2:]
            batch_size = mask.shape[0] if len(mask.shape) >= 3 else 1

        x = ttnn.arange(w, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.arange(h, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_embed = self.col_embed(
            x,
            weight=self.params.col_embed.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        y_embed = self.row_embed(y, weight=self.params.row_embed.weight, layout=ttnn.TILE_LAYOUT)
        x_embed = ttnn.unsqueeze(x_embed, 0)
        x_embed = ttnn.repeat(x_embed, (h, 1, 1))
        y_embed = ttnn.unsqueeze(y_embed, 1)
        y_embed = ttnn.repeat(y_embed, (1, w, 1))

        out = ttnn.concat((x_embed, y_embed), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(y_embed)
        ttnn.deallocate(x_embed)
        out = ttnn.permute(out, (2, 0, 1))
        out = ttnn.unsqueeze(out, 0)
        out = ttnn.repeat(out, (batch_size, 1, 1, 1))
        pos = out
        return pos
