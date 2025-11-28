import warnings

import torch
import torch.nn as nn
import os

os.environ["MMCV_DISABLE_OPENCV"] = "1"

# from mmcv.runner.base_module import BaseModule
# from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple)
from mmcv.cnn.bricks.drop import build_dropout


# from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING, TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
class GroupMultiheadAttention(nn.Module):
    """Pure PyTorch version of MMCV GroupMultiheadAttention.

    Matches MMCV behavior:
    - Identity connection
    - Optional positional encodings
    - Grouped attention during training
    - Batch-first support
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        group=1,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__()

        if "dropout" in kwargs:
            warnings.warn(
                "`dropout` argument is deprecated; use attn_drop, proj_drop, dropout_layer instead",
                DeprecationWarning,
            )
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.group = group
        self.batch_first = batch_first

        # Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, batch_first=False)

        # Projection dropout
        self.proj_drop = nn.Dropout(proj_drop)

        # Residual dropout
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        # default K, V, identity
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        # positional encodings
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos
            else:
                warnings.warn("key_pos missing and query_pos shape mismatch.")

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # convert to (seq, batch, dim)
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        num_queries, bs, _ = query.shape

        # --------- GROUPED ATTENTION (training only) ---------
        if self.training and self.group > 1:
            # split queries into groups, concatenate across batch
            query = torch.cat(query.split(num_queries // self.group, dim=0), dim=1)
            key = torch.cat(key.split(num_queries // self.group, dim=0), dim=1)
            value = torch.cat(value.split(num_queries // self.group, dim=0), dim=1)

        # MultiheadAttention
        out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        if self.training and self.group > 1:
            # reverse grouping
            out = torch.cat(out.split(bs, dim=1), dim=0)

        # restore batch_first
        if self.batch_first:
            out = out.transpose(0, 1)

        # residual + dropout
        return identity + self.dropout_layer(self.proj_drop(out))
