# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class FFN(nn.Module):
    """Feed Forward Network for transformer layers (inference-only).

    Args:
        embed_dims (int): The embedding dimension. Default: 256.
        feedforward_channels (int): Hidden dimension of FFN. Default: 1024.
        num_fcs (int): Number of fully connected layers. Default: 2.
        act_cfg (str): Activation function type. Default: 'relu'.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: str = "relu",
    ):
        super().__init__()
        assert num_fcs >= 2, f"num_fcs should be >= 2, got {num_fcs}"

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        # Build layers
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            if act_cfg == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif act_cfg == "gelu":
                layers.append(nn.GELU())
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        identity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input tensor with shape (bs, num_queries, embed_dims).
            identity: Tensor for residual connection. Default: None.

        Returns:
            Output tensor with shape (bs, num_queries, embed_dims).
        """
        if identity is None:
            identity = x
        out = self.layers(x)
        return out + identity


class MyCustomBaseTransformerLayer(nn.Module):
    """Base TransformerLayer for vision transformer (inference-only).

    This layer supports flexible customization with any number of FFN, LayerNorm,
    self-attention and cross-attention modules by specifying operation_order.

    Args:
        attentions (List[nn.Module]): List of attention modules. The order should
            match the attention operations in operation_order.
        embed_dims (int): The embedding dimension. Default: 256.
        feedforward_channels (int): Hidden dimension of FFN. Default: 1024.
        num_fcs (int): Number of fully connected layers in FFN. Default: 2.
        operation_order (tuple[str]): The execution order of operations.
            Such as ('self_attn', 'norm', 'ffn', 'norm', 'cross_attn', 'norm').
        act_cfg (str): Activation function type for FFN. Default: 'relu'.
        batch_first (bool): Whether batch is the first dimension. Default: True.
    """

    def __init__(
        self,
        attentions: List[nn.Module],
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        operation_order: Tuple[str, ...] = None,
        act_cfg: str = "relu",
        batch_first: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.operation_order = operation_order

        # Count operations
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        num_ffns = operation_order.count("ffn")
        num_norms = operation_order.count("norm")

        assert len(attentions) == num_attn, (
            f"Number of attentions ({len(attentions)}) must match "
            f"number of attention ops in operation_order ({num_attn})"
        )

        self.num_attn = num_attn
        self.pre_norm = operation_order[0] == "norm"

        # Store attention modules
        self.attentions = nn.ModuleList(attentions)

        # Build FFN modules
        self.ffns = nn.ModuleList(
            [
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels,
                    num_fcs=num_fcs,
                    act_cfg=act_cfg,
                )
                for _ in range(num_ffns)
            ]
        )

        # Build LayerNorm modules
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for TransformerLayer.

        Args:
            query: Input query with shape (bs, num_queries, embed_dims) if batch_first.
            key: Key tensor with shape (bs, num_keys, embed_dims) if batch_first.
            value: Value tensor with same shape as key.
            query_pos: Positional encoding for query. Default: None.
            key_pos: Positional encoding for key. Default: None.
            attn_masks: List of attention masks. Default: None.
            query_key_padding_mask: Padding mask for query. Default: None.
            key_padding_mask: Padding mask for key. Default: None.

        Returns:
            Output tensor with shape (bs, num_queries, embed_dims).
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attn_masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks for _ in range(self.num_attn)]

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class MyCustomBaseTransformerLayerWithoutSelfAttn(nn.Module):
    """Base TransformerLayer without self-attention (inference-only).

    This is optimized for decoder layers that only need cross-attention.

    Args:
        attentions (List[nn.Module]): List of cross-attention modules. The order
            should match the cross_attn operations in operation_order.
        embed_dims (int): The embedding dimension. Default: 256.
        feedforward_channels (int): Hidden dimension of FFN. Default: 1024.
        num_fcs (int): Number of fully connected layers in FFN. Default: 2.
        operation_order (tuple[str]): The execution order of operations.
            Such as ('norm', 'cross_attn', 'norm', 'ffn', 'norm').
        act_cfg (str): Activation function type for FFN. Default: 'relu'.
        batch_first (bool): Whether batch is the first dimension. Default: True.
    """

    def __init__(
        self,
        attentions: List[nn.Module],
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        operation_order: Tuple[str, ...] = None,
        act_cfg: str = "relu",
        batch_first: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.operation_order = operation_order

        # Count operations (no self_attn)
        num_attn = operation_order.count("cross_attn")
        num_ffns = operation_order.count("ffn")
        num_norms = operation_order.count("norm")

        assert len(attentions) == num_attn, (
            f"Number of attentions ({len(attentions)}) must match "
            f"number of cross_attn ops in operation_order ({num_attn})"
        )

        self.num_attn = num_attn
        self.pre_norm = operation_order[0] == "norm"

        # Store attention modules
        self.attentions = nn.ModuleList(attentions)

        # Build FFN modules
        self.ffns = nn.ModuleList(
            [
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels,
                    num_fcs=num_fcs,
                    act_cfg=act_cfg,
                )
                for _ in range(num_ffns)
            ]
        )

        # Build LayerNorm modules
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for TransformerLayer without self-attention.

        Args:
            query: Input query with shape (bs, num_queries, embed_dims) if batch_first.
            key: Key tensor with shape (bs, num_keys, embed_dims) if batch_first.
            value: Value tensor with same shape as key.
            query_pos: Positional encoding for query. Default: None.
            key_pos: Positional encoding for key. Default: None.
            attn_masks: List of attention masks. Default: None.
            query_key_padding_mask: Padding mask for query (unused). Default: None.
            key_padding_mask: Padding mask for key. Default: None.

        Returns:
            Output tensor with shape (bs, num_queries, embed_dims).
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attn_masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks for _ in range(self.num_attn)]

        for layer in self.operation_order:
            if layer == "self_attn":
                # Note: This branch is typically not reached since this class
                # is designed for layers without self-attention
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
