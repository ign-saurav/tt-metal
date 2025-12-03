# ---------------------------------------------
# Pure PyTorch implementation without mmcv
# ---------------------------------------------

import copy
import warnings
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


class FFN(nn.Module):
    """Feed Forward Network - MMCV-style implementation.

    Args:
        embed_dims (int): The feature dimension. Default: 256
    """

    def __init__(self, embed_dims=256):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.activate = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(self.embed_dims, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.1)),
            nn.Linear(512, self.embed_dims),
            nn.Dropout(p=0.1),
        )

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.layers(x)


class MyCustomBaseTransformerLayer(nn.Module):
    """Pure PyTorch Base Transformer Layer for vision transformer.

    It can be built with flexible customization, for example, using any number
    of FFN or LayerNorm and use different kinds of attention by providing
    pre-built attention modules. It supports prenorm when you specify 'norm'
    as the first element of operation_order.

    Args:
        attn_cfgs (list[dict] | dict | None): Configs or modules for self_attention
            or cross_attention. The order should be consistent with operation_order.
            If it is a dict, all attention modules in operation_order will be built
            with this config. Default: None.
        ffn_cfgs (list[dict] | dict | None): Configs for FFN. The order should be
            consistent with operation_order. If it is a dict, all FFN modules will
            be built with this config.
        operation_order (tuple[str]): The execution order of operations in transformer.
            Such as ('self_attn', 'norm', 'ffn', 'norm'). Support prenorm when you
            specify first element as 'norm'. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='LN').
        init_cfg (dict): Config for initialization. Default: None.
        batch_first (bool): Whether Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Default: True.
    """

    def __init__(
        self,
        attn_cfgs: Optional[Union[List[Union[Dict, nn.Module]], Dict, nn.Module]] = None,
        ffn_cfgs: Optional[Union[List[Dict], Dict]] = None,
        operation_order: Optional[tuple] = None,
        norm_cfg: dict = None,
        init_cfg: Optional[dict] = None,
        batch_first: bool = True,
        **kwargs,
    ):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="LN")

        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type="FFN",
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="ReLU", inplace=True),
            )

        # Handle deprecated arguments
        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments to a dict named `ffn_cfgs`."
                )
                if isinstance(ffn_cfgs, dict):
                    ffn_cfgs[new_name] = kwargs[ori_name]

        self.batch_first = batch_first

        # Validate operation_order
        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of {self.__class__.__name__} should contain all four "
            f"operation types {['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        # Process attention configs
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        # Handle attn_cfgs - can be dict, list of dicts, or pre-built modules
        if attn_cfgs is not None:
            if isinstance(attn_cfgs, (dict, nn.Module)):
                attn_cfgs = [
                    copy.deepcopy(attn_cfgs) if isinstance(attn_cfgs, dict) else attn_cfgs for _ in range(num_attn)
                ]
            else:
                assert num_attn == len(attn_cfgs), (
                    f"The length of attn_cfg {len(attn_cfgs)} is not consistent with "
                    f"the number of attention in operation_order {num_attn}."
                )
        else:
            attn_cfgs = [None for _ in range(num_attn)]

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"

        # Build attention modules
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if attn_cfgs[index] is None:
                    raise ValueError(f"Attention config at index {index} is None")

                # If it's already a module, use it directly
                if isinstance(attn_cfgs[index], nn.Module):
                    attention = attn_cfgs[index]
                else:
                    # Otherwise, it should be built externally
                    raise ValueError(
                        "attn_cfgs should contain pre-built nn.Module instances. "
                        "Please build attention modules before passing to this layer."
                    )

                # Store operation name for reference
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        # Get embed_dims from first attention
        if len(self.attentions) > 0:
            if hasattr(self.attentions[0], "embed_dims"):
                self.embed_dims = self.attentions[0].embed_dims
            elif hasattr(self.attentions[0], "embed_dim"):
                self.embed_dims = self.attentions[0].embed_dim
            else:
                # Try to infer from parameters
                self.embed_dims = 256  # default fallback
                warnings.warn("Cannot infer embed_dims from attention module, using default 256")
        else:
            self.embed_dims = 256

        # Build FFN modules
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")

        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]

        assert len(ffn_cfgs) == num_ffns, f"Length of ffn_cfgs {len(ffn_cfgs)} != num_ffns {num_ffns}"

        for ffn_index in range(num_ffns):
            ffn_cfg = copy.deepcopy(ffn_cfgs[ffn_index])

            if "embed_dims" not in ffn_cfg:
                ffn_cfg["embed_dims"] = self.embed_dims
            else:
                assert (
                    ffn_cfg["embed_dims"] == self.embed_dims
                ), f"FFN embed_dims {ffn_cfg['embed_dims']} != layer embed_dims {self.embed_dims}"

            # Build FFN
            if isinstance(ffn_cfg, nn.Module):
                ffn = ffn_cfg
            else:
                # Remove 'type' if present
                ffn_cfg.pop("type", None)
                ffn = FFN(**ffn_cfg)

            self.ffns.append(ffn)

        # Build normalization layers
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(self._build_norm_layer(norm_cfg, self.embed_dims))

    def _build_norm_layer(self, norm_cfg, embed_dims):
        """Build normalization layer."""
        norm_type = norm_cfg.get("type", "LN")

        if norm_type == "LN" or norm_type == "LayerNorm":
            return nn.LayerNorm(embed_dims)
        elif norm_type == "BN" or norm_type == "BatchNorm1d":
            return nn.BatchNorm1d(embed_dims)
        else:
            return nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward function for TransformerLayer.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape [num_queries, bs, embed_dims]
                if self.batch_first is False, else [bs, num_queries, embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs, embed_dims]
                if self.batch_first is False, else [bs, num_keys, embed_dims].
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in calculation of
                corresponding attention. The length of it should equal to the
                number of `attention` in `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Process attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in {self.__class__.__name__}")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of attn_masks {len(attn_masks)} must be equal to "
                f"the number of attention in operation_order {self.num_attn}"
            )

        # Execute operations in order
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
