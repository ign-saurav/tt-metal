# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ttnn

import warnings
from functools import partial
from typing import Optional
from typing import Tuple, List

from models.common.utility_functions import (
    tt2torch_tensor,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.functional import *
from torch.nn.init import trunc_normal_
from torch.nn.modules.activation import *
from transformers.integrations import is_deepspeed_zero3_enabled


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class TTResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        parameters,
        device,
        input_dtype,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        print("Initializing Resampler...")
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size
        self.kv_dim = kv_dim
        self.parameters = parameters

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        print(embed_dim, num_heads)
        self.attn = TTMultiheadAttention(
            embed_dim, num_heads, self.parameters["attn"], input_dtype=input_dtype, tt_device=device
        )
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        if is_deepspeed_zero3_enabled():
            device = "cuda"
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = torch.device("cpu")
        dtype = torch.bfloat16

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        # x = self.kv_proj(x)  # B * L * D
        if self.kv_dim is not None and self.kv_dim != self.embed_dim:
            x = ttnn.linear(
                x,
                self.parameters["kv_proj"]["weight"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
        else:
            x = ttnn.identity(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.layer_norm(
            x,
            weight=self.parameters["ln_kv"]["weight"],
            bias=self.parameters["ln_kv"]["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.permute(x, (1, 0, 2))  # L * B * D
        q = ttnn.layer_norm(
            self.parameters["query"],
            weight=self.parameters["ln_q"]["weight"],
            bias=self.parameters["ln_q"]["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x = tt2torch_tensor(x)
        q = tt2torch_tensor(q)
        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = ttnn.permute(out, (1, 0, 2))  # B * Q * D

        x = ttnn.layer_norm(
            x,
            weight=self.parameters["ln_post"]["weight"],
            bias=self.parameters["ln_post"]["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.matmul(x, self.parameters["proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

    # def _tt_repeat(self, query, N: int):
    #     # query shape: (Q, D)
    #     # Add dimension at position 1: (Q, 1, D)
    #     query_unsqueezed = ttnn.unsqueeze(query, dim=1)

    #     # Repeat N times along dimension 1: (Q, N, D)
    #     query_repeated = ttnn.repeat(query_unsqueezed, ttnn.Shape([1, N, 1]))

    #     return query_repeated


class TTMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        parameters,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        tt_device=None,
        dtype=None,
        input_dtype=None,
    ):
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype
        )
        self.tt_device = tt_device
        self.input_dtype = input_dtype

        # rewrite out_proj layer，with nn.Linear
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.parameters = parameters

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    print("1")
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            tt_query = ttnn.from_torch(
                query,
                device=self.tt_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.input_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_key = ttnn.from_torch(
                key,
                device=self.tt_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.input_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_value = ttnn.from_torch(
                value,
                device=self.tt_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.input_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            attn_output = self.multi_head_attention_forward(
                self.tt_device,
                self.input_dtype,
                tt_query,
                tt_key,
                tt_value,
                self.embed_dim,
                self.num_heads,
                self.parameters["in_proj_weight"],
                self.parameters["in_proj_bias"],
                self.parameters["bias_k"],
                self.parameters["bias_v"],
                self.add_zero_attn,
                self.dropout,
                self.parameters["out_proj_weight"],
                self.parameters["out_proj_bias"],
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0)
        else:
            return attn_output

    def multi_head_attention_forward(
        self,
        device,
        input_dtype,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)

        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = ttnn.unsqueeze(query, dim=1)
            key = ttnn.unsqueeze(key, dim=1)
            value = ttnn.unsqueeze(value, dim=1)
            if key_padding_mask is not None:
                key_padding_mask = ttnn.unsqueeze(key_padding_mask, dim=0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=torch.bfloat16
            # target_type=query.dtype
        )
        key_padding_mask = ttnn.from_torch(
            key_padding_mask,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        assert (
            embed_dim == embed_dim_to_check
        ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert (
                key.shape[:2] == value.shape[:2]
            ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        # else:
        #     assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        #     assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        #     assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        #     if in_proj_bias is None:
        #         b_q = b_k = b_v = None
        #     else:
        #         b_q, b_k, b_v = ttnn.chunk(in_proj_bias, 3, dim=1)
        #     q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # prep attention mask

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if len(attn_mask.shape) == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = ttnn.unsqueeze(attn_mask, 0)
            elif len(attn_mask.shape) == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {len(attn_mask.shape)} is not supported")

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = ttnn.cat([k, ttnn.repeat(bias_k, (1, bsz, 1))], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v = ttnn.cat([v, ttnn.repeat(bias_v, (1, bsz, 1))], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if attn_mask is not None:
                attn_mask = ttnn.pad(attn_mask, (0, 1), mode="constant", value=0)
            if key_padding_mask is not None:
                key_padding_mask = ttnn.pad(key_padding_mask, (0, 1), mode="constant", value=0)
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = ttnn.reshape(q, (tgt_len, bsz * num_heads, head_dim))
        q = ttnn.transpose(q, 0, 1)
        if static_k is None:
            k = ttnn.reshape(k, (k.shape[0], bsz * num_heads, head_dim))
            k = ttnn.transpose(k, 0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_k.shape[0] == bsz * num_heads
            ), f"expecting static_k.shape[0] of {bsz * num_heads}, but got {static_k.shape[0]}"
            assert (
                static_k.shape[2] == head_dim
            ), f"expecting static_k.shape[2] of {head_dim}, but got {static_k.shape[2]}"
            k = static_k
        if static_v is None:
            v = ttnn.reshape(v, (v.shape[0], bsz * num_heads, head_dim))
            v = ttnn.transpose(v, 0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_v.shape[0] == bsz * num_heads
            ), f"expecting static_v.shape[0] of {bsz * num_heads}, but got {static_v.shape[0]}"
            assert (
                static_v.shape[2] == head_dim
            ), f"expecting static_v.shape[2] of {head_dim}, but got {static_v.shape[2]}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.shape[1]

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"

            key_padding_mask = ttnn.reshape(key_padding_mask, (bsz, 1, 1, src_len))

            key_padding_mask = ttnn.expand(key_padding_mask, [-1, num_heads, -1, -1])

            key_padding_mask = ttnn.reshape(key_padding_mask, (bsz * num_heads, 1, src_len))
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and len(attn_mask.shape) == 3:
                attn_mask = ttnn.unsqueeze(attn_mask, 0)
            else:
                attn_mask = ttnn.reshape(attn_mask, (bsz, num_heads, -1, src_len))

        # Scaled dot product attention - single fused operation
        q = ttnn.unsqueeze(q, 0)
        k = ttnn.unsqueeze(k, 0)
        v = ttnn.unsqueeze(v, 0)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=None,
            program_config=None,
            compute_kernel_config=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output = ttnn.permute(attn_output, (2, 0, 1, 3))
        attn_output = ttnn.reshape(attn_output, (bsz * tgt_len, embed_dim))

        attn_output = ttnn.linear(attn_output, out_proj_weight, bias=out_proj_bias)

        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, -1))

        if not is_batched:
            attn_output = ttnn.squeeze(attn_output, 1)
        return attn_output, None


def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if len(query.shape) == 3:
        # Batched Inputs
        is_batched = True
        assert len(key.shape) == 3 and len(value.shape) == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert len(key_padding_mask.shape) == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {len(key_padding_mask.shape)}-D tensor instead"
            )
        if attn_mask is not None:
            assert len(attn_mask.shape) in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {len(attn_mask.shape)}-D tensor instead"
            )
    elif len(query.shape) == 2:
        # Unbatched Inputs
        is_batched = False
        assert len(key.shape) == 2 and len(value.shape) == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {len(key.shape)}-D and {len(value.shape)}-D tensors respectively"
        )

        if key_padding_mask is not None:
            assert len(key_padding_mask.shape) == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {len(key_padding_mask.shape)}-D tensor instead"
            )

        if attn_mask is not None:
            assert len(attn_mask.shape) in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {len(attn_mask.shape)}-D tensor instead"
            )
            if len(attn_mask.shape) == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert (
                    attn_mask.shape == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {len(attn_mask.shape)}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {len(query.shape)}-D query tensor"
        )

    return is_batched


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type: Optional[DType],
    other_name: str,
    target_type: DType,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            # proj = linear(q, w, b)
            proj = ttnn.linear(
                q,
                w,
                b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=q.shape[0], x=8),
            )
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            # q_proj = linear(q, w_q, b_q)
            q_proj = ttnn.linear(
                q,
                w_q,
                b_q,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=q.shape[0], x=8),
            )
            # kv_proj = linear(k, w_kv, b_kv)
            kv_proj = ttnn.linear(
                k,
                w_kv,
                b_kv,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=k.shape[0], x=8),
            )
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        # w_q, w_k, w_v = w.chunk(3)
        w_q, w_k, w_v = ttnn.chunk(w, 3, dim=1)
        if b is None:
            b_q = b_k = b_v = None
        else:
            # b_q, b_k, b_v = b.chunk(3)
            b_q, b_k, b_v = ttnn.chunk(b, 3, dim=1)

        # q_proj, k_proj, v_proj = linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

        # q = ttnn.permute(q, (0, 2, 1))

        q_proj = ttnn.linear(
            q,
            w_q,
            bias=b_q,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=q.shape[0], x=8),
        )
        k_proj = ttnn.linear(
            k,
            w_k,
            bias=b_k,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=k.shape[0], x=8),
        )
        v_proj = ttnn.linear(
            v,
            w_v,
            bias=b_v,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=v.shape[0], x=8),
        )
        return q_proj, k_proj, v_proj


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
