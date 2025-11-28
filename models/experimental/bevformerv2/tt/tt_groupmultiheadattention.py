# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


class TtMultiheadAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        params = params.attn
        super().__init__()
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.attn_in_proj__weight = params.in_proj_weight
        self.attn_in_proj__weight = ttnn.to_layout(self.attn_in_proj__weight, layout=ttnn.TILE_LAYOUT)
        self.attn_in_proj__bias = params.in_proj_bias
        self.attn_in_proj__bias = ttnn.to_layout(self.attn_in_proj__bias, layout=ttnn.TILE_LAYOUT)
        # Canonicalize fused QKV weight to packed shape [3 * embed_dims, embed_dims]
        w = self.attn_in_proj__weight
        if w.shape[0] == 3 * self.embed_dims and w.shape[1] == self.embed_dims:
            # Already in PyTorch layout [3*D, D]
            self.attn_in_proj__packed = w
        elif w.shape[0] == self.embed_dims and w.shape[1] == 3 * self.embed_dims:
            # Transposed layout [D, 3*D] (e.g., produced via preprocess_linear_weight) – transpose back
            self.attn_in_proj__packed = ttnn.permute(w, (1, 0))
        else:
            raise ValueError(
                f"Unexpected in_proj_weight shape {w.shape}; expected (3*{self.embed_dims}, {self.embed_dims}) "
                f"or ({self.embed_dims}, 3*{self.embed_dims})"
            )
        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_weight = ttnn.to_layout(self.attn_out_proj_weight, layout=ttnn.TILE_LAYOUT)
        self.attn_out_proj_bias = params.out_proj.bias
        self.attn_out_proj_bias = ttnn.to_layout(self.attn_out_proj_bias, layout=ttnn.TILE_LAYOUT)

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=None,
        **kwargs,
    ):
        if batch_first is None:
            batch_first = self.batch_first

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            if query.get_layout() != ttnn.TILE_LAYOUT:
                query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            if query_pos.get_layout() != ttnn.TILE_LAYOUT:
                query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            if key.get_layout() != ttnn.TILE_LAYOUT:
                key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            if key_pos.get_layout() != ttnn.TILE_LAYOUT:
                key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        if batch_first:
            # Convert from (batch, seq, embed) to (seq, batch, embed)
            query = ttnn.permute(query, (1, 0, 2))
            key = ttnn.permute(key, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
            # Also convert identity to match
            if identity is not None:
                identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)
                identity = ttnn.permute(identity, (1, 0, 2))

        in_proj_bias = self.attn_in_proj__bias_squeeze

        in_proj_weight = self.attn_in_proj__packed

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # in_proj_weight should be in format [3*embed_dims, embed_dims] (from ttnn.from_torch)
        # Slice along dim 0 to get Q, K, V weights, each of shape [embed_dims, embed_dims]
        # Then transpose to get [out_features, in_features] format for ttnn.linear
        q_weight = in_proj_weight[: self.embed_dims, :]
        k_weight = in_proj_weight[self.embed_dims : 2 * self.embed_dims, :]
        v_weight = in_proj_weight[2 * self.embed_dims :, :]

        # Transpose weights to get [out_features, in_features] format for ttnn.linear
        # ttnn.linear expects weight in format [out_features, in_features]
        q_weight = ttnn.permute(q_weight, (1, 0))
        k_weight = ttnn.permute(k_weight, (1, 0))
        v_weight = ttnn.permute(v_weight, (1, 0))

        q_bias = in_proj_bias[: self.embed_dims]
        k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims]
        v_bias = in_proj_bias[2 * self.embed_dims :]

        q_batch_size, q_sequence_size, q_hidden_size = query.shape
        q_head_size = q_hidden_size // self.num_heads

        k_batch_size, k_sequence_size, k_hidden_size = key.shape
        k_head_size = k_hidden_size // self.num_heads

        v_batch_size, v_sequence_size, v_hidden_size = value.shape
        v_head_size = v_hidden_size // self.num_heads

        query = ttnn.linear(query, q_weight, bias=q_bias)

        key = ttnn.linear(key, k_weight, bias=k_bias)

        if value.get_layout() != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, v_weight, bias=v_bias)

        query = ttnn.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size))
        query = ttnn.permute(query, (1, 0, 2))

        key = ttnn.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size))
        key = ttnn.permute(key, (1, 0, 2))

        value = ttnn.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size))
        value = ttnn.permute(value, (1, 0, 2))

        src_len = key.shape[1]

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))
        key_transposed = ttnn.permute(key, (0, 2, 1))

        if attn_mask is not None:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)
            attn_output_weights = attn_output_weights + attn_mask
        else:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)

        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not supported yet in TtMultiheadAttention")

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)

        attn_output = ttnn.matmul(attn_output_weights, value)

        attn_output = ttnn.permute(attn_output, (1, 0, 2))
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = ttnn.to_layout(attn_output_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_output_weights = ttnn.mean(attn_output_weights, dim=1)

        if identity is not None:
            identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        attn_output = attn_output + identity

        # Convert back to batch_first format if needed
        if batch_first:
            # Convert from (seq, batch, embed) back to (batch, seq, embed)
            attn_output = ttnn.permute(attn_output, (1, 0, 2))

        return attn_output
