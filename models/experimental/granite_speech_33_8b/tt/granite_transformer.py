# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.model import Transformer


class GraniteSpeechTransformer(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
    ):
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=attention_class,
            rope_setup_class=rope_setup_class,
        )

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        # tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.to_layout(tokens, ttnn.TILE_LAYOUT)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, tt_page_table, tt_chunk_page_table

    def prepare_inputs_prefill(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, last_token_idx=None
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device if trace is disabled or on host if trace is enabled.
        TODO: Debate whether this function is responsible for padding
        """

        # We set the device to None if trace is enabled so we keep the tensors on host instead of sending it to the device (None - keeps on host, device - sends to specified device)
        # We will send them to device later (copy_host_to_device)
        device = None if trace_enabled else self.mesh_device

        # assert tokens.dim() == 2, "tokens must be a 2D tensor"
        # tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-2]
        tokens = ttnn.from_torch(
            tokens,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 2),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        # self.embd expects that tokens are on device ; if trace is enabled, the tensors will be later on device, so we will do these 2 steps when we copy the tokens to the device
        if not trace_enabled:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens)

        # Slice the rot mats to the prefill seqlen
        # Use cos_matrix_prefill/sin_matrix_prefill which are TILE_LAYOUT (required by rotary_embedding_llama)
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        # Use last_token_idx if provided, otherwise fall back to S (padded sequence length)
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Seqence length {seq_len} exceeds max seq len {mat_len}"

        # The padding is needed just to make SDPA happy, we will be selecting the token that is within the range of the rot mat.
        required_end = start_pos + S
        if required_end > mat_len:
            pad_len = required_end - mat_len
        else:
            pad_len = 0

        # We set slice_end to max_seq_len so that we don't create a new tensor for the whole cos_matrix and sin_matrix ; in case of trace, we will use the whole matrix for all seq_lens supported by trace
        slice_start = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, required_end)
        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, slice_start:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, slice_start:slice_end, :]
        if pad_len > 0:
            # padding: [(before, after), ...] for each dim; pad at end of 3rd dim (dim=2) by pad_len
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)
        tt_rot_mats_prefill_global = [
            cos_slice,
            sin_slice,
        ]

        if hasattr(self, "rope_local_setup"):
            # Use cos_matrix_prefill/sin_matrix_prefill which are TILE_LAYOUT (required by rotary_embedding_llama)
            local_mat_len = self.rope_local_setup.cos_matrix_prefill.shape[2]
            local_required_end = start_pos + S
            if local_required_end > local_mat_len:
                local_pad_len = local_required_end - local_mat_len
            else:
                local_pad_len = 0

            local_slice_end = self.args.max_seq_len if trace_enabled else min(local_mat_len, local_required_end)
            local_cos_slice = self.rope_local_setup.cos_matrix_prefill[:, :, slice_start:local_slice_end, :]
            local_sin_slice = self.rope_local_setup.sin_matrix_prefill[:, :, slice_start:local_slice_end, :]
            if local_pad_len > 0:
                # pad at end of 3rd dim (dim=2) by local_pad_len
                local_padding = [(0, 0)] * 4
                local_padding[2] = (0, local_pad_len)
                local_cos_slice = ttnn.pad(local_cos_slice, padding=local_padding, value=0.0)
                local_sin_slice = ttnn.pad(local_sin_slice, padding=local_padding, value=0.0)

            tt_rot_mats_prefill_local = [
                local_cos_slice,
                local_sin_slice,
            ]
        else:
            tt_rot_mats_prefill_local = None

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return (
            tokens if trace_enabled else tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
        )
