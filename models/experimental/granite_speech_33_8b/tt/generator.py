# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import (
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.experimental.granite_speech_33_8b.tt.granite_transformer import GraniteSpeechTransformer


def create_submeshes(mesh_device, data_parallel):
    if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = mesh_device.shape
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    if num_rows == 8 and num_cols == 4 and num_cols % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows, num_cols // data_parallel))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    model = GraniteSpeechTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
    num_layers,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            num_layers=num_layers,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[
        0
    ].tokenizer  # TODO Should we support Data Parallel different models? If so, we need to support multiple tokenizers
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor


class Generator(Generator):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        super().__init__(model, model_args, mesh_device, processor, tokenizer)

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        page_table=None,
        kv_cache=None,
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        start_pos: list[int] = None,  # Cached prefixes lengths
        **kwargs,
    ):
        self.mode = "prefill"
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        batch_size, batch_seq_len, emb_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        out_list = []
        for idx, user_id in enumerate(empty_slots):
            # if model_id is not None, it means that prefill is called from warmup_prefill
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])  # Full length of the current prompt
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx = seq_len - 1  # Last token index of the current full prompt, including the cached tokens
            prefill_seq_len = get_padded_prefill_len(
                seq_len - num_cached_tokens
            )  # Without the cached tokens, then padded
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            # Extracting data for the current user
            # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len, :],  # Select this user, skip the cached tokens
                    torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens), emb_len).long(),  # Pad
                ],
                dim=-2,
            )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(
                prefill_seq_len, num_cached_tokens
            )

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            page_table_user = (
                self._get_prefill_user_page_table(
                    page_table=page_table[idx : idx + 1],  # Slice page table for the current user
                    kv_cache=kv_cache[model_id],
                    prefill_len=seq_len,  # Full length of the current prompt
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                )
                if page_table is not None
                else None
            )
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Check if 'pixel_values' exists and index it safely
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=num_cached_tokens,
                    **local_kwargs,
                )
            if enable_trace_current_prompt:
                # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
                # We need to do this here, because we can't do this part in forward() if we have trace enabled
                # The reason we can't do it in trace is because we can't pass the correct get_last_token to trace
                logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)

            # We have to dispatch copy to host to avoid corruption by the next user's prefill
            out_list.append(logits.cpu(blocking=False))

        # Process the logits after all the prefill are done in data parallel mode
        for idx, out in enumerate(out_list):
            seq_len = int(prompt_lens[idx])
            last_token_idx = seq_len - 1
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx_relative = last_token_idx - num_cached_tokens
            user_id = empty_slots[idx]
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup

            # Ensure all copying is done
            ttnn.synchronize_device(self.model[model_id].mesh_device)

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[idx] = self.model[model_id].process_output_prefill(
                out, last_token_idx=((last_token_idx_relative) % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_single_user_text(
        self,
        tokens,  # New tokens to prefill (without the cached tokens), padded by get_padded_prefill_len()
        page_table,  # Cached and new pages
        user_id,
        last_token_idx,  # Last token index of the full prompt, including the cached tokens
        kv_cache=None,
        model_id=-1,
        num_cached_tokens: int = 0,
        **kwargs,
    ):
        seq_len = tokens.shape[-2]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        use_prefix_caching = num_cached_tokens > 0
        if use_chunked_prefill or use_prefix_caching:
            """
            Chunked prefill requires paged attention. There are some strange constraints which we must meet:
             - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
             checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
             - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
             - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
             to keep it otherwise unaware that it is operating on a chunk.
             - due to the above point, we must always set user_id to 0 for chunked prefill.
            """
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert last_token_idx is not None and last_token_idx < seq_len + num_cached_tokens, (
                f"last_token_idx must be provided and less than seq_len + num_cached_tokens: "
                f"last_token_idx={last_token_idx}, seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
            )

            if use_chunked_prefill:
                # If chunked prefill (more than one chunk is needed), we want to use the maximum chunk size.
                chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            else:
                # Otherwise we only have one chunk.
                chunk_size = seq_len

            last_token_idx_in_seq = last_token_idx - num_cached_tokens  # Excluding the cached tokens
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx_in_seq % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                # These are absolute, i.e. including the cached tokens
                chunk_end = chunk_start + chunk_size
                # These are relative, i.e. excluding the cached tokens
                chunk_start_relative = chunk_start - num_cached_tokens
                chunk_end_relative = chunk_end - num_cached_tokens
                assert chunk_end <= num_cached_tokens + seq_len, (
                    f"chunk_end should be less or equal to "
                    f"num_cached_tokens + seq_len. "
                    f"Got: chunk_end={chunk_end}, "
                    f"num_cached_tokens={num_cached_tokens}, seq_len={seq_len}"
                )

                # Select tokens for the current chunk.
                # Cached tokens were allready excluded (not part of the input),
                # so using relative indexes.
                chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]

                # Select pages for the current chunk.
                # Cached pages must be skipped as well,
                # so using absolute indexes.
                chunk_page_table = page_table_user_padded[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_global_prefill,
                    chunk_rot_mats_local_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    last_token_idx=last_token_idx,
                    **kwargs,
                )
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global_prefill,
                    rot_mats_local=chunk_rot_mats_local_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start_relative == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            (
                prefill_input,
                rot_mats_global_prefill,
                rot_mats_local_prefill,
                page_table_tt,
                _,
            ) = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
                last_token_idx=last_token_idx,
                **kwargs,
            )

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global_prefill,
                rot_mats_local=rot_mats_local_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )
            return tt_logits
