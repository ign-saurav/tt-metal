# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT Qwen2 For Causal LM - Following Qwen2.5-VL Pattern

Direct generation loop (not using HuggingFace GenerationMixin)
for proper KV cache handling during decode.
"""

import torch
import torch.nn as nn
import ttnn
from typing import Optional, List, Union
from loguru import logger

from transformers import AutoConfig


class TTQwen2ForCausalLM(nn.Module):
    """
    TT-backed Qwen2 for Causal LM following Qwen2.5-VL pattern.

    Key differences from GenerationMixin approach:
    1. Direct generate() loop - no HF generate() complexity
    2. Prefill: embeddings -> ttnn_prefill_forward -> first token
    3. Decode: tokens -> ttnn_decode_forward -> next token
    4. Explicit position tracking with current_pos
    """

    def __init__(
        self,
        config,
        tt_model,  # The TT Transformer model
        tt_model_args,  # Model args
        mesh_device,
        tt_kv_cache=None,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.tt_model = tt_model
        self.tt_model_args = tt_model_args
        self.mesh_device = mesh_device
        self.tt_kv_cache = tt_kv_cache
        self.tokenizer = tokenizer

        # Track generation state
        self.current_position = 0

        logger.info("TTQwen2ForCausalLM initialized (Qwen2.5-VL pattern)")

    @classmethod
    def from_tt_model(
        cls,
        tt_model,
        tt_model_args,
        mesh_device,
        config=None,
        model_path: str = "openbmb/MiniCPM-o-2_6",
        tt_kv_cache=None,
        tokenizer=None,
    ):
        """Create from existing TT model"""
        if config is None:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config = config.llm_config if hasattr(config, "llm_config") else config

        return cls(
            config=config,
            tt_model=tt_model,
            tt_model_args=tt_model_args,
            mesh_device=mesh_device,
            tt_kv_cache=tt_kv_cache,
            tokenizer=tokenizer,
        )

    def reset_cache(self):
        """Reset for new generation"""
        self.current_position = 0

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens following Qwen2.5-VL pattern.

        Direct generation loop without HuggingFace GenerationMixin.

        Args:
            inputs_embeds: Embeddings [batch, seq_len, hidden_dim]
            max_new_tokens: Maximum tokens to generate
            eos_token_id: EOS token(s) to stop generation

        Returns:
            Generated token IDs [batch, seq_len]
        """
        # Reset state
        self.reset_cache()

        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        batch_size, seq_len, hidden_dim = inputs_embeds.shape

        # Normalize eos_token_id to list
        if eos_token_id is None:
            eos_token_ids = []
        elif isinstance(eos_token_id, int):
            eos_token_ids = [eos_token_id]
        else:
            eos_token_ids = list(eos_token_id)

        logger.info(f"Generate: batch={batch_size}, seq_len={seq_len}, max_new_tokens={max_new_tokens}")
        logger.info(f"EOS tokens: {eos_token_ids}")

        # ====== PREFILL PHASE ======
        logger.info("Starting prefill...")

        # Pad to 128 boundary
        padded_len = ((seq_len + 127) // 128) * 128
        if seq_len != padded_len:
            inputs_embeds_padded = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padded_len - seq_len), value=0)
        else:
            inputs_embeds_padded = inputs_embeds

        # Convert embeddings to TT tensor - [batch, 1, seq, hidden]
        tt_embeds = ttnn.from_torch(
            inputs_embeds_padded.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Slice rotation matrices for prefill
        tt_rot_mats_prefill = [
            self.tt_model.rope_setup.cos_matrix[:, :, :padded_len, :],
            self.tt_model.rope_setup.sin_matrix[:, :, :padded_len, :],
        ]

        tt_rot_mats_local = None
        if hasattr(self.tt_model, "rope_local_setup") and self.tt_model.rope_local_setup:
            tt_rot_mats_local = [
                self.tt_model.rope_local_setup.cos_matrix[:, :, :padded_len, :],
                self.tt_model.rope_local_setup.sin_matrix[:, :, :padded_len, :],
            ]

        # Run prefill
        tt_out = self.tt_model.ttnn_prefill_forward(
            tt_embeds,
            rot_mats_global=tt_rot_mats_prefill,
            rot_mats_local=tt_rot_mats_local,
            user_id=0,
            page_table=None,
            kv_cache=self.tt_kv_cache,
            get_last_token=(seq_len - 1) // 32 * 32,
        )

        # Get first token from prefill logits
        logits = ttnn.to_torch(tt_out).float()
        ttnn.deallocate(tt_embeds)
        ttnn.deallocate(tt_out)

        # Extract last token's logits
        while logits.dim() > 2:
            logits = logits.squeeze(0) if logits.shape[0] == 1 else logits.squeeze(1)
        if logits.dim() == 2:
            logits = logits[-1:]  # Last token only

        # Get first generated token
        first_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # [1, 1]
        logger.info(f"First token: {first_token.item()}")

        # Track position
        self.current_position = seq_len

        # Initialize output with first token
        generated_tokens = [first_token]
        current_token = first_token

        # Check if first token is EOS
        if current_token.item() in eos_token_ids:
            logger.info("EOS reached at first token")
            return torch.cat(generated_tokens, dim=1)

        # ====== DECODE LOOP ======
        logger.info("Starting decode loop...")

        for iteration in range(max_new_tokens - 1):
            # Prepare position tensors
            current_pos_torch = torch.tensor([self.current_position], dtype=torch.int32)
            rot_mat_idxs = torch.tensor([self.current_position], dtype=torch.int64)

            logger.debug(f"Decode iteration {iteration}: token={current_token.item()}, pos={self.current_position}")

            # Convert token to ttnn tensor for decode (following prepare_decode_inputs_host pattern)
            token_padded = torch.nn.functional.pad(
                current_token.view(-1), (0, 32 - current_token.numel()), "constant", 0
            )
            tt_token = ttnn.from_torch(
                token_padded,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_token = ttnn.unsqueeze_to_4D(tt_token)

            # Convert current_pos to ttnn tensor for cache update
            tt_current_pos = ttnn.from_torch(
                current_pos_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Use tt_model's decode forward directly
            # The model will use rope_setup.get_rot_mats(rot_mat_idxs) internally
            tt_out = self.tt_model.ttnn_decode_forward(
                tt_token,
                tt_current_pos,
                rot_mat_idxs=rot_mat_idxs,
                kv_cache=self.tt_kv_cache,
            )

            # Clean up intermediate tensors
            ttnn.deallocate(tt_token)
            ttnn.deallocate(tt_current_pos)

            # Get next token
            logits = ttnn.to_torch(tt_out).float()
            ttnn.deallocate(tt_out)

            # Extract logits - decode output is [1, 1, 32, vocab] or similar
            # We only want the first position (the actual token, not padding)
            while logits.dim() > 2:
                logits = logits.squeeze(0) if logits.shape[0] == 1 else logits.squeeze(1)

            # Take only the first position's logits (rest is padding)
            if logits.dim() == 2 and logits.shape[0] > 1:
                logits = logits[0:1, :]  # [1, vocab]

            # Sample next token (greedy for now)
            next_token = torch.argmax(logits, dim=-1).reshape(1, 1)  # [1, 1]

            # Decode token for logging
            if self.tokenizer is not None:
                try:
                    decoded = self.tokenizer.decode([next_token.item()])
                    logger.info(f"Token {iteration+1}: {next_token.item()} ('{decoded}')")
                except:
                    logger.info(f"Token {iteration+1}: {next_token.item()}")
            else:
                logger.info(f"Token {iteration+1}: {next_token.item()}")

            generated_tokens.append(next_token)
            current_token = next_token
            self.current_position += 1

            # Check EOS - but only after generating at least 5 tokens
            # This helps debug cases where EOS is generated too early
            if iteration >= 5 and next_token.item() in eos_token_ids:
                logger.info(f"EOS reached at iteration {iteration}")
                break

        # Concatenate all generated tokens
        output = torch.cat(generated_tokens, dim=1)
        logger.info(f"Generated {output.shape[1]} tokens: {output.tolist()}")

        return output

    def get_hidden_states(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get hidden states by running TT prefill with get_last_token=-1.

        This returns hidden states AFTER all transformer layers but BEFORE norm/lm_head.
        Used for TTS speaker embedding extraction.

        Args:
            input_ids: Token IDs [batch, seq_len] (optional, will be embedded)
            inputs_embeds: Pre-computed embeddings [batch, seq_len, hidden] (optional)

        Returns:
            Hidden states [batch, seq_len, hidden_dim]
        """
        if inputs_embeds is not None:
            # Use provided embeddings directly
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch_size, seq_len, hidden_dim = inputs_embeds.shape
            logger.info(f"Getting hidden states for {seq_len} tokens via TT prefill (from embeddings)...")

            # Pad to 128 boundary
            padded_len = ((seq_len + 127) // 128) * 128
            if seq_len != padded_len:
                inputs_embeds_padded = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padded_len - seq_len), value=0)
            else:
                inputs_embeds_padded = inputs_embeds

            # Convert to TT tensor [batch, 1, seq, hidden]
            tt_embeds = ttnn.from_torch(
                inputs_embeds_padded.unsqueeze(1),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        elif input_ids is not None:
            # Embed token IDs
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            batch_size, seq_len = input_ids.shape
            logger.info(f"Getting hidden states for {seq_len} tokens via TT prefill (from token IDs)...")

            # Get embeddings from model's embedding layer
            tt_embd = self.tt_model.embd

            # Pad to 128 boundary
            padded_len = ((seq_len + 127) // 128) * 128
            if seq_len != padded_len:
                input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - seq_len), value=0)
            else:
                input_ids_padded = input_ids

            # Convert to TT tensor
            tt_input_ids = ttnn.from_torch(
                input_ids_padded,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Get embeddings via TT embd layer
            tt_embeds = tt_embd(tt_input_ids)
            ttnn.deallocate(tt_input_ids)

            # Reshape for transformer: [batch, 1, seq, hidden]
            tt_embeds = ttnn.reshape(tt_embeds, [batch_size, 1, padded_len, -1])
            tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Prepare rotation matrices for prefill
        tt_rot_mats_prefill = [
            self.tt_model.rope_setup.cos_matrix[:, :, :padded_len, :],
            self.tt_model.rope_setup.sin_matrix[:, :, :padded_len, :],
        ]

        tt_rot_mats_local = None
        if hasattr(self.tt_model, "rope_local_setup") and self.tt_model.rope_local_setup:
            tt_rot_mats_local = [
                self.tt_model.rope_local_setup.cos_matrix[:, :, :padded_len, :],
                self.tt_model.rope_local_setup.sin_matrix[:, :, :padded_len, :],
            ]

        # Run prefill with get_last_token=-1 to get hidden states
        tt_hidden = self.tt_model.ttnn_prefill_forward(
            tt_embeds,
            rot_mats_global=tt_rot_mats_prefill,
            rot_mats_local=tt_rot_mats_local,
            user_id=0,
            page_table=None,
            kv_cache=self.tt_kv_cache,
            get_last_token=-1,  # Returns hidden states before norm/lm_head
        )

        # Convert back to torch
        hidden_states = ttnn.to_torch(tt_hidden).float()
        ttnn.deallocate(tt_embeds)
        ttnn.deallocate(tt_hidden)

        # Reshape: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)

        # Trim padding
        if seq_len != padded_len:
            hidden_states = hidden_states[:, :seq_len, :]

        logger.info(f"Got hidden states shape: {hidden_states.shape}")
        return hidden_states
