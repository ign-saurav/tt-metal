# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT Qwen2 For Causal LM

A custom Qwen2ForCausalLM that replaces the transformer layers with
Tenstorrent NPU-accelerated TT Transformer layers.

This file is designed to be a drop-in replacement for HuggingFace's Qwen2ForCausalLM
where the heavy transformer computation happens on TT hardware while maintaining
full compatibility with HuggingFace's generate() method.

Usage:
    # Create TT model first
    tt_model, tt_model_args, tt_kv_cache = create_tt_model(...)

    # Create custom Qwen
    model = TTQwen2ForCausalLM.from_tt_model(tt_model, tt_model_args, ...)

    # Use like normal HuggingFace model
    outputs = model.generate(inputs_embeds=audio_embeddings, ...)
"""

import torch
import torch.nn as nn
import ttnn
from typing import Optional, List, Union, Dict, Any
from loguru import logger

from transformers import AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


class TTQwen2Model(nn.Module):
    """
    TT-backed Qwen2 transformer model.

    This replaces HuggingFace's Qwen2Model with a version that uses
    TT Transformer for the heavy computation.
    """

    def __init__(
        self,
        config,
        tt_model,
        tt_model_args,
        mesh_device,
        tt_kv_cache=None,
    ):
        super().__init__()
        self.config = config
        self.tt_model = tt_model
        self.tt_model_args = tt_model_args
        self.mesh_device = mesh_device
        self.tt_kv_cache = tt_kv_cache

        # Track current position for KV cache
        self.current_position = 0

        # Config attributes needed by HuggingFace
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        logger.info(f"TTQwen2Model initialized with TT backend")

    def get_input_embeddings(self):
        """Return the TT model's embedding layer"""
        if hasattr(self.tt_model, "embd"):
            return self.tt_model.embd
        return None

    def _prepare_tt_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        seq_len: int,
    ) -> ttnn.Tensor:
        """
        Convert PyTorch embeddings to TTNN format for TT Transformer.

        TT model expects: [1, 1, seq_len, hidden_dim] in TILE_LAYOUT
        """
        # Ensure bfloat16
        if inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # Reshape to [1, 1, seq, hidden] for TTNN
        if inputs_embeds.dim() == 3:
            # [batch, seq, hidden] -> [1, 1, seq, hidden]
            inputs_embeds = inputs_embeds.unsqueeze(0)
            if inputs_embeds.shape[1] > 1:
                # Handle batch > 1 case
                inputs_embeds = inputs_embeds.squeeze(1).unsqueeze(0).unsqueeze(0)
        elif inputs_embeds.dim() == 2:
            # [seq, hidden] -> [1, 1, seq, hidden]
            inputs_embeds = inputs_embeds.unsqueeze(0).unsqueeze(0)

        # Final shape check
        if inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != 1:
            inputs_embeds = inputs_embeds.view(1, 1, seq_len, -1)

        # Convert to TTNN
        tt_embeds = ttnn.from_torch(
            inputs_embeds,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return tt_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass using TT Transformer backend.

        Supports both token mode (input_ids) and embedding mode (inputs_embeds).
        """
        # Determine if this is prefill or decode based on cache
        is_prefill = past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0
        )

        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")

            # Token mode - use TT model's embedding
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            batch_size, seq_len = input_ids.shape
            original_seq_len = seq_len

            # Pad to 128 for prefill
            if is_prefill and seq_len % 128 != 0:
                pad_len = 128 - (seq_len % 128)
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
                seq_len = input_ids.shape[1]

            # Use TT model's prepare_inputs_prefill
            host_inputs = self.tt_model.prepare_inputs_prefill(
                input_ids, start_pos=self.current_position, page_table=None, chunk_page_table=None, trace_enabled=False
            )
            tt_embeds = host_inputs[0]
            rot_mats_global = host_inputs[1]
            rot_mats_local = host_inputs[2]

        else:
            # Embedding mode (audio/multimodal) - embeddings already provided
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            batch_size, seq_len, hidden_dim = inputs_embeds.shape
            original_seq_len = seq_len

            # Pad to 128 for prefill
            if is_prefill and seq_len % 128 != 0:
                pad_len = 128 - (seq_len % 128)
                inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, pad_len))
                seq_len = inputs_embeds.shape[1]

            # Convert to TTNN format
            tt_embeds = self._prepare_tt_embeddings(inputs_embeds, seq_len)

            # Prepare rotation matrices manually
            end_pos = self.current_position + seq_len
            rot_mats_global = [
                self.tt_model.rope_setup.cos_matrix[:, :, self.current_position : end_pos, :],
                self.tt_model.rope_setup.sin_matrix[:, :, self.current_position : end_pos, :],
            ]
            rot_mats_local = None
            if hasattr(self.tt_model, "rope_local_setup") and self.tt_model.rope_local_setup is not None:
                rot_mats_local = [
                    self.tt_model.rope_local_setup.cos_matrix[:, :, self.current_position : end_pos, :],
                    self.tt_model.rope_local_setup.sin_matrix[:, :, self.current_position : end_pos, :],
                ]

        # Run TT forward
        if is_prefill:
            # Prefill forward
            tt_out = self.tt_model.ttnn_prefill_forward(
                x=tt_embeds,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=None,
                kv_cache=self.tt_kv_cache,
                get_last_token=(original_seq_len - 1) // 32 * 32,
            )

            # Update position
            self.current_position += original_seq_len
        else:
            # Decode forward (single token)
            # Get current position tensor
            current_pos_tt = ttnn.from_torch(
                torch.tensor([self.current_position], dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Get rotation matrices for decode
            rot_idxs = self.tt_model.rope_setup.get_rot_idxs(
                torch.tensor([self.current_position], dtype=torch.int64), on_host=True
            )
            rot_mats_decode = self.tt_model.rope_setup.get_rot_mats(rot_idxs)
            rot_mats_local_decode = None
            if hasattr(self.tt_model, "rope_local_setup") and self.tt_model.rope_local_setup is not None:
                rot_mats_local_decode = self.tt_model.rope_local_setup.get_rot_mats(rot_idxs)

            # Move to decode memory config
            if hasattr(self.tt_model, "model_config") and "DECODE_RESIDUAL_MEMCFG" in self.tt_model.model_config:
                tt_embeds = ttnn.to_memory_config(tt_embeds, self.tt_model.model_config["DECODE_RESIDUAL_MEMCFG"])

            # Run decode forward
            tt_out = self.tt_model.forward(
                tt_embeds,
                current_pos=current_pos_tt,
                rot_mats_global=rot_mats_decode,
                rot_mats_local=rot_mats_local_decode,
                user_id=0,
                mode="decode",
                page_table=None,
                kv_cache=self.tt_kv_cache,
            )

            # Update position
            self.current_position += 1

        # Convert output to torch
        hidden_states = ttnn.to_torch(tt_out)

        # Remove extra dimensions
        while hidden_states.dim() > 3:
            hidden_states = hidden_states.squeeze(0)

        # For prefill, extract last token's hidden state
        if is_prefill:
            offset = (original_seq_len - 1) % 32
            hidden_states = hidden_states[:, offset : offset + 1, :]
        else:
            hidden_states = hidden_states[:, :1, :]

        # Return HuggingFace-compatible output
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.float(),
            past_key_values=past_key_values,  # We manage cache internally
            hidden_states=None,
            attentions=None,
        )

    def reset_cache(self):
        """Reset position tracking for new generation"""
        self.current_position = 0
        logger.info("TTQwen2Model: Reset position to 0")


class TTQwen2ForCausalLM(nn.Module):
    """
    TT-backed Qwen2 for Causal LM.

    Drop-in replacement for HuggingFace's Qwen2ForCausalLM that uses
    TT Transformer for the heavy computation while maintaining full
    compatibility with HuggingFace's generate() method.

    NOTE: Does NOT inherit from Qwen2ForCausalLM to avoid memory overhead.
    Instead, implements the required interfaces for HuggingFace generate().
    """

    # Required class attributes for GenerationMixin
    main_input_name = "inputs_embeds"
    _supports_cache_class = False

    def __init__(self, config, tt_model, tt_model_args, mesh_device, tt_kv_cache=None):
        """
        Initialize TTQwen2ForCausalLM.

        Args:
            config: HuggingFace config
            tt_model: Initialized TT Transformer
            tt_model_args: ModelArgs from tt_transformers
            mesh_device: TTNN mesh device
            tt_kv_cache: KV cache from TT model
        """
        super().__init__()

        self.config = config
        self._tt_model = tt_model
        self._mesh_device = mesh_device

        # Create TT-backed model
        self.model = TTQwen2Model(
            config=config,
            tt_model=tt_model,
            tt_model_args=tt_model_args,
            mesh_device=mesh_device,
            tt_kv_cache=tt_kv_cache,
        )

        # Create LM head on CPU (lightweight)
        # We need to load this from the TT model's output weights
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Copy weights from TT model if available
        if hasattr(tt_model, "lm_head") and hasattr(tt_model.lm_head, "weights"):
            try:
                lm_head_weight = ttnn.to_torch(tt_model.lm_head.weights)
                # Handle shape - might be split or transposed
                if lm_head_weight.dim() > 2:
                    lm_head_weight = lm_head_weight.squeeze()
                # Transpose if needed (TT might have [hidden, vocab])
                if lm_head_weight.shape[0] == config.hidden_size:
                    lm_head_weight = lm_head_weight.t()
                self.lm_head.weight.data = lm_head_weight[: config.vocab_size, :].to(torch.float32)
                logger.info(f"Loaded LM head weights: {self.lm_head.weight.shape}")
            except Exception as e:
                logger.warning(f"Could not load LM head weights from TT model: {e}")

        # Generation config
        self.generation_config = GenerationConfig(
            max_length=getattr(config, "max_position_embeddings", 2048),
            eos_token_id=getattr(config, "eos_token_id", [151645, 151643]),
            pad_token_id=getattr(config, "pad_token_id", 0),
        )

        # Device for HF compatibility
        self.device = torch.device("cpu")

        logger.info("TTQwen2ForCausalLM: Initialized with TT backend")

    @property
    def _device(self):
        """Return the device for GenerationMixin compatibility"""
        return self.device

    def can_generate(self) -> bool:
        """Return True to indicate this model can generate"""
        return True

    def get_output_embeddings(self):
        """Return the LM head"""
        return self.lm_head

    def get_input_embeddings(self):
        """Return the embedding layer"""
        return self.model.get_input_embeddings()

    @classmethod
    def from_tt_model(
        cls,
        tt_model,
        tt_model_args,
        mesh_device,
        tt_kv_cache=None,
        model_path: str = "openbmb/MiniCPM-o-2_6",
    ) -> "TTQwen2ForCausalLM":
        """
        Factory method to create TTQwen2ForCausalLM from a TT model.

        Args:
            tt_model: Initialized TT Transformer model
            tt_model_args: ModelArgs from tt_transformers
            mesh_device: TTNN mesh device
            tt_kv_cache: KV cache from TT model (optional)
            model_path: HuggingFace model path for config

        Returns:
            TTQwen2ForCausalLM instance with TT backend
        """
        # Load config from HuggingFace
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Create instance with TT model
        instance = cls(
            config=config,
            tt_model=tt_model,
            tt_model_args=tt_model_args,
            mesh_device=mesh_device,
            tt_kv_cache=tt_kv_cache,
        )

        return instance

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass using TT backend.

        This method is called by HuggingFace's generate() method.
        """
        # Run through TT-backed model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Check if TT model already outputs logits (vocab-sized output)
        # TT model outputs logits directly if last dim > hidden_size
        if hidden_states.shape[-1] > self.config.hidden_size:
            # Already logits - TT model applied LM head internally
            logits = hidden_states
            # Trim to actual vocab size if padded
            if logits.shape[-1] > self.vocab_size:
                logits = logits[..., : self.vocab_size]
        else:
            # Hidden states - apply LM head
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Simple cross-entropy loss
            from torch.nn import CrossEntropyLoss

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation loop.

        CRITICAL: Preserves inputs_embeds for audio mode.
        """
        model_inputs = {}

        # For first forward (prefill), use inputs_embeds if provided
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        else:
            # Decode mode - use last generated token
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
            model_inputs["input_ids"] = input_ids
            model_inputs["inputs_embeds"] = None

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "position_ids": kwargs.get("position_ids"),
                "cache_position": cache_position,
            }
        )

        return model_inputs

    def reset_cache(self):
        """Reset KV cache for new generation"""
        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        max_length: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate tokens using TT Transformer backend.

        Simple greedy generation loop. Does not use HuggingFace's GenerationMixin
        to avoid complexity and memory overhead.

        Args:
            input_ids: Input token IDs (optional, use inputs_embeds for audio)
            inputs_embeds: Input embeddings (for audio/multimodal mode)
            attention_mask: Attention mask (optional)
            max_new_tokens: Maximum new tokens to generate
            max_length: Maximum total length (alternative to max_new_tokens)
            eos_token_id: End of sequence token ID(s)
            pad_token_id: Padding token ID
            do_sample: Whether to sample (only greedy supported currently)
            temperature: Sampling temperature (only used if do_sample=True)

        Returns:
            Generated token IDs tensor
        """
        # Reset cache for new generation
        self.reset_cache()

        # Handle EOS token IDs
        if eos_token_id is None:
            eos_token_id = [151645, 151643]  # Default MiniCPM terminators
        elif isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # Determine input mode
        if inputs_embeds is not None:
            # Audio/multimodal mode
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch_size, seq_len, _ = inputs_embeds.shape
            # We need to track generated tokens - start with empty
            generated_ids = torch.zeros((batch_size, 0), dtype=torch.long)
        elif input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch_size, seq_len = input_ids.shape
            generated_ids = input_ids.clone()
            inputs_embeds = None  # Will use input_ids for embedding
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Calculate max tokens
        if max_length is not None:
            max_new_tokens = max_length - seq_len

        # First forward (prefill)
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        # Get first token
        if do_sample and temperature > 0:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Check for EOS
        if next_token.item() in eos_token_id:
            return generated_ids

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            # Decode forward (single token)
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=next_token,
                    attention_mask=None,  # Not needed for decode
                )
                logits = outputs.logits

            # Get next token
            if do_sample and temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() in eos_token_id:
                break

        return generated_ids

    def _decode(self, inputs_embeds, tokenizer, attention_mask=None, **kwargs):
        """
        Decode method compatible with MiniCPMO's _decode interface.

        This allows TTQwen2ForCausalLM to be used as a drop-in replacement
        for self.llm in MiniCPMO.
        """
        # Get terminators
        terminators = kwargs.pop("eos_token_id", None)
        if terminators is None:
            terminators = [151645, 151643]  # Default MiniCPM terminators

        # Remove args that conflict with generate
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict_in_generate", None)

        # Call our generate
        sequences = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=terminators,
            **kwargs,
        )

        # Return in HuggingFace-like format
        from transformers.generation.utils import GenerateDecoderOnlyOutput

        return GenerateDecoderOnlyOutput(sequences=sequences)
