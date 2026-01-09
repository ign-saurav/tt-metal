# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Drop-in replacement classes for MiniCPM-o components.

These classes wrap TT implementations to provide the same interface as the
HuggingFace reference model components, enabling seamless replacement.

Usage:
    from transformers import AutoModel
    from models.experimental.miniCPMo.tt.drop_in_replacements import (
        DropInChatTTSDecoder,
        DropInAudioEncoder,
    )

    # Load model from HuggingFace
    model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', ...)

    # Replace components with TT implementations
    model.tts = DropInChatTTSDecoder(model.tts, device, model.embed_dim)
    model.apm = DropInAudioEncoder(model.apm, device, model.config.audio_config)
"""

import torch
import torch.nn as nn
import ttnn
from typing import Optional, List, Tuple, Union
from loguru import logger

# Import TT implementations
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from models.experimental.miniCPMo.tt.ttnn_whisper_encoder import TtnnWhisperEncoder


class DropInChatTTSDecoder(nn.Module):
    """
    Drop-in replacement for ConditionalChatTTS.

    Wraps TtnnChatTTSDecoder with the same interface as the HuggingFace model,
    so it can be swapped in without changing any calling code.

    The reference model's TTS decoder methods are preserved and forwarded to
    the TT implementation where accelerated, or to the reference model for
    methods that run on CPU (like embedding lookups, vocoder, etc.)
    """

    def __init__(self, reference_model, device, llm_embed_dim: int):
        """
        Initialize drop-in replacement for ConditionalChatTTS.

        Args:
            reference_model: The original ConditionalChatTTS model from HuggingFace
            device: TT device (ttnn.Device or mesh device)
            llm_embed_dim: LLM hidden dimension (typically 3584 for Qwen2.5)
        """
        super().__init__()
        self._reference = reference_model
        self.tt_device = device
        self.llm_embed_dim = llm_embed_dim

        # Copy essential config and attributes from reference
        self.config = reference_model.config
        self.num_vq = reference_model.num_vq
        self.num_audio_tokens = reference_model.num_audio_tokens
        self.num_spk_embs = reference_model.num_spk_embs
        self.streaming_text_reserved_len = reference_model.streaming_text_reserved_len
        self.audio_bos_token_id = reference_model.audio_bos_token_id
        self.use_speaker_embedding = reference_model.use_speaker_embedding
        self.use_llm_hidden_state = reference_model.use_llm_hidden_state
        self.spk_emb_token_id = reference_model.spk_emb_token_id

        # Keep reference components that don't need TT acceleration
        # These run on CPU/GPU and are fast enough
        self.emb_text = reference_model.emb_text
        self.emb_code = reference_model.emb_code
        self.head_code = reference_model.head_code
        self.projector = reference_model.projector
        self.dvae = reference_model.dvae
        self.model = reference_model.model  # Keep reference LlamaModel for compatibility

        # Initialize TT decoder
        logger.info("Initializing TT ChatTTS decoder...")
        self.tt_decoder = TtnnChatTTSDecoder(
            device=device,
            llm_dim=llm_embed_dim,
            hidden_size=reference_model.config.hidden_size,
            num_attention_heads=reference_model.config.num_attention_heads,
            num_hidden_layers=reference_model.config.num_hidden_layers,
            intermediate_size=reference_model.config.intermediate_size,
            num_text_tokens=reference_model.emb_text.num_embeddings,
            num_audio_tokens=reference_model.num_audio_tokens,
            num_vq=reference_model.num_vq,
            num_spk_embs=reference_model.num_spk_embs,
            max_position_embeddings=reference_model.config.max_position_embeddings,
        )

        # Load weights from reference model
        self.tt_decoder.load_weights(reference_model.state_dict())
        logger.info("✅ TT ChatTTS decoder initialized with weights")

        # Store reference for method forwarding
        self.tt_model = self.tt_decoder  # Alias for compatibility

    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Forward to reference implementation (CPU-side embedding merge)."""
        return self._reference.merge_inputs_embeds(input_ids, lm_spk_emb_last_hidden_states)

    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Prefill text tokens using TT decoder.

        This is a key method for TTS streaming - accelerated with TT.
        """
        # Use TT decoder's prefill if available
        if hasattr(self.tt_decoder, "prefill_text"):
            return self.tt_decoder.prefill_text(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
            )
        # Fallback to reference
        return self._reference.prefill_text(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers=None,
        logits_processors=None,
        show_tqdm=False,
    ):
        """
        Generate audio codes using TT decoder.

        This is the main generation loop - accelerated with TT.
        """
        if logits_warpers is None:
            logits_warpers = []
        if logits_processors is None:
            logits_processors = []

        # Use TT decoder's generate if available
        if hasattr(self.tt_decoder, "generate"):
            return self.tt_decoder.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                temperature=temperature,
                eos_token=eos_token,
                streaming_tts_text_mask=streaming_tts_text_mask,
                force_no_stop=force_no_stop,
                min_new_token=min_new_token,
                max_new_token=max_new_token,
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
                show_tqdm=show_tqdm,
            )
        # Fallback to reference
        return self._reference.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            temperature=temperature,
            eos_token=eos_token,
            streaming_tts_text_mask=streaming_tts_text_mask,
            force_no_stop=force_no_stop,
            min_new_token=min_new_token,
            max_new_token=max_new_token,
            logits_warpers=logits_warpers,
            logits_processors=logits_processors,
            show_tqdm=show_tqdm,
        )

    def forward(self, *args, **kwargs):
        """Forward pass through TT decoder."""
        return self.tt_decoder.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward unknown attributes to reference model for compatibility."""
        # First, let nn.Module handle its own attributes (_modules, _parameters, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then forward unknown attributes to reference model
        _modules = object.__getattribute__(self, "_modules")
        _reference = _modules.get("_reference")
        if _reference is not None:
            try:
                return getattr(_reference, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class DropInAudioEncoder(nn.Module):
    """
    Drop-in replacement for MiniCPMWhisperEncoder.

    Wraps TtnnWhisperEncoder with the same interface as the HuggingFace model.
    """

    def __init__(self, reference_model, device, config):
        """
        Initialize drop-in replacement for Whisper audio encoder.

        Args:
            reference_model: The original MiniCPMWhisperEncoder from HuggingFace
            device: TT device
            config: Audio config (WhisperConfig or dict)
        """
        super().__init__()
        self._reference = reference_model
        self.tt_device = device

        # Store config
        if hasattr(config, "to_dict"):
            self.config_dict = config.to_dict()
            self.config = config
        else:
            self.config_dict = config
            self.config = config

        # Initialize TT encoder
        logger.info("Initializing TT Whisper encoder...")
        self.tt_encoder = TtnnWhisperEncoder(
            mesh_device=device,
            config=self.config_dict,
        )

        # Load weights from reference
        self.tt_encoder.load_weights(reference_model.state_dict())
        logger.info("✅ TT Whisper encoder initialized with weights")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Forward pass through TT Whisper encoder.

        Args:
            input_features: Mel spectrogram features [batch, mel_bins, time]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to output attention weights (not supported in TT)
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict (ignored, always returns object)
            past_key_values: Optional past key values for caching (not supported in TT, falls back to reference)
            use_cache: Whether to use caching (not supported in TT, falls back to reference)

        Returns:
            Object with last_hidden_state and optionally hidden_states attributes
        """
        # If past_key_values or use_cache is requested, fall back to reference model
        # as TT encoder doesn't support KV caching for streaming yet
        if past_key_values is not None or use_cache:
            logger.debug("Using reference model for audio encoder (KV caching requested)")
            return self._reference(
                input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # TT encoder forward
        output = self.tt_encoder.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        # Convert ttnn tensor to torch if needed
        if isinstance(output, ttnn.Tensor):
            output = ttnn.to_torch(output)

        # Wrap output in an object that matches HuggingFace interface
        class AudioEncoderOutput:
            def __init__(self, last_hidden_state, hidden_states=None, past_key_values=None):
                self.last_hidden_state = last_hidden_state
                # hidden_states is a tuple of all layer outputs; for simplicity,
                # we return the last hidden state for all requested layers
                self.hidden_states = hidden_states
                self.past_key_values = past_key_values

        # If output_hidden_states is requested, create a tuple of hidden states
        # The reference model accesses hidden_states[layer_idx], so we need enough entries
        hidden_states = None
        if output_hidden_states:
            # Create a tuple with the output repeated for each layer + 1 (for embeddings)
            num_layers = self.config_dict.get("encoder_layers", 32) + 1
            hidden_states = tuple([output] * num_layers)

        return AudioEncoderOutput(
            last_hidden_state=output,
            hidden_states=hidden_states,
            past_key_values=None,
        )

    def __getattr__(self, name: str):
        """Forward unknown attributes to reference model for compatibility."""
        # First, let nn.Module handle its own attributes (_modules, _parameters, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then forward unknown attributes to reference model
        _modules = object.__getattribute__(self, "_modules")
        _reference = _modules.get("_reference")
        if _reference is not None:
            try:
                return getattr(_reference, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class DropInVisionEncoder(nn.Module):
    """
    Drop-in replacement for SiglipVisionTransformer.

    Uses TT-accelerated vision encoder with dynamic NaViT position embeddings.
    """

    def __init__(self, reference_model, device, config):
        """
        Initialize drop-in replacement for SigLip vision encoder.

        Args:
            reference_model: The original SiglipVisionTransformer from HuggingFace
            device: TT device
            config: Vision config
        """
        super().__init__()
        self._reference = reference_model
        self.tt_device = device
        self.config = config

        # Copy essential attributes
        self.embed_dim = reference_model.embed_dim if hasattr(reference_model, "embed_dim") else config.hidden_size
        self.patch_size = reference_model.patch_size if hasattr(reference_model, "patch_size") else config.patch_size
        self.num_patches_per_side = (
            reference_model.embeddings.num_patches_per_side
            if hasattr(reference_model.embeddings, "num_patches_per_side")
            else config.image_size // config.patch_size
        )

        logger.info("Initializing TT SigLip vision encoder with NaViT position embeddings...")

        # Import TT vision encoder and preprocessing utilities
        from models.experimental.miniCPMo.tt.ttnn_siglip_vision import TtSiglipVisionTransformer
        from models.experimental.miniCPMo.tests.test_siglip_vision_emb import (
            create_siglip_vision_embedding_preprocessor,
        )
        from ttnn.model_preprocessing import preprocess_model_parameters

        # Preprocess embeddings model parameters
        embeddings_model = reference_model.embeddings
        emb_parameters = preprocess_model_parameters(
            initialize_model=lambda: embeddings_model,
            custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, ttnn.bfloat16),
            device=device,
        )

        # Store position embedding weight for dynamic position embedding computation
        self.position_embedding_weight = emb_parameters["position_embedding"]["weight"]

        # Initialize TT vision encoder with preprocessed parameters
        self.tt_encoder = TtSiglipVisionTransformer(
            mesh_device=device,
            config=config,
            parameters=emb_parameters,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            patch_size=config.patch_size,
            image_size=config.image_size,
            num_channels=config.num_channels,
        )

        # Load remaining weights from reference
        self.tt_encoder.load_weights(reference_model.state_dict())
        logger.info("✅ TT SigLip vision encoder initialized with weights")

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        """
        Forward pass through TT vision encoder with dynamic position embeddings.
        """
        # Import the position embedding generator from tt_modeling_minicpmo
        from models.experimental.miniCPMo.tt.tt_modeling_minicpmo import generate_position_embeddings
        from models.common.utility_functions import tt2torch_tensor

        # Generate dynamic position embeddings (NaViT style)
        position_embeddings = generate_position_embeddings(
            pixel_values,
            self.patch_size,
            self.num_patches_per_side,
            self.position_embedding_weight,
            patch_attention_mask,
            tgt_sizes,
            self.tt_device,
        )

        # Run TT encoder forward
        hidden_states = self.tt_encoder.forward(pixel_values, position_embeddings)

        # Convert ttnn tensor back to torch
        hidden_states = tt2torch_tensor(hidden_states)

        # Wrap in output object to match HuggingFace interface (.last_hidden_state)
        class VisionOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return VisionOutput(hidden_states)

    def __getattr__(self, name: str):
        """Forward unknown attributes to reference model for compatibility."""
        # First, let nn.Module handle its own attributes (_modules, _parameters, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then forward unknown attributes to reference model
        _modules = object.__getattribute__(self, "_modules")
        _reference = _modules.get("_reference")
        if _reference is not None:
            try:
                return getattr(_reference, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class DropInQwen2LLM(nn.Module):
    """
    Drop-in replacement for Qwen2ForCausalLM.

    Wraps TTQwen2ForCausalLM with the same interface as the HuggingFace model,
    so it can be swapped in without changing any calling code.

    Uses the working pattern from test_tt_qwen2_audio.py.
    """

    def __init__(
        self,
        reference_model,
        device,
        model_path: str = "openbmb/MiniCPM-o-2_6",
        max_seq_len: int = 1024,
    ):
        """
        Initialize drop-in replacement for Qwen2ForCausalLM.

        Args:
            reference_model: The original Qwen2ForCausalLM from HuggingFace
            device: TT device (ttnn.Device or mesh device)
            model_path: HuggingFace model path for loading weights
            max_seq_len: Maximum sequence length for KV cache
        """
        import os
        from transformers import AutoTokenizer
        from models.experimental.miniCPMo.tt.tt_qwen2_for_causal_lm import TTQwen2ForCausalLM
        from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
        from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

        super().__init__()
        self._reference = reference_model
        self.tt_device = device
        self.model_path = model_path

        # Copy config from reference
        self.config = reference_model.config

        # Set environment variable for model loading
        if not os.environ.get("HF_MODEL"):
            os.environ["HF_MODEL"] = model_path

        logger.info("Initializing TT Qwen2 LLM...")

        # Load weights from MiniCPM checkpoint
        bridge = MiniCPMWeightBridge(model_path)
        qwen_weights = bridge.get_qwen_weights()
        logger.info(f"Loaded {len(qwen_weights)} weight tensors from MiniCPM checkpoint")

        # Create TT model
        tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
            mesh_device=device,
            instruct=False,
            max_batch_size=1,
            optimizations=None,
            max_seq_len=max_seq_len,
            paged_attention_config=None,
            dtype=ttnn.bfloat8_b,
            state_dict=qwen_weights,
            dummy_weights=False,
        )
        logger.info(f"TT Model: {tt_model_args.n_layers} layers, dim={tt_model_args.dim}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Create TTQwen2ForCausalLM wrapper
        self.tt_llm = TTQwen2ForCausalLM.from_tt_model(
            tt_model=tt_model,
            tt_model_args=tt_model_args,
            mesh_device=device,
            tt_kv_cache=tt_kv_cache,
            model_path=model_path,
            tokenizer=tokenizer,
        )
        self.tt_llm.eval()

        # Store tokenizer for decoding
        self.tokenizer = tokenizer

        # Keep reference embedding layer for get_input_embeddings()
        self._input_embeddings = reference_model.get_input_embeddings()

        logger.info("✅ TT Qwen2 LLM initialized")

    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        output_hidden_states: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs,
    ):
        """
        Generate tokens using TT hardware.

        This method matches the HuggingFace generate() interface but runs on TT.
        """
        # If input_ids provided but not embeddings, get embeddings from reference
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._input_embeddings(input_ids)

        if inputs_embeds is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided")

        # Reset cache before generation
        self.tt_llm.reset_cache()

        # Run TT generation
        output_ids = self.tt_llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
        )

        # Handle return format
        if return_dict_in_generate:
            from dataclasses import dataclass

            @dataclass
            class GenerateOutput:
                sequences: torch.Tensor
                hidden_states: Optional[Tuple] = None
                attentions: Optional[Tuple] = None
                past_key_values: Optional[Tuple] = None  # TT uses internal KV cache

            hidden_states = None
            tt_hidden_states_raw = None
            # If hidden states needed (for TTS), get them via TT prefill on FULL sequence
            if output_hidden_states:
                logger.info("Getting hidden states via TT prefill (for TTS)...")

                # Build FULL sequence: input embeddings + generated token embeddings
                # inputs_embeds is [batch, input_seq_len, hidden]
                # output_ids is [batch, num_generated] - need to embed these
                generated_embeds = self._input_embeddings(output_ids)  # [batch, num_generated, hidden]

                # Concatenate: [batch, input_seq_len + num_generated, hidden]
                full_embeds = torch.cat([inputs_embeds, generated_embeds], dim=1)
                logger.info(
                    f"Full sequence: input={inputs_embeds.shape[1]} + generated={generated_embeds.shape[1]} = {full_embeds.shape[1]} tokens"
                )

                # Run TT prefill on full sequence to get hidden states
                tt_hidden_states_raw = self.tt_llm.get_hidden_states(inputs_embeds=full_embeds)
                # tt_hidden_states_raw is [batch, full_seq_len, hidden]

                # HuggingFace generation format:
                # hidden_states is tuple of tuples - one outer tuple per token
                # Each inner tuple has layer hidden states (we provide last layer only)
                # Format: ((token_0_last_layer_hs,), (token_1_last_layer_hs,), ...)
                # Where each token_i_last_layer_hs is [batch, 1, hidden]
                seq_len = tt_hidden_states_raw.shape[1]
                hidden_states = tuple(
                    (tt_hidden_states_raw[:, i : i + 1, :],)  # [batch, 1, hidden] wrapped in tuple
                    for i in range(seq_len)
                )
                logger.info(f"Got hidden states for {seq_len} tokens, shape per token: {hidden_states[0][0].shape}")

            return GenerateOutput(
                sequences=output_ids,
                hidden_states=hidden_states,
                attentions=None,
                past_key_values=None,  # TT KV cache is managed internally
            )

        return output_ids

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass - falls back to reference for now.

        TT model is optimized for generate(), not forward().
        """
        logger.debug("DropInQwen2LLM.forward() - using reference model")
        return self._reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def get_input_embeddings(self):
        """Return the input embeddings layer."""
        return self._input_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Forward to reference model's prepare_inputs_for_generation."""
        return self._reference.prepare_inputs_for_generation(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward unknown attributes to reference model for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        _modules = object.__getattribute__(self, "_modules")
        _reference = _modules.get("_reference")
        if _reference is not None:
            try:
                return getattr(_reference, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
