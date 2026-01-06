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

import os
import torch
import torch.nn as nn
import ttnn
import copy
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Any, Dict
from loguru import logger

# Import TT implementations
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from models.experimental.miniCPMo.tt.ttnn_whisper_encoder import TtnnWhisperEncoder
from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
from models.experimental.miniCPMo.tt_transformers.common import create_tt_model
from models.experimental.miniCPMo.tt_transformers.model_config import ModelArgs
from models.experimental.miniCPMo.tt_transformers.generator import Generator, create_submeshes
from models.experimental.miniCPMo.tt_transformers.common import PagedAttentionConfig


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


class DropInQwenModel:
    """
    A Drop-in replacement for the MiniCPM-o LLM (Qwen) that handles Text + Vision.

    Implementation Details:
    - Inputs: Reference Model Processor
    - Embeddings: Reference Model VLLM Embedding
    - Execution: TTNN Prefill/Decode with Padding
    - Sampling: CPU-based Top-K/Top-P (Fixed implementation)
    """

    def __init__(
        self,
        device: ttnn.Device,
        config: Any,
        weight_bridge: Optional[MiniCPMWeightBridge] = None,
        reference_model=None,
    ):
        self.device = device
        self.config = config
        self.reference_model = reference_model

        self.weight_bridge = weight_bridge if weight_bridge else MiniCPMWeightBridge()

        self.model_args = None
        self.model = None
        self.page_table = None
        self.tt_kv_cache = None

        self._init_qwen_llm()

    def _init_qwen_llm(self):
        logger.info("Initializing TT Qwen Model...")

        if "HF_MODEL" not in os.environ:
            os.environ["HF_MODEL"] = "openbmb/MiniCPM-o-2_6"

        qwen_weights = self.weight_bridge.get_qwen_weights()

        self.model_args = ModelArgs(
            mesh_device=self.device,
            instruct=False,
            max_batch_size=1,
            max_seq_len=4096,
            dummy_weights=False,
        )
        self.model_args.max_prefill_chunk_size = 4096

        self.paged_attention_config = PagedAttentionConfig(
            block_size=32,
            max_num_blocks=1024,
        )

        data_parallel = 1
        global_batch_size = 1

        permutation = torch.randperm(self.paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        self.page_table = reverse_permutation.reshape(
            global_batch_size, self.paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )

        submesh_devices = create_submeshes(self.device, data_parallel)
        model_args_i, model_i, tt_kv_cache_i, _ = create_tt_model(
            submesh_devices[0],
            instruct=False,
            max_batch_size=global_batch_size,
            optimizations=None,
            max_seq_len=self.model_args.max_seq_len,
            paged_attention_config=self.paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=qwen_weights,
            dummy_weights=False,
        )

        self.model = [model_i]
        self.tt_kv_cache = [tt_kv_cache_i]
        self.model_args = [model_args_i]

        if self.reference_model and hasattr(self.reference_model, "processor"):
            self.tokenizer = self.reference_model.processor.tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)

        self.generator = Generator(self.model, self.model_args, self.device, tokenizer=self.tokenizer)
        logger.info("✅ TT Qwen Model Initialized")

    def chat(
        self, msgs: List[Dict[str, Any]], tokenizer: Any, sampling: bool = True, max_new_tokens: int = 128, **kwargs
    ):
        logger.info("TT Qwen Chat Triggered")

        msgs_processed = copy.deepcopy(msgs)
        images = []

        for msg in msgs_processed:
            content = msg["content"]
            if isinstance(content, list):
                new_content_parts = []
                for item in content:
                    if isinstance(item, str):
                        new_content_parts.append(item)
                    elif hasattr(item, "mode"):
                        images.append(item)
                        new_content_parts.append("(<image>./</image>)")
                msg["content"] = "\n".join(new_content_parts)

        prompt = tokenizer.apply_chat_template(msgs_processed, tokenize=False, add_generation_prompt=True)

        try:
            model_inputs = self.reference_model.processor(
                [prompt],
                [images] if images else None,
                max_slice_nums=None,
                use_image_id=False,
                chunk_input=True,
                return_tensors="pt",
                max_length=4096,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            logger.info("Generating Multimodal Embeddings via Reference Model...")
            inputs_embeds, _ = self.reference_model.get_vllm_embedding(model_inputs)
            logger.info(f"Embeddings Generated. Shape: {inputs_embeds.shape}")

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise e

        # Use MiniCPM-o Default Sampling Parameters
        return self._generate_with_embeds(
            inputs_embeds, max_new_tokens, self.tokenizer, temperature=0.7, top_k=50, top_p=0.8
        )

    def _generate_with_embeds(self, inputs_embeds, max_new_tokens, tokenizer, temperature=0.7, top_k=50, top_p=0.8):
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        batch_size, seq_len, hidden_dim = inputs_embeds.shape

        # --- Pad Sequence Length ---
        original_seq_len = seq_len
        padding_needed = 0
        if seq_len % 128 != 0:
            padding_needed = 128 - (seq_len % 128)
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, padding_needed))

        padded_seq_len = inputs_embeds.shape[1]
        logger.info(f"Running Prefill on TT. Original Len: {original_seq_len}, Padded: {padded_seq_len}")

        tt_embeds = ttnn.from_torch(
            inputs_embeds.view(1, 1, padded_seq_len, hidden_dim),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

        transformer_model = self.model[0]

        tt_rot_mats_prefill_global = [
            transformer_model.rope_setup.cos_matrix[:, :, :padded_seq_len, :],
            transformer_model.rope_setup.sin_matrix[:, :, :padded_seq_len, :],
        ]

        tt_page_table = None
        if self.page_table is not None:
            tt_page_table = ttnn.from_torch(
                self.page_table,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        # Run Prefill
        logits = transformer_model.ttnn_prefill_forward(
            x=tt_embeds,
            rot_mats_global=tt_rot_mats_prefill_global,
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache[0],
            user_id=0,
            get_last_token=(original_seq_len - 1) // 32 * 32,
        )

        # Extract Initial Token Logits
        offset_in_chunk = (original_seq_len - 1) - ((original_seq_len - 1) // 32 * 32)
        logits_torch = ttnn.to_torch(logits)
        last_token_logits = logits_torch[0, 0, offset_in_chunk, :]

        # Sample First Token
        prefilled_token = self._manual_hf_sampling(last_token_logits.unsqueeze(0), temperature, top_k, top_p)
        logger.info(f"Prefilled Token ID: {prefilled_token.item()}")

        logger.info("Running Decode on TT (CPU Sampling)...")

        out_tok = prefilled_token
        current_pos = torch.tensor([original_seq_len])
        generated_ids = []

        for i in range(max_new_tokens):
            logits = self.generator.decode_forward_text(
                out_tok, current_pos, enable_trace=False, page_table=self.page_table, kv_cache=self.tt_kv_cache
            )

            # Extract next token logits
            next_token_logits = logits[0, 0, :].unsqueeze(0)

            # Sample next token
            out_tok = self._manual_hf_sampling(next_token_logits, temperature, top_k, top_p)
            token_id = out_tok.item()

            if token_id in tokenizer.all_special_ids or token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            current_pos += 1

        res_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return res_text

    def _manual_hf_sampling(self, logits, temperature=0.7, top_k=50, top_p=0.8):
        """
        Robust Polyfill for HF Sampling.
        Supports: Temperature, Top-K, and Top-P (Nucleus).
        Repetition penalty is removed to prevent grammar breakage.
        """
        # 1. Temperature
        if temperature > 0:
            logits = logits / temperature

        # 2. Top-K Filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            # v[:, -1] is the k-th largest value. Mask anything smaller.
            logits[logits < v[:, [-1]]] = float("-inf")

        # 3. Top-P (Nucleus) Filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Restore order and mask
            # FIX: Use correct dim=1 for vocabulary dimension
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # 4. Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.unsqueeze(0)
