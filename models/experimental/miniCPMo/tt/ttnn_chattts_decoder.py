# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of ConditionalChatTTS Decoder for miniCPMo.

Implements the transformer decoder component of ChatTTS with:
- Input embeddings (text, audio codes, speaker conditioning)
- Llama-style transformer decoder layers
- Output heads for audio code prediction
"""

import ttnn
from typing import Optional, List
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
    )


class TtnnChatTTSDecoder:
    """
    TTNN implementation of ConditionalChatTTS transformer decoder.

    This implements the core transformer decoder of ChatTTS that:
    1. Embeds text tokens, audio codes, and speaker embeddings
    2. Applies LLM conditioning via projection layer
    3. Runs through Llama-style transformer layers
    4. Produces logits for audio code prediction

    Architecture:
        - Input embeddings: text + audio codes (4 codebooks) + speaker conditioning
        - LLM projector: projects LLM hidden states to TTS embedding space
        - Transformer decoder: Llama-style layers with causal attention
        - Output heads: 4 linear heads (weight normalized) for audio code prediction
    """

    def __init__(
        self,
        device: ttnn.Device,
        llm_dim: int = 3584,  # Qwen2.5 hidden size
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        intermediate_size: int = 3072,
        num_text_tokens: int = 21178,
        num_audio_tokens: int = 626,
        num_vq: int = 4,
        num_spk_embs: int = 1,
        max_position_embeddings: int = 4096,
    ):
        self.device = device
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_text_tokens = num_text_tokens
        self.num_audio_tokens = num_audio_tokens
        self.num_vq = num_vq
        self.num_spk_embs = num_spk_embs
        self.max_position_embeddings = max_position_embeddings

        # Derived dimensions
        self.head_dim = hidden_size // num_attention_heads

        # Compute kernel configs (following TTNN LLM patterns)
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2)
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)

        # Initialize components that will be loaded
        self.projector = None  # LLM hidden state projector

        # Embeddings
        self.emb_text = None  # Text token embeddings
        self.emb_code = []  # Audio code embeddings (4 codebooks)
        for _ in range(num_vq):
            self.emb_code.append(None)

        # Transformer layers - full LlamaModel implementation
        self.layers = []
        for layer_idx in range(num_hidden_layers):
            layer_weights = self._create_transformer_layer_weights(layer_idx)
            self.layers.append(layer_weights)

        # Output heads (weight normalized)
        self.head_code = []
        for _ in range(num_vq):
            self.head_code.append(None)

        # Final layer norm
        self.norm = None

        logger.info(
            f"TtnnChatTTSDecoder initialized: hidden_size={hidden_size}, "
            f"num_layers={num_hidden_layers}, num_heads={num_attention_heads}, "
            f"intermediate_size={intermediate_size}, num_vq={num_vq}"
        )

    def _create_transformer_layer_weights(self, layer_idx: int) -> dict:
        """
        Create weights for a single transformer layer.

        Each layer contains:
        - Self-attention: q_proj, k_proj, v_proj, o_proj
        - MLP: gate_proj, up_proj, down_proj
        - RMS norms: input_layernorm, post_attention_layernorm

        Args:
            layer_idx: Layer index for weight naming

        Returns:
            Dict containing all layer weights
        """
        layer_weights = {}

        # Self-attention weights
        layer_weights["self_attn"] = {
            "q_proj": {
                "weight": None,  # [hidden_size, hidden_size]
            },
            "k_proj": {
                "weight": None,  # [hidden_size, hidden_size]
            },
            "v_proj": {
                "weight": None,  # [hidden_size, hidden_size]
            },
            "o_proj": {
                "weight": None,  # [hidden_size, hidden_size]
            },
        }

        # MLP weights (Llama-style: gate_proj -> up_proj -> down_proj)
        layer_weights["mlp"] = {
            "gate_proj": {
                "weight": None,  # [intermediate_size, hidden_size]
            },
            "up_proj": {
                "weight": None,  # [intermediate_size, hidden_size]
            },
            "down_proj": {
                "weight": None,  # [hidden_size, intermediate_size]
            },
        }

        # RMS normalization weights
        layer_weights["input_layernorm"] = {
            "weight": None,  # [hidden_size]
        }
        layer_weights["post_attention_layernorm"] = {
            "weight": None,  # [hidden_size]
        }

        return layer_weights

    def load_weights(self, weights_dict: dict):
        """
        Load weights from PyTorch state dict.

        Args:
            weights_dict: Dictionary containing weight tensors with keys:
                - 'projector.linear1.weight', 'projector.linear1.bias': LLM projector
                - 'projector.linear2.weight', 'projector.linear2.bias': LLM projector
                - 'emb_text.weight': Text embeddings
                - 'emb_code.0.weight', ..., 'emb_code.3.weight': Audio code embeddings
                - 'model.layers.{i}.*': Transformer layer weights
                - 'model.norm.weight': Final layer norm
                - 'head_code.{i}.weight': Audio code prediction heads
        """
        logger.info("Loading ChatTTS Decoder weights...")

        # LLM projector
        if "projector.linear1.weight" in weights_dict:
            # MLP projector
            self.projector = {
                "linear1_weight": torch_to_ttnn(
                    weights_dict["projector.linear1.weight"].transpose(-1, -2),
                    self.device,
                    memory_config=get_weights_memory_config(),
                ),
                "linear1_bias": torch_to_ttnn(
                    weights_dict["projector.linear1.bias"],
                    self.device,
                    memory_config=get_weights_memory_config(),
                ),
                "linear2_weight": torch_to_ttnn(
                    weights_dict["projector.linear2.weight"].transpose(-1, -2),
                    self.device,
                    memory_config=get_weights_memory_config(),
                ),
                "linear2_bias": torch_to_ttnn(
                    weights_dict["projector.linear2.bias"],
                    self.device,
                    memory_config=get_weights_memory_config(),
                ),
            }
        else:
            # Linear projector
            self.projector = {
                "weight": torch_to_ttnn(
                    weights_dict["projector.weight"].transpose(-1, -2),
                    self.device,
                    memory_config=get_weights_memory_config(),
                ),
            }

        # Text embeddings
        self.emb_text = torch_to_ttnn(
            weights_dict["emb_text.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # Audio code embeddings (4 codebooks)
        for i in range(self.num_vq):
            self.emb_code[i] = torch_to_ttnn(
                weights_dict[f"emb_code.{i}.weight"],
                self.device,
                memory_config=get_weights_memory_config(),
            )

        # Transformer layers
        self.layers = []
        for layer_idx in range(self.num_hidden_layers):
            layer_weights = self._extract_layer_weights(weights_dict, layer_idx)
            self.layers.append(layer_weights)

        # Final layer norm
        self.norm = torch_to_ttnn(
            weights_dict["model.norm.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # Output heads (handle weight normalization parametrization)
        for i in range(self.num_vq):
            head_key = f"head_code.{i}.weight"
            param_key0 = f"head_code.{i}.parametrizations.weight.original0"
            param_key1 = f"head_code.{i}.parametrizations.weight.original1"

            if param_key0 in weights_dict and param_key1 in weights_dict:
                # Reconstruct from weight normalization parametrization
                direction = weights_dict[param_key0]
                magnitude = weights_dict[param_key1]
                weight = direction * magnitude
                logger.debug(f"Reconstructed head_code.{i}.weight from parametrization")
            elif head_key in weights_dict:
                weight = weights_dict[head_key]
            else:
                raise KeyError(f"Missing head_code.{i} weights")

            self.head_code[i] = torch_to_ttnn(
                weight.transpose(-1, -2),
                self.device,
                memory_config=get_weights_memory_config(),
            )

        logger.info("✅ ChatTTS Decoder weights loaded")

    def _extract_layer_weights(self, weights_dict: dict, layer_idx: int) -> dict:
        """
        Extract weights for a single transformer layer.
        """
        prefix = f"model.layers.{layer_idx}"
        layer_weights = self._create_transformer_layer_weights(layer_idx)

        # Self-attention weights
        layer_weights["self_attn"]["q_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.self_attn.q_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["self_attn"]["k_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.self_attn.k_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["self_attn"]["v_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.self_attn.v_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["self_attn"]["o_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.self_attn.o_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # MLP weights
        layer_weights["mlp"]["gate_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.mlp.gate_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["mlp"]["up_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.mlp.up_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["mlp"]["down_proj"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.mlp.down_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # RMS norms
        layer_weights["input_layernorm"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.input_layernorm.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        layer_weights["post_attention_layernorm"]["weight"] = torch_to_ttnn(
            weights_dict[f"{prefix}.post_attention_layernorm.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        return layer_weights

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of ChatTTS decoder.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs [batch_size, seq_len]

        Returns:
            last_hidden_state: Output hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        logger.info(
            f"[TTNN ChatTTS Decoder] Forward pass: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        )

        hidden_states = inputs_embeds

        # Apply transformer layers
        for layer_idx, layer_weights in enumerate(self.layers):
            logger.debug(f"[TTNN ChatTTS Decoder] Processing layer {layer_idx}/{self.num_hidden_layers}")

            hidden_states = self._transformer_layer(hidden_states, layer_weights, attention_mask, position_ids)

        # Final RMS norm
        logger.debug("[TTNN ChatTTS Decoder] Applying final RMS norm")
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=self.norm,
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        logger.info(f"[TTNN ChatTTS Decoder] Forward pass completed: output shape={hidden_states.shape}")
        return hidden_states

    def _transformer_layer(
        self,
        hidden_states: ttnn.Tensor,
        layer_weights: dict,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Process one transformer layer.

        Llama-style layer:
        1. RMSNorm (input)
        2. Self-attention + residual
        3. RMSNorm (post-attention)
        4. MLP + residual
        """
        logger.debug("[TTNN ChatTTS Decoder] Transformer layer: Input RMS norm")
        residual = hidden_states

        # 1. Input RMS normalization
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=layer_weights["input_layernorm"]["weight"],
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        # 2. Self-attention
        logger.debug("[TTNN ChatTTS Decoder] Transformer layer: Self-attention")
        attn_output = self._self_attention(hidden_states, layer_weights["self_attn"], attention_mask)

        # 3. Residual connection
        hidden_states = ttnn.add(attn_output, residual, memory_config=get_activations_memory_config())

        # 4. Post-attention RMS normalization
        logger.debug("[TTNN ChatTTS Decoder] Transformer layer: Post-attention RMS norm")
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=layer_weights["post_attention_layernorm"]["weight"],
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        # 5. MLP
        logger.debug("[TTNN ChatTTS Decoder] Transformer layer: MLP")
        mlp_output = self._mlp(hidden_states, layer_weights["mlp"])

        # 6. Residual connection
        hidden_states = ttnn.add(mlp_output, residual, memory_config=get_activations_memory_config())

        logger.debug("[TTNN ChatTTS Decoder] Transformer layer: Completed")
        return hidden_states

    def _self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_weights: dict,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Self-attention mechanism.
        """
        batch_size, seq_len, _ = hidden_states.shape
        logger.debug(f"[TTNN ChatTTS Decoder] Self-attention: batch_size={batch_size}, seq_len={seq_len}")

        # Reshape to 4D for ttnn.linear
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)  # [B, 1, S, H]

        # Query projection
        logger.debug("[TTNN ChatTTS Decoder] Self-attention: Q projection")
        query = ttnn.linear(
            hidden_states_4d,
            attn_weights["q_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Key projection
        key = ttnn.linear(
            hidden_states_4d,
            attn_weights["k_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Value projection
        value = ttnn.linear(
            hidden_states_4d,
            attn_weights["v_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        query = ttnn.squeeze(query, dim=1)  # [B, S, H]
        key = ttnn.squeeze(key, dim=1)  # [B, S, H]
        value = ttnn.squeeze(value, dim=1)  # [B, S, H]

        # Reshape for multi-head attention
        logger.debug(f"[TTNN ChatTTS Decoder] Self-attention: Reshaping for {self.num_attention_heads} heads")
        query = self._reshape_for_attention(query, self.num_attention_heads)
        key = self._reshape_for_attention(key, self.num_attention_heads)
        value = self._reshape_for_attention(value, self.num_attention_heads)

        # SDPA requires all inputs to be in DRAM
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

        # Scaled dot-product attention
        logger.debug("[TTNN ChatTTS Decoder] Self-attention: Scaled dot-product attention")
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=True,  # Causal attention for autoregressive generation
            scale=1.0 / (self.head_dim**0.5),
            compute_kernel_config=self.compute_kernel_config_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape back from attention format
        attn_output = self._reshape_from_attention(attn_output, seq_len)

        # Output projection (reshape to 4D for ttnn.linear)
        logger.debug("[TTNN ChatTTS Decoder] Self-attention: Output projection")
        attn_output_4d = ttnn.unsqueeze(attn_output, dim=1)  # [B, 1, S, H]

        attn_output = ttnn.linear(
            attn_output_4d,
            attn_weights["o_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        attn_output = ttnn.squeeze(attn_output, dim=1)  # [B, S, H]

        logger.debug(f"[TTNN ChatTTS Decoder] Self-attention: Completed, output_shape={attn_output.shape}")
        return attn_output

    def _mlp(self, hidden_states: ttnn.Tensor, mlp_weights: dict) -> ttnn.Tensor:
        """
        MLP block (Llama-style: gate_proj * up_proj -> down_proj).
        """
        logger.debug("[TTNN ChatTTS Decoder] MLP: Starting")
        # Reshape to 4D for ttnn.linear
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)  # [B, 1, S, H]

        # Gate projection
        logger.debug("[TTNN ChatTTS Decoder] MLP: Gate projection")
        gate = ttnn.linear(
            hidden_states_4d,
            mlp_weights["gate_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Up projection
        logger.debug("[TTNN ChatTTS Decoder] MLP: Up projection")
        up = ttnn.linear(
            hidden_states_4d,
            mlp_weights["up_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # SiLU activation on gate
        logger.debug("[TTNN ChatTTS Decoder] MLP: SiLU activation")
        gate = ttnn.silu(gate)

        # Element-wise multiplication (gate * up)
        logger.debug("[TTNN ChatTTS Decoder] MLP: Gate * Up multiplication")
        hidden_states_expanded = ttnn.mul(gate, up, memory_config=get_activations_memory_config())

        # Down projection
        logger.debug("[TTNN ChatTTS Decoder] MLP: Down projection")
        output = ttnn.linear(
            hidden_states_expanded,
            mlp_weights["down_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        output = ttnn.squeeze(output, dim=1)  # [B, S, H]

        logger.debug(f"[TTNN ChatTTS Decoder] MLP: Completed, output_shape={output.shape}")
        return output

    def _reshape_for_attention(self, x: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        """
        Reshape tensor for multi-head attention.
        """
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // num_heads

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, seq_len, num_heads, head_dim))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        x = ttnn.permute(x, (0, 2, 1, 3))  # [B, H, S, D]

        return x

    def _reshape_from_attention(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """
        Reshape tensor back from multi-head attention format.
        """
        batch_size, num_heads, _, head_dim = x.shape
        embed_dim = num_heads * head_dim

        x = ttnn.permute(x, (0, 2, 1, 3))  # [B, S, H, D]
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, seq_len, embed_dim))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

        return x

    def get_logits(self, hidden_states: ttnn.Tensor) -> List[ttnn.Tensor]:
        """
        Get logits for each codebook from hidden states.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]

        Returns:
            List of logits for each codebook [batch_size, seq_len, num_audio_tokens]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        logger.debug(
            f"[TTNN ChatTTS Decoder] Get logits: batch_size={batch_size}, seq_len={seq_len}, num_vq={self.num_vq}"
        )

        logits = []
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)  # [B, 1, S, H]

        for i in range(self.num_vq):
            logger.debug(f"[TTNN ChatTTS Decoder] Get logits: Processing codebook {i+1}/{self.num_vq}")
            logit_4d = ttnn.linear(
                hidden_states_4d,
                self.head_code[i],
                bias=None,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logit = ttnn.squeeze(logit_4d, dim=1)  # [B, S, num_audio_tokens]
            logits.append(logit)

        logger.info(f"[TTNN ChatTTS Decoder] Get logits: Completed, {len(logits)} codebooks")
        return logits
