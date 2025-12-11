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

    def _create_transformer_layer_weights(self, layer_idx: int) -> dict:
        """Create weights for a single transformer layer."""
        layer_weights = {}

        # Self-attention weights
        layer_weights["self_attn"] = {
            "q_proj": {"weight": None},
            "k_proj": {"weight": None},
            "v_proj": {"weight": None},
            "o_proj": {"weight": None},
        }

        # MLP weights (Llama-style: gate_proj -> up_proj -> down_proj)
        layer_weights["mlp"] = {
            "gate_proj": {"weight": None},
            "up_proj": {"weight": None},
            "down_proj": {"weight": None},
        }

        # RMS normalization weights
        layer_weights["input_layernorm"] = {"weight": None}
        layer_weights["post_attention_layernorm"] = {"weight": None}

        return layer_weights

    def load_weights(self, weights_dict: dict):
        """Load weights from PyTorch state dict."""
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
                direction = weights_dict[param_key0]
                magnitude = weights_dict[param_key1]
                weight = direction * magnitude
            elif head_key in weights_dict:
                weight = weights_dict[head_key]
            else:
                raise KeyError(f"Missing head_code.{i} weights")

            self.head_code[i] = torch_to_ttnn(
                weight.transpose(-1, -2),
                self.device,
                memory_config=get_weights_memory_config(),
            )

    def _extract_layer_weights(self, weights_dict: dict, layer_idx: int) -> dict:
        """Extract weights for a single transformer layer."""
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
        past_key_values: Optional[List[tuple]] = None,
        use_cache: bool = False,
    ) -> tuple:
        """
        Forward pass of ChatTTS decoder.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: List of (key, value) tuples for each layer
            use_cache: Whether to return key/value states for caching

        Returns:
            tuple: (last_hidden_state, past_key_values)
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape

        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        # Apply transformer layers
        for layer_idx, layer_weights in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, new_past_key_value = self._transformer_layer(
                hidden_states, layer_weights, attention_mask, position_ids, past_key_value, use_cache
            )
            if use_cache:
                new_past_key_values.append(new_past_key_value)

        # Final RMS norm
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=self.norm,
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        return hidden_states, new_past_key_values

    def _transformer_layer(
        self,
        hidden_states: ttnn.Tensor,
        layer_weights: dict,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple:
        """
        Process one transformer layer.

        Llama-style layer:
        1. RMSNorm (input)
        2. Self-attention + residual
        3. RMSNorm (post-attention)
        4. MLP + residual
        """
        residual = hidden_states

        # 1. Input RMS normalization
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=layer_weights["input_layernorm"]["weight"],
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        # 2. Self-attention
        attn_output, new_past_key_value = self._self_attention(
            hidden_states, layer_weights["self_attn"], attention_mask, past_key_value, use_cache
        )

        # 3. Residual connection
        hidden_states = ttnn.add(attn_output, residual, memory_config=get_activations_memory_config())

        # 4. Post-attention RMS normalization
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=layer_weights["post_attention_layernorm"]["weight"],
            epsilon=1e-5,
            memory_config=get_activations_memory_config(),
        )

        # 5. MLP
        mlp_output = self._mlp(hidden_states, layer_weights["mlp"])

        # 6. Residual connection
        hidden_states = ttnn.add(mlp_output, residual, memory_config=get_activations_memory_config())

        return hidden_states, new_past_key_value

    def _self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_weights: dict,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple:
        """
        Self-attention mechanism with optional KV caching for decode mode.

        Returns:
            tuple: (attn_output, past_key_value)
        """
        if len(hidden_states.shape) == 4:
            hidden_states = ttnn.squeeze(hidden_states, 0)
        batch_size, seq_len, _ = hidden_states.shape

        # Reshape to 4D for ttnn.linear
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)

        # Project Q, K, V separately
        query = ttnn.linear(
            hidden_states_4d,
            attn_weights["q_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        key = ttnn.linear(
            hidden_states_4d,
            attn_weights["k_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        value = ttnn.linear(
            hidden_states_4d,
            attn_weights["v_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        query = ttnn.squeeze(query, dim=1)
        key = ttnn.squeeze(key, dim=1)
        value = ttnn.squeeze(value, dim=1)

        # Concatenate QKV for proper reshaping
        xqkv_fused = ttnn.concat([query, key, value], dim=-1)
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        # Reshape to 4D format expected by nlp_create_qkv_heads_decode
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, seq_len, xqkv_fused.shape[-1]))

        # Reshape using dedicated decode operation
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # KV caching: concatenate with past key/value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Convert from torch tensor to ttnn if needed
            if not isinstance(past_key, ttnn.Tensor):
                past_key = torch_to_ttnn(
                    past_key.permute(0, 2, 1, 3),
                    self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
            if not isinstance(past_value, ttnn.Tensor):
                past_value = torch_to_ttnn(
                    past_value.permute(0, 2, 1, 3),
                    self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )

            # Convert new K/V to interleaved to match cache format
            key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
            value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

            # Concatenate with past tensors
            key = ttnn.concat([past_key, key], dim=1)
            value = ttnn.concat([past_value, value], dim=1)

            # Permute to SDPA decode format [batch, heads, seq, dim]
            key = ttnn.permute(key, (0, 2, 1, 3))
            value = ttnn.permute(value, (0, 2, 1, 3))

        # Store current key/value for next iteration if caching
        new_past_key_value = (key, value) if use_cache else None

        # SDPA requires all inputs to be in DRAM
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

        # Get current sequence position for decode
        current_seq_pos = key.shape[1] - 1

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=self.num_attention_heads,
            k_chunk_size=32,  # Must divide K sequence length (padded to tile size)
            exp_approx_mode=False,
        )

        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key,
            value,
            is_causal=True,
            cur_pos=[current_seq_pos],
            scale=1.0 / (self.head_dim**0.5),
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Convert to sharded for nlp_concat_heads_decode
        num_cores_for_sharding = batch_size

        shard_spec = ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_for_sharding - 1, 0))}
            ),
            shard_shape=[32, 64],  # Must be tile-aligned
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        attn_output = ttnn.to_memory_config(attn_output, sharded_mem_config)

        # Use nlp_concat_heads_decode to reshape back
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.num_attention_heads,
        )

        # Slice to remove padding: nlp_concat_heads_decode pads to 32 users
        attn_output = ttnn.to_memory_config(attn_output, ttnn.DRAM_MEMORY_CONFIG)
        attn_output = ttnn.slice(
            attn_output,
            [0, 0, 0, 0],
            [1, 1, batch_size, self.hidden_size],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        attn_output = ttnn.squeeze(attn_output, dim=2)

        # Output projection
        attn_output_4d = ttnn.unsqueeze(attn_output, dim=1)

        attn_output = ttnn.linear(
            attn_output_4d,
            attn_weights["o_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        attn_output = ttnn.squeeze(attn_output, dim=1)

        return attn_output, new_past_key_value

    def _mlp(self, hidden_states: ttnn.Tensor, mlp_weights: dict) -> ttnn.Tensor:
        """MLP block (Llama-style: gate_proj * up_proj -> down_proj)."""
        # Reshape to 4D for ttnn.linear
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)

        # Gate projection
        gate = ttnn.linear(
            hidden_states_4d,
            mlp_weights["gate_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Up projection
        up = ttnn.linear(
            hidden_states_4d,
            mlp_weights["up_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # SiLU activation on gate
        gate = ttnn.silu(gate)

        # Element-wise multiplication (gate * up)
        hidden_states_expanded = ttnn.mul(gate, up, memory_config=get_activations_memory_config())

        # Down projection
        output = ttnn.linear(
            hidden_states_expanded,
            mlp_weights["down_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
        )

        # Squeeze back to 3D
        output = ttnn.squeeze(output, dim=1)

        return output

    def get_logits(self, hidden_states: ttnn.Tensor) -> List[ttnn.Tensor]:
        """
        Get logits for each codebook from hidden states.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]

        Returns:
            List of logits for each codebook [batch_size, seq_len, num_audio_tokens]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        logits = []
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)

        for i in range(self.num_vq):
            logit_4d = ttnn.linear(
                hidden_states_4d,
                self.head_code[i],
                bias=None,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logit = ttnn.squeeze(logit_4d, dim=1)
            logits.append(logit)

        return logits
