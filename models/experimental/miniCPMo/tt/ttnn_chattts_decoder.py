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
import torch
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def _compute_rotary_cos_sin(head_dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
    """Precompute rotary embedding cos/sin tables."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()  # [max_pos, head_dim]
    sin = emb.sin()  # [max_pos, head_dim]
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple:
    """Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine table [max_pos, head_dim]
        sin: Sine table [max_pos, head_dim]
        position_ids: Position indices [batch, seq_len] or [seq_len]

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied
    """
    # Get positions - handle both 1D and 2D position_ids
    if position_ids.dim() == 1:
        positions = position_ids
    else:
        positions = position_ids.squeeze(0)

    # Index into cos/sin tables: [seq_len, head_dim]
    cos_pos = cos[positions]  # [seq_len, head_dim]
    sin_pos = sin[positions]  # [seq_len, head_dim]

    # Expand for broadcasting: [1, 1, seq_len, head_dim]
    cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)
    sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)

    # Apply rotary embeddings
    q_embed = (q * cos_pos) + (_rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (_rotate_half(k) * sin_pos)

    return q_embed, k_embed


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

        # Precompute rotary embeddings (cos/sin tables)
        self.rotary_cos, self.rotary_sin = _compute_rotary_cos_sin(self.head_dim, max_position_embeddings)

        # TTNN cos/sin cache for native RoPE (will be initialized when device is set)
        # Shape: [1, 1, max_pos, head_dim] for ttnn.experimental.rotary_embedding
        self.rotary_cos_ttnn = None
        self.rotary_sin_ttnn = None

        # Compute kernel configs (following TTNN LLM patterns)
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2)
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)

        # Core grid for matmul operations
        # Use smaller grid for decode (small tensors) to reduce overhead
        self.core_grid = ttnn.CoreGrid(y=4, x=8)  # 32 cores

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
        # Initialize TTNN cos/sin cache for native RoPE (decode mode optimization)
        if self.rotary_cos_ttnn is None:
            cos_cache = self.rotary_cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, max_pos, head_dim]
            sin_cache = self.rotary_sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            self.rotary_cos_ttnn = ttnn.from_torch(
                cos_cache, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            self.rotary_sin_ttnn = ttnn.from_torch(
                sin_cache, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

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
        cache_position: Optional["torch.Tensor"] = None,
    ) -> tuple:
        """
        Forward pass of ChatTTS decoder.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: List of (key, value) tuples for each layer
            use_cache: Whether to return key/value states for caching
            cache_position: Position indices for KV cache (torch.Tensor), used for incremental prefill

        Returns:
            tuple: (last_hidden_state, past_key_values)
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape

        # Determine mode based on sequence length
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        # Apply transformer layers
        for layer_idx, layer_weights in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, new_past_key_value = self._transformer_layer(
                hidden_states,
                layer_weights,
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
                cache_position,
                layer_idx,
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
        cache_position: Optional["torch.Tensor"] = None,
        layer_idx: int = -1,
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
            hidden_states,
            layer_weights["self_attn"],
            attention_mask,
            past_key_value,
            use_cache,
            cache_position,
            position_ids,
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
        cache_position: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> tuple:
        """
        Self-attention mechanism with support for both prefill and decode modes.

        Prefill mode (seq_len > 1): Uses standard reshape operations
        Decode mode (seq_len == 1): Uses optimized decode-specific operations

        Args:
            hidden_states: Input hidden states
            attn_weights: Attention layer weights
            attention_mask: Optional attention mask
            past_key_value: Optional past key/value for KV cache
            use_cache: Whether to return updated KV cache
            cache_position: Position indices for incremental prefill (torch.Tensor)
            position_ids: Position IDs for decode mode (torch.Tensor)

        Returns:
            tuple: (attn_output, past_key_value)
        """
        if len(hidden_states.shape) == 4:
            hidden_states = ttnn.squeeze(hidden_states, 0)
        batch_size, seq_len, _ = hidden_states.shape

        # Determine mode based on sequence length
        is_prefill = seq_len > 1

        # Initialize storage variable (used in decode path)
        new_past_key_value_storage = None

        # Reshape to 4D for ttnn.linear
        hidden_states_4d = ttnn.unsqueeze(hidden_states, dim=1)

        # Project Q, K, V separately
        query = ttnn.linear(
            hidden_states_4d,
            attn_weights["q_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
            core_grid=self.core_grid,
        )

        key = ttnn.linear(
            hidden_states_4d,
            attn_weights["k_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
            core_grid=self.core_grid,
        )

        value = ttnn.linear(
            hidden_states_4d,
            attn_weights["v_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
            core_grid=self.core_grid,
        )

        # Squeeze back to 3D: [batch, seq_len, hidden_size]
        query = ttnn.squeeze(query, dim=1)
        key = ttnn.squeeze(key, dim=1)
        value = ttnn.squeeze(value, dim=1)

        if is_prefill:
            # PREFILL MODE: Use standard reshape operations
            # Reshape to [batch, seq_len, num_heads, head_dim]
            query = ttnn.reshape(query, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
            key = ttnn.reshape(key, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
            value = ttnn.reshape(value, (batch_size, seq_len, self.num_attention_heads, self.head_dim))

            # Permute to [batch, heads, seq, dim] for SDPA
            query = ttnn.permute(query, (0, 2, 1, 3))
            key = ttnn.permute(key, (0, 2, 1, 3))
            value = ttnn.permute(value, (0, 2, 1, 3))

            # Apply RoPE (Rotary Position Embeddings) to Q and K
            # This must happen BEFORE concatenating with past KV cache
            # Use cache_position if provided, otherwise use sequential positions [0, 1, 2, ...]
            rope_positions = cache_position
            if rope_positions is None:
                # For initial prefill, create sequential positions
                rope_positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

            # For prefill mode, use PyTorch RoPE (TTNN RoPE without token_idx has different behavior)
            # Convert Q, K to torch for RoPE application
            query_torch = ttnn.to_torch(query).to(torch.bfloat16)
            key_torch = ttnn.to_torch(key).to(torch.bfloat16)

            # Apply RoPE to Q and K
            query_torch, key_torch = _apply_rotary_pos_emb(
                query_torch, key_torch, self.rotary_cos, self.rotary_sin, rope_positions
            )

            # Convert back to TTNN
            query = torch_to_ttnn(
                query_torch, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
            )
            key = torch_to_ttnn(key_torch, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

            # KV caching: concatenate with past key/value if provided and non-empty
            if past_key_value is not None:
                past_key, past_value = past_key_value
                # Check if past cache has any content (seq_len > 0)
                past_seq_len = past_key.shape[2] if hasattr(past_key, "shape") else 0

                if past_seq_len > 0:
                    # Convert from torch tensor to ttnn if needed
                    if not isinstance(past_key, ttnn.Tensor):
                        past_key = torch_to_ttnn(
                            past_key,  # Already in [batch, heads, seq, dim] format (with RoPE already applied)
                            self.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            layout=ttnn.TILE_LAYOUT,
                        )
                    if not isinstance(past_value, ttnn.Tensor):
                        past_value = torch_to_ttnn(
                            past_value,
                            self.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            layout=ttnn.TILE_LAYOUT,
                        )

                    # Concatenate along sequence dimension
                    key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
                    value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                    key = ttnn.concat([past_key, key], dim=2)
                    value = ttnn.concat([past_value, value], dim=2)

            # Store current key/value for next iteration if caching
            new_past_key_value = (key, value) if use_cache else None

            # SDPA requires inputs in DRAM
            query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
            key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
            value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

            # Check if this is incremental prefill (Q_len != K_len)
            q_len = query.shape[2]
            k_len = key.shape[2]
            is_incremental_prefill = q_len != k_len

            if is_incremental_prefill and cache_position is not None:
                # Incremental prefill: TTNN SDPA requires Q and K to be tile-aligned (multiples of 32)
                # Since our sequences may not meet this requirement, use manual attention computation
                # Create causal mask where query at position cache_position[i] can attend to K positions <= cache_position[i]
                cache_pos = cache_position.squeeze(0) if cache_position.dim() > 1 else cache_position
                # Create position indices for K
                k_positions = torch.arange(k_len, device=cache_pos.device)
                # For each query position, it can attend to K positions <= its cache position
                # cache_pos has shape [q_len], k_positions has shape [k_len]
                # Result: [q_len, k_len] where mask[i, j] = True if k_positions[j] <= cache_pos[i]
                causal_mask = k_positions.unsqueeze(0) <= cache_pos.unsqueeze(1)
                # Convert to attention mask format: 0 for attend, -inf for mask
                attn_mask_values = torch.where(
                    causal_mask,
                    torch.tensor(0.0, dtype=torch.bfloat16),
                    torch.tensor(float("-inf"), dtype=torch.bfloat16),
                )
                # Add batch and head dimensions: [1, 1, q_len, k_len]
                attn_mask_values = attn_mask_values.unsqueeze(0).unsqueeze(0)
                # Convert to TTNN
                causal_attn_mask = torch_to_ttnn(
                    attn_mask_values,
                    self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )

                # Manual attention: scores = Q @ K^T / sqrt(d), then softmax, then @ V
                # query: [batch, heads, q_len, dim], key: [batch, heads, k_len, dim]
                scale = 1.0 / (self.head_dim**0.5)

                # Transpose key for matmul: [batch, heads, dim, k_len]
                key_t = ttnn.permute(key, (0, 1, 3, 2))

                # Compute attention scores: [batch, heads, q_len, k_len]
                attn_scores = ttnn.matmul(
                    query,
                    key_t,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    core_grid=self.core_grid,
                )

                # Scale
                attn_scores = ttnn.multiply(attn_scores, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                # Apply causal mask
                attn_scores = ttnn.add(attn_scores, causal_attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                # Softmax over last dimension
                attn_probs = ttnn.softmax(attn_scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                # Apply attention to values: [batch, heads, q_len, dim]
                attn_output = ttnn.matmul(
                    attn_probs,
                    value,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    core_grid=self.core_grid,
                )

                # Cleanup
                ttnn.deallocate(key_t)
                ttnn.deallocate(attn_scores)
                ttnn.deallocate(attn_probs)
                ttnn.deallocate(causal_attn_mask)
            else:
                # Standard prefill with causal mask - use optimized SDPA
                attn_output = ttnn.transformer.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    is_causal=True,
                    scale=1.0 / (self.head_dim**0.5),
                    compute_kernel_config=self.compute_kernel_config_sdpa,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            # Reshape back: [batch, heads, seq, dim] -> [batch, seq, heads, dim] -> [batch, seq, hidden]
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
            attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        else:
            # DECODE MODE: Use optimized decode-specific operations
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

            # Apply RoPE to the new single-token Q and K using TTNN native RoPE
            # Determine position for this decode step
            if cache_position is not None:
                rope_pos = cache_position.item() if cache_position.numel() == 1 else cache_position[0, 0].item()
            elif position_ids is not None:
                rope_pos = position_ids.item() if position_ids.numel() == 1 else position_ids[0, 0].item()
            elif past_key_value is not None:
                # Infer position from past KV cache length (shape is [batch, heads, seq, dim])
                past_len = past_key_value[0].shape[2] if hasattr(past_key_value[0], "shape") else 0
                rope_pos = past_len
            else:
                rope_pos = 0

            # Move Q, K to DRAM for RoPE processing (convert from HEIGHT_SHARDED)
            query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
            key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)

            # nlp_create_qkv_heads_decode outputs [batch=1, seq=1, heads=12, dim=64]
            # Apply TTNN native RoPE per-head (TTNN RoPE requires dim[1]=1)
            # Convert to torch for slicing, process in TTNN, convert back
            query_torch = ttnn.to_torch(query).to(torch.bfloat16)
            key_torch = ttnn.to_torch(key).to(torch.bfloat16)

            q_rope_results = []
            k_rope_results = []

            for head_idx in range(self.num_attention_heads):
                # Extract single head: [1, 1, 1, 64]
                q_head = query_torch[:, :, head_idx : head_idx + 1, :]
                k_head = key_torch[:, :, head_idx : head_idx + 1, :]

                # Permute for TTNN RoPE: [batch=1, seq=1, 1, 64] -> [seq=1, 1, batch=1, 64]
                q_transposed = q_head.permute(1, 2, 0, 3)
                k_transposed = k_head.permute(1, 2, 0, 3)

                # Convert to TTNN and apply RoPE
                q_tt = ttnn.from_torch(q_transposed, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                k_tt = ttnn.from_torch(k_transposed, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

                q_rope_tt = ttnn.experimental.rotary_embedding(
                    q_tt, self.rotary_cos_ttnn, self.rotary_sin_ttnn, rope_pos
                )
                k_rope_tt = ttnn.experimental.rotary_embedding(
                    k_tt, self.rotary_cos_ttnn, self.rotary_sin_ttnn, rope_pos
                )

                # Convert back, slice off tile padding, permute to original format
                q_rope = ttnn.to_torch(q_rope_tt)[:1, :1, :1, : self.head_dim].permute(2, 0, 1, 3)
                k_rope = ttnn.to_torch(k_rope_tt)[:1, :1, :1, : self.head_dim].permute(2, 0, 1, 3)

                q_rope_results.append(q_rope)
                k_rope_results.append(k_rope)

            # Concatenate all heads: [1, 1, 12, 64]
            query_torch = torch.cat(q_rope_results, dim=2)
            key_torch = torch.cat(k_rope_results, dim=2)

            # Convert back to TTNN
            query = torch_to_ttnn(
                query_torch, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
            )
            key = torch_to_ttnn(key_torch, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

            # KV caching: concatenate with past key/value if provided
            # Internal storage: [batch, seq, heads, dim] (concat-friendly on dim=1)
            # Pipeline expects: [batch, heads, seq, dim]
            if past_key_value is not None:
                past_key, past_value = past_key_value
                # Convert from torch tensor to ttnn if needed
                if not isinstance(past_key, ttnn.Tensor):
                    # torch past_key is [batch, heads, seq, dim] - permute to storage format
                    past_key = torch_to_ttnn(
                        past_key.permute(0, 2, 1, 3),  # to [batch, seq, heads, dim]
                        self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    past_value = torch_to_ttnn(
                        past_value.permute(0, 2, 1, 3),
                        self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.TILE_LAYOUT,
                    )
                # else: ttnn KV is already in storage format [batch, seq, heads, dim]

                # Current key/value from RoPE are [batch, seq=1, heads, dim] - already in storage format
                # Concat along seq dim (dim=1)
                key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
                value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                key = ttnn.concat([past_key, key], dim=1)
                value = ttnn.concat([past_value, value], dim=1)

            # key/value are in storage format [batch, seq, heads, dim]
            # Store in storage format (will be converted to pipeline format on return)
            new_past_key_value_storage = (key, value) if use_cache else None

            # Permute to SDPA format [batch, heads, seq, dim] for attention
            key_sdpa = ttnn.permute(key, (0, 2, 1, 3))
            value_sdpa = ttnn.permute(value, (0, 2, 1, 3))

            # Ensure inputs are in DRAM for attention
            query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
            key_sdpa = ttnn.to_memory_config(key_sdpa, ttnn.DRAM_MEMORY_CONFIG)
            value_sdpa = ttnn.to_memory_config(value_sdpa, ttnn.DRAM_MEMORY_CONFIG)

            # Query is [batch, seq=1, heads, dim], Key/Value are [batch, heads, seq, dim]
            # Permute query to match key/value format: [batch, heads, seq=1, dim]
            query = ttnn.permute(query, (0, 2, 1, 3))

            # Use manual attention for decode mode
            # query: [batch, heads, seq=1, dim], key_sdpa/value_sdpa: [batch, heads, seq, dim]
            scale = 1.0 / (self.head_dim**0.5)

            # Transpose key for matmul: [batch, heads, dim, seq]
            key_t = ttnn.permute(key_sdpa, (0, 1, 3, 2))

            # Compute attention scores: [batch, heads, 1, seq]
            attn_scores = ttnn.matmul(
                query,
                key_t,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                core_grid=self.core_grid,
            )

            # Scale
            attn_scores = ttnn.multiply(attn_scores, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Apply attention mask if provided (for streaming TTS chunk masking)
            if attention_mask is not None:
                if not isinstance(attention_mask, ttnn.Tensor):
                    attn_mask_ttnn = torch_to_ttnn(
                        attention_mask.to(torch.bfloat16),
                        self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.TILE_LAYOUT,
                    )
                else:
                    attn_mask_ttnn = attention_mask
                attn_scores = ttnn.add(attn_scores, attn_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Softmax over last dimension
            attn_probs = ttnn.softmax(attn_scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Apply attention to values: [batch, heads, 1, dim]
            attn_output = ttnn.matmul(
                attn_probs,
                value_sdpa,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                core_grid=self.core_grid,
            )

            # Cleanup temporary tensors
            ttnn.deallocate(key_sdpa)
            ttnn.deallocate(value_sdpa)
            ttnn.deallocate(key_t)
            ttnn.deallocate(attn_scores)
            ttnn.deallocate(attn_probs)

            # Reshape back: [batch, heads, 1, dim] -> [batch, 1, heads, dim] -> [batch, 1, hidden]
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
            attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Output projection
        attn_output_4d = ttnn.unsqueeze(attn_output, dim=1)

        attn_output = ttnn.linear(
            attn_output_4d,
            attn_weights["o_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
            core_grid=self.core_grid,
        )

        # Squeeze back to 3D
        attn_output = ttnn.squeeze(attn_output, dim=1)

        # For decode path: convert storage format [batch, seq, heads, dim] to pipeline format [batch, heads, seq, dim]
        if new_past_key_value_storage is not None:
            storage_k, storage_v = new_past_key_value_storage
            # Permute to pipeline format [batch, heads, seq, dim]
            new_past_key_value = (
                ttnn.permute(storage_k, (0, 2, 1, 3)),
                ttnn.permute(storage_v, (0, 2, 1, 3)),
            )
        # For prefill path: new_past_key_value is already set in the correct format

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
            core_grid=self.core_grid,
        )

        # Up projection
        up = ttnn.linear(
            hidden_states_4d,
            mlp_weights["up_proj"]["weight"],
            bias=None,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=get_activations_memory_config(),
            core_grid=self.core_grid,
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
            core_grid=self.core_grid,
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
                core_grid=self.core_grid,
            )
            logit = ttnn.squeeze(logit_4d, dim=1)
            logits.append(logit)

        return logits
