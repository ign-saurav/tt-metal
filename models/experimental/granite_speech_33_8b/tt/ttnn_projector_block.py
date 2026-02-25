import ttnn
import torch
import math
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class Blip2QFormerIntermediateTTNN:
    """TTNN implementation of Blip2QFormerIntermediate."""

    def __init__(self, device, config):
        self.device = device

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, dense_weight, dense_bias):
        """Load and convert PyTorch weights to TTNN format."""
        self.dense_weight = ttnn.from_torch(
            dense_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.dense_bias = ttnn.from_torch(dense_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward(self, hidden_states):
        # Linear projection (dense)
        hidden_states = ttnn.linear(
            hidden_states, self.dense_weight, bias=self.dense_bias, compute_kernel_config=self.compute_config
        )

        # GELU Activation
        hidden_states = ttnn.gelu(hidden_states)

        return hidden_states


class Blip2QFormerOutputTTNN:
    """TTNN implementation of Blip2QFormerOutput."""

    def __init__(self, device, config):
        self.device = device

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, dense_weight, dense_bias, layernorm_weight, layernorm_bias):
        """Load and convert PyTorch weights to TTNN format."""
        self.dense_weight = ttnn.from_torch(
            dense_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.dense_bias = ttnn.from_torch(dense_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.layernorm_weight = ttnn.from_torch(
            layernorm_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.layernorm_bias = ttnn.from_torch(
            layernorm_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(self, hidden_states, input_tensor):
        # Linear projection (dense)
        hidden_states = ttnn.linear(
            hidden_states, self.dense_weight, bias=self.dense_bias, compute_kernel_config=self.compute_config
        )
        hidden_states = ttnn.add(hidden_states, input_tensor)

        # Layer Normalization
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layernorm_weight,
            bias=self.layernorm_bias,
            compute_kernel_config=self.compute_config,
        )

        return hidden_states


class Blip2QFormerSelfOutputTTNN:
    """TTNN implementation of Blip2QFormerSelfOutput."""

    def __init__(self, device, config):
        self.device = device

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, dense_weight, dense_bias, layernorm_weight, layernorm_bias):
        """Load and convert PyTorch weights to TTNN format."""
        self.dense_weight = ttnn.from_torch(
            dense_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.dense_bias = ttnn.from_torch(dense_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.layernorm_weight = ttnn.from_torch(
            layernorm_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.layernorm_bias = ttnn.from_torch(
            layernorm_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(self, hidden_states, input_tensor):
        # Linear projection (dense)
        hidden_states = ttnn.linear(
            hidden_states,
            self.dense_weight,
            bias=self.dense_bias,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(hidden_states, input_tensor)

        # Layer Normalization
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layernorm_weight,
            bias=self.layernorm_bias,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return hidden_states


class Blip2QFormerMultiHeadAttentionTTNN:
    """TTNN implementation of Blip2QFormerMultiHeadAttention with past_key_value support."""

    def __init__(self, device, config, use_optimized_attention=True):
        self.device = device
        self.config = config

        # Extract config values
        self.hidden_size = config.projector_config.hidden_size
        self.num_attention_heads = config.projector_config.num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_probs_dropout_prob = config.projector_config.attention_probs_dropout_prob

        # Position embedding settings
        self.position_embedding_type = "absolute"
        self.optimized = use_optimized_attention

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.sdpa_compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias):
        """Load and convert PyTorch weights to TTNN format."""
        self.query_weight = ttnn.from_torch(
            query_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.query_bias = ttnn.from_torch(query_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.key_weight = ttnn.from_torch(
            key_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.key_bias = ttnn.from_torch(key_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.value_weight = ttnn.from_torch(
            value_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.value_bias = ttnn.from_torch(value_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def transpose_for_scores(self, x):
        """Transpose and reshape for multi-head attention."""
        shape_tuple = tuple(x.shape)
        new_x_shape = shape_tuple[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ttnn.reshape(x, new_x_shape)
        return ttnn.permute(x, [0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass using ttnn.transformer.scaled_dot_product_attention."""

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(
                ttnn.linear(
                    encoder_hidden_states,
                    self.key_weight,
                    bias=self.key_bias,
                    compute_kernel_config=self.compute_config,
                )
            )
            value_layer = self.transpose_for_scores(
                ttnn.linear(
                    encoder_hidden_states,
                    self.value_weight,
                    bias=self.value_bias,
                    compute_kernel_config=self.compute_config,
                )
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            current_key_layer = self.transpose_for_scores(
                ttnn.linear(
                    hidden_states, self.key_weight, bias=self.key_bias, compute_kernel_config=self.compute_config
                )
            )
            current_value_layer = self.transpose_for_scores(
                ttnn.linear(
                    hidden_states, self.value_weight, bias=self.value_bias, compute_kernel_config=self.compute_config
                )
            )
            key_layer = ttnn.concat([past_key_value[0], current_key_layer], dim=2)
            value_layer = ttnn.concat([past_key_value[1], current_value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(
                ttnn.linear(
                    hidden_states, self.key_weight, bias=self.key_bias, compute_kernel_config=self.compute_config
                )
            )
            value_layer = self.transpose_for_scores(
                ttnn.linear(
                    hidden_states, self.value_weight, bias=self.value_bias, compute_kernel_config=self.compute_config
                )
            )

        mixed_query_layer = ttnn.linear(
            hidden_states, self.query_weight, bias=self.query_bias, compute_kernel_config=self.compute_config
        )
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)
        scale = 1.0 / math.sqrt(self.attention_head_size)

        if self.optimized:
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=32,
                exp_approx_mode=False,
            )

            original_seq_len = query_layer.shape[2]
            original_mask_q_len = attention_mask.shape[2]
            original_mask_k_len = attention_mask.shape[3]

            if query_layer.shape[2] % 32 != 0:
                pad_size = 32 - (query_layer.shape[2] % 32)
                query_layer = ttnn.pad(query_layer, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)
            if key_layer.shape[2] % 32 != 0:
                pad_size = 32 - (key_layer.shape[2] % 32)
                key_layer = ttnn.pad(key_layer, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)
                value_layer = ttnn.pad(value_layer, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)

                q_pad_needed = 32 - (original_mask_q_len % 32) if original_mask_q_len % 32 != 0 else 0
                if original_mask_q_len == 1:
                    q_pad_needed = 32 - 1
                    attention_mask = ttnn.pad(
                        attention_mask, padding=[(0, 0), (0, 0), (0, q_pad_needed), (0, 0)], value=0.0
                    )
                    attention_mask = ttnn.pad(
                        attention_mask, padding=[(0, 0), (0, 0), (0, 0), (0, pad_size)], value=-10000.0
                    )
                else:
                    attention_mask = ttnn.pad(
                        attention_mask, padding=[(0, 0), (0, 0), (0, q_pad_needed), (0, pad_size)], value=-10000.0
                    )
            if query_layer.shape[0] != key_layer.shape[0]:
                batch = key_layer.shape[0]
                query_layer = ttnn.repeat(query_layer, [batch, 1, 1, 1])
            if attention_mask.shape[0] != query_layer.shape[0]:
                batch = query_layer.shape[0]
                attention_mask = ttnn.repeat(attention_mask, [batch, 1, 1, 1])
            context_layer = ttnn.transformer.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                is_causal=False,
                attn_mask=attention_mask,
                scale=scale,
                compute_kernel_config=self.sdpa_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program_config,
            )

            context_layer = ttnn.slice(
                context_layer,
                [0, 0, 0, 0],
                [context_layer.shape[0], context_layer.shape[1], original_seq_len, context_layer.shape[3]],
            )
        else:
            if query_layer.shape[0] != key_layer.shape[0]:
                batch = key_layer.shape[0]
                query_layer = ttnn.repeat(query_layer, [batch, 1, 1, 1])
            key_layer_transposed = ttnn.transpose(key_layer, -2, -1)
            attention_scores = ttnn.matmul(query_layer, key_layer_transposed, compute_kernel_config=self.compute_config)

            scale_factor = 1.0 / math.sqrt(self.attention_head_size)
            attention_scores = ttnn.mul(attention_scores, scale_factor)

            if attention_mask is not None:
                attention_scores = ttnn.add(attention_scores, attention_mask)

            attention_probs = ttnn.softmax(attention_scores, dim=-1)

            if head_mask is not None:
                attention_probs = ttnn.mul(attention_probs, head_mask)

            context_layer = ttnn.matmul(attention_probs, value_layer, compute_kernel_config=self.compute_config)

        context_layer = ttnn.permute(context_layer, [0, 2, 1, 3])
        shape_tuple = tuple(context_layer.shape)
        new_context_layer_shape = shape_tuple[:-2] + (self.all_head_size,)
        context_layer = ttnn.reshape(context_layer, new_context_layer_shape)

        outputs = (context_layer,)
        if output_attentions:
            attention_probs = None
            outputs = (context_layer, attention_probs)

        outputs = outputs + (past_key_value,)
        return outputs


class Blip2QFormerAttentionTTNN:
    def __init__(self, device, config, use_optimized_attention=True):
        self.device = device
        self.config = config
        self.attention = Blip2QFormerMultiHeadAttentionTTNN(device, config, use_optimized_attention)
        self.output = Blip2QFormerSelfOutputTTNN(device, config)

    def prepare_weights(self, attention, output):
        self.attention.prepare_weights(
            attention.query.weight,
            attention.query.bias,
            attention.key.weight,
            attention.key.bias,
            attention.value.weight,
            attention.value.bias,
        )
        self.output.prepare_weights(
            output.dense.weight, output.dense.bias, output.LayerNorm.weight, output.LayerNorm.bias
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.attention.forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        attention_output = self_outputs[0]
        attention_output = self.output.forward(attention_output, hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class Blip2QFormerLayerTTNN:
    """TTNN implementation of Blip2QFormerLayer."""

    def __init__(self, device, config, layer_idx, use_optimized_attention=True):
        self.device = device
        self.config = config
        self.layer_idx = layer_idx
        self.seq_len_dim = 1

        # Initialize attention components
        self.attention = Blip2QFormerAttentionTTNN(device, config, use_optimized_attention)

        # Conditional cross-attention setup
        if layer_idx % config.projector_config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttentionTTNN(device, config, use_optimized_attention)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        # Initialize feedforward components
        if config.projector_config.use_qformer_text_input:
            self.intermediate = Blip2QFormerIntermediateTTNN(device, config)
            self.output = Blip2QFormerOutputTTNN(device, config)

        self.intermediate_query = Blip2QFormerIntermediateTTNN(device, config)
        self.output_query = Blip2QFormerOutputTTNN(device, config)

    def prepare_weights(self, layer):
        """Prepare weights from PyTorch layer."""
        self.attention.prepare_weights(layer.attention.attention, layer.attention.output)

        if self.has_cross_attention:
            self.crossattention.prepare_weights(layer.crossattention.attention, layer.crossattention.output)

        self.intermediate_query.prepare_weights(
            layer.intermediate_query.dense.weight, layer.intermediate_query.dense.bias
        )
        self.output_query.prepare_weights(
            layer.output_query.dense.weight,
            layer.output_query.dense.bias,
            layer.output_query.LayerNorm.weight,
            layer.output_query.LayerNorm.bias,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        self_attention_outputs = self.attention.forward(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = ttnn.slice(
                attention_output, [0, 0, 0], [attention_output.shape[0], query_length, attention_output.shape[2]]
            )

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")

                cross_attention_outputs = self.crossattention.forward(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = self._apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                text_attention_output = ttnn.slice(
                    attention_output,
                    [0, query_length, 0],
                    [attention_output.shape[0], attention_output.shape[1] - query_length, attention_output.shape[2]],
                )

                layer_output_text = self._apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    text_attention_output,
                )

                layer_output = ttnn.concat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = self._apply_chunking_to_forward(
                self.feed_forward_chunk,
                attention_output,
            )

        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """Feed forward chunk for text processing."""
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        """Feed forward chunk for query processing."""
        intermediate_output = self.intermediate_query.forward(attention_output)
        layer_output = self.output_query.forward(intermediate_output, attention_output)
        return layer_output

    def _apply_chunking_to_forward(self, feed_forward_chunk, attention_output):
        """Apply chunking to forward pass."""
        return feed_forward_chunk(attention_output)


class Blip2QFormerEncoderTTNN:
    """TTNN implementation of Blip2QFormerEncoder."""

    def __init__(self, device, config, use_optimized_attention=True):
        self.device = device
        self.config = config
        self.gradient_checkpointing = False

        self.layer = [Blip2QFormerLayerTTNN(device, config, layer_idx, use_optimized_attention) for layer_idx in range(config.projector_config.num_hidden_layers)]

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, encoder):
        """Prepare weights from PyTorch encoder."""
        for i, layer_tt in enumerate(self.layer):
            layer_tt.prepare_weights(encoder.layer[i])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training and use_cache:
                use_cache = False

            layer_outputs = layer_module.forward(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                query_length=query_length,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Blip2QFormerModelTTNN:
    """TTNN implementation of Blip2QFormerModel."""

    def __init__(self, device, config, use_optimized_attention=True):
        self.device = device
        self.config = config

        self.encoder = Blip2QFormerEncoderTTNN(device, config, use_optimized_attention)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, model):
        """Prepare weights from PyTorch model."""
        self.layernorm_weight = ttnn.from_torch(
            model.layernorm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.layernorm_bias = ttnn.from_torch(
            model.layernorm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.encoder.prepare_weights(model.encoder)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Create extended attention mask for TTNN."""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")

        # Convert to TTNN format
        extended_attention_mask = extended_attention_mask.to(dtype=torch.bfloat16)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return ttnn.from_torch(
            extended_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def invert_attention_mask(self, encoder_attention_mask):
        """Invert attention mask for cross-attention."""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for encoder_attention_mask: {encoder_attention_mask.shape}")

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.bfloat16)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        return ttnn.from_torch(
            encoder_extended_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(
        self,
        query_embeds,
        query_length=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2] - self.config.query_length

        query_length = (
            query_length if query_length is not None else query_embeds.shape[1] if query_embeds is not None else 0
        )

        embedding_output = ttnn.layer_norm(
            query_embeds,
            weight=self.layernorm_weight,
            bias=self.layernorm_bias,
            epsilon=1e-12,
            compute_kernel_config=self.compute_config,
        )

        embedding_output_shape = tuple(embedding_output.shape)
        input_shape = embedding_output_shape[:-1]
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length))

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)

        encoder_extended_attention_mask = None
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                if isinstance(encoder_hidden_states, list):
                    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].shape
                else:
                    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length))

            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = head_mask if head_mask is not None else [None] * self.config.projector_config.num_hidden_layers

        encoder_outputs = self.encoder.forward(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = ttnn.slice(sequence_output, [0, 0, 0], [sequence_output.shape[0], 1, sequence_output.shape[2]])
        pooled_output = ttnn.reshape(pooled_output, (sequence_output.shape[0], sequence_output.shape[2]))

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class GraniteSpeechEncoderProjectorTTNN:
    """TTNN implementation of GraniteSpeechEncoderProjector."""

    def __init__(self, device, config, use_optimized_attention=True):
        self.device = device
        self.config = config
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.qformer = Blip2QFormerModelTTNN(device, config, use_optimized_attention)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, model):
        """Prepare weights from PyTorch model."""
        self.query_tt = ttnn.from_torch(model.query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.qformer.prepare_weights(model.qformer)

        self.linear_weight = ttnn.from_torch(
            model.linear.weight.transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.linear_bias = ttnn.from_torch(
            model.linear.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(self, hidden_states):
        """Forward pass through the encoder projector."""
        batch_size, seq_len, dim = hidden_states.shape

        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len

        if pad > 0:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, pad), (0, 0)], 0.0)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        hidden_states = ttnn.reshape(hidden_states, (batch_size * nblocks, self.window_size, dim))

        query_output = self.qformer.forward(
            query_embeds=self.query_tt,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=None,
            return_dict=True,
        )

        last_hidden_state = query_output.last_hidden_state
        last_hidden_state = ttnn.reshape(last_hidden_state, (batch_size, nblocks * self.num_queries, -1))

        query_proj = ttnn.linear(
            last_hidden_state, self.linear_weight, bias=self.linear_bias, compute_kernel_config=self.compute_config
        )

        return query_proj
