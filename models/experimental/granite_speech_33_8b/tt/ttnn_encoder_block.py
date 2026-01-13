import ttnn
import torch
import math


class GraniteSpeechConformerFeedForwardTTNN:
    """TTNN implementation of Conformer feedforward module."""

    def __init__(self, device, config):
        self.device = device
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.hidden_dim * config.feedforward_mult
        self.dropout = config.dropout

        # Initialize weight tensors
        self.pre_norm_weight = None
        self.pre_norm_bias = None
        self.up_proj_weight = None
        self.up_proj_bias = None
        self.down_proj_weight = None
        self.down_proj_bias = None

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration for high accuracy."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(
        self, pre_norm_weight, pre_norm_bias, up_proj_weight, up_proj_bias, down_proj_weight, down_proj_bias
    ):
        """Load and convert PyTorch weights to TTNN format."""
        # LayerNorm weights and bias
        self.pre_norm_weight = ttnn.from_torch(
            pre_norm_weight.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.pre_norm_bias = ttnn.from_torch(
            pre_norm_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Linear projection weights (transpose for TTNN)
        self.up_proj_weight = ttnn.from_torch(
            up_proj_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.up_proj_bias = ttnn.from_torch(
            up_proj_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.down_proj_weight = ttnn.from_torch(
            down_proj_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.down_proj_bias = ttnn.from_torch(
            down_proj_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. LayerNorm (pre_norm)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.pre_norm_weight,
            bias=self.pre_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_config,
        )

        # 2. Linear projection (up_proj)
        hidden_states = ttnn.linear(
            hidden_states, self.up_proj_weight, bias=self.up_proj_bias, compute_kernel_config=self.compute_config
        )

        # 3. SiLU activation
        hidden_states = ttnn.silu(hidden_states)

        # 4. Linear projection (down_proj)
        hidden_states = ttnn.linear(
            hidden_states, self.down_proj_weight, bias=self.down_proj_bias, compute_kernel_config=self.compute_config
        )

        return hidden_states


class GraniteSpeechConformerAttentionTTNN:
    def __init__(self, device, config):
        self.num_heads = config.num_heads
        self.head_dim = config.dim_head
        self.device = device
        self.scale = self.head_dim**-0.5
        self.context_size = config.context_size

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration for high accuracy."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

    def prepare_weights(
        self, pre_norm_weight, pre_norm_bias, to_q_weight, to_kv_weight, to_out_weight, to_out_bias, rel_pos_emb_weight
    ):
        """Load and convert PyTorch weights to TTNN format."""
        # LayerNorm weights and bias
        self.pre_norm_weight = ttnn.from_torch(
            pre_norm_weight.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.pre_norm_bias = ttnn.from_torch(
            pre_norm_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Linear projection weights (transpose for TTNN)
        self.to_q_weight = ttnn.from_torch(
            to_q_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.to_kv_weight = ttnn.from_torch(
            to_kv_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.to_out_weight = ttnn.from_torch(
            to_out_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.to_out_bias = ttnn.from_torch(
            to_out_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.rel_pos_emb_weight = rel_pos_emb_weight

    def forward(self, hidden_states, dist):
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.pre_norm_weight,
            bias=self.pre_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_config,
        )
        bsz, num_features, _ = hidden_states.shape

        num_blocks = math.ceil(num_features / self.context_size)
        remainder = num_features % self.context_size
        if remainder > 0:
            # right padding to reach block size
            pad_size = self.context_size - remainder
            hidden_states = ttnn.pad(hidden_states, padding=[(0, 0), (0, pad_size), (0, 0)], value=0)

        # Apply linear projections separately to avoid concat issues
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        query_states = ttnn.linear(
            hidden_states,
            self.to_q_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        kv_states = ttnn.linear(
            hidden_states,
            self.to_kv_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        key_states, value_states = ttnn.chunk(kv_states, 2, dim=-1)
        ttnn.deallocate(kv_states)

        # Reshape each tensor separately for multi-head attention
        query_states = ttnn.reshape(query_states, (bsz * num_blocks, self.context_size, self.num_heads, -1))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key_states = ttnn.reshape(key_states, (bsz * num_blocks, self.context_size, self.num_heads, -1))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value_states = ttnn.reshape(value_states, (bsz * num_blocks, self.context_size, self.num_heads, -1))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Shaw's relative positional embedding
        rel_pos_emb = torch.nn.functional.embedding(dist, weight=self.rel_pos_emb_weight)
        composer = ttnn.concat_mesh_to_tensor_composer(self.device, dim=1)
        query_states_torch = ttnn.to_torch(query_states, mesh_composer=composer)
        query_states_torch = query_states_torch.reshape(
            num_blocks, query_states_torch.shape[-3], query_states_torch.shape[-2], query_states_torch.shape[-1]
        )
        pos_attn = torch.einsum("m h c d, c r d -> m h c r", query_states_torch, rel_pos_emb) * self.scale

        if remainder > 0:
            # masked attention in the extended block
            mask = torch.ones((self.context_size, self.context_size), dtype=bool)
            mask[:remainder, :remainder] = 0
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)
            pos_attn_sdpa = pos_attn[:, :1, :, :]

        ttnn_mask = ttnn.from_torch(pos_attn_sdpa, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Pad sequence length to multiple of 32 for SDPA
        if query_states.shape[2] % 32 != 0:
            pad_size = 32 - (query_states.shape[2] % 32)
            query_states = ttnn.pad(query_states, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)
            ttnn_mask = ttnn.pad(ttnn_mask, padding=[(0, 0), (0, 0), (0, pad_size), (0, pad_size)], value=0)
            key_states = ttnn.pad(key_states, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)
            value_states = ttnn.pad(value_states, padding=[(0, 0), (0, 0), (0, pad_size), (0, 0)], value=0)
        context = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=False,
            scale=self.scale,
            attn_mask=ttnn_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(query_states)
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        # Concatenate heads back to original format
        context = ttnn.transformer.concatenate_heads(
            context[:, :, : self.context_size, :], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        context = ttnn.reshape(context, (bsz, num_blocks * self.context_size, -1))

        context = context[:, :num_features, :]

        # Output projection
        output = ttnn.linear(
            context,
            self.to_out_weight,
            bias=self.to_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(context)

        return output


class GraniteSpeechConformerConvModuleTTNN:
    """TTNN implementation of Conformer convolution module."""

    def __init__(self, device, config):
        self.device = device
        self.hidden_dim = config.hidden_dim
        self.inner_dim = config.hidden_dim * config.conv_expansion_factor
        self.conv_kernel_size = config.conv_kernel_size
        self.dropout = config.dropout

        # Initialize weight tensors
        self.norm_weight = None
        self.norm_bias = None
        self.up_conv_weight = None
        self.down_conv_weight = None
        self.batch_norm_weight = None
        self.batch_norm_bias = None
        self.batch_norm_running_mean = None
        self.batch_norm_running_var = None

        # Initialize depthwise conv module
        self.depth_conv = GraniteSpeechConformerDepthWiseConv1dTTNN(
            device, self.inner_dim, self.inner_dim, self.conv_kernel_size
        )
        self.dropout = torch.nn.Dropout(config.dropout)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(
        self,
        norm_weight,
        norm_bias,
        up_conv_weight,
        down_conv_weight,
        batch_norm_weight,
        batch_norm_bias,
        batch_norm_running_mean,
        batch_norm_running_var,
        depth_conv_weights,
    ):
        """Load and convert PyTorch weights to TTNN format."""
        # LayerNorm weights and bias
        self.norm_weight = ttnn.from_torch(
            norm_weight.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.norm_bias = ttnn.from_torch(
            norm_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Conv1d weights (transpose for TTNN)
        self.up_conv_weight = ttnn.from_torch(
            up_conv_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        self.down_conv_weight = ttnn.from_torch(
            down_conv_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        # BatchNorm parameters
        self.batch_norm_weight = ttnn.from_torch(
            batch_norm_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.batch_norm_bias = ttnn.from_torch(
            batch_norm_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.batch_norm_running_mean = ttnn.from_torch(
            batch_norm_running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.batch_norm_running_var = ttnn.from_torch(
            batch_norm_running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Prepare depthwise conv weights using your class method
        self.depth_conv.prepare_weights(depth_conv_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. LayerNorm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_config,
        )

        # 2. Up Conv1d (permute to NLC format for TTNN)
        hidden_states_nlc = hidden_states
        bsz, seq_len, hidden_dim = hidden_states_nlc.shape

        # Configure conv1d
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=True,
        )

        # Up projection
        [up_output, out_length, _] = ttnn.conv1d(
            input_tensor=hidden_states_nlc,
            weight_tensor=self.up_conv_weight,
            in_channels=self.hidden_dim,
            out_channels=self.inner_dim * 2,
            device=self.device,
            bias_tensor=None,
            kernel_size=1,
            stride=1,
            padding=0,
            batch_size=bsz,
            input_length=seq_len,
            conv_config=conv_config,
            compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 3. GLU activation (splits tensor in half along last dimension)
        glu_output = ttnn.glu(up_output, dim=-1)
        glu_output = glu_output[:, :, :seq_len, :]
        glu_output = ttnn.reshape(glu_output, (bsz, out_length, self.inner_dim))

        # 4. Depthwise Convolution
        depth_output = self.depth_conv.forward(glu_output)

        # 5. BatchNorm1d + SiLU
        # BatchNorm expects NCL format
        batch_size, channels, seq_len = depth_output.shape

        # Reshape to 4D for batch_norm: [batch, channels, seq_len, 1]
        depth_output_4d = ttnn.reshape(depth_output, (batch_size, channels, seq_len, 1))

        # Apply batch_norm
        batch_norm_output = ttnn.batch_norm(
            input=depth_output_4d,
            running_mean=self.batch_norm_running_mean,
            running_var=self.batch_norm_running_var,
            training=False,
            weight=self.batch_norm_weight,
            bias=self.batch_norm_bias,
            eps=1e-05,
            compute_kernel_config=self.compute_config,
        )

        # Reshape back to 3D: [batch, seq_len, channels]
        batch_norm_output = ttnn.reshape(batch_norm_output, (batch_size, channels, seq_len))
        batch_norm_output = ttnn.permute(batch_norm_output, (0, 2, 1))

        # SiLU activation
        silu_output = ttnn.silu(batch_norm_output)

        # 6. Down Conv1d
        [down_output, _, _] = ttnn.conv1d(
            input_tensor=silu_output,
            weight_tensor=self.down_conv_weight,
            in_channels=self.inner_dim,
            out_channels=self.hidden_dim,
            device=self.device,
            bias_tensor=None,
            kernel_size=1,
            stride=1,
            padding=0,
            batch_size=bsz,
            input_length=out_length,
            conv_config=conv_config,
            compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        down_output = ttnn.reshape(down_output, (bsz, out_length, self.hidden_dim))

        return down_output


class GraniteSpeechConformerDepthWiseConv1dTTNN:
    """TTNN implementation of padded 1D depthwise convolution."""

    def __init__(self, device, chan_in: int, chan_out: int, kernel_size: int):
        self.device = device
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size

        # Calculate padding same as PyTorch version
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = [pad, pad - pad_offset]

        self.weight_tensor = None
        self._setup_conv_config()

    def _setup_conv_config(self):
        """Setup TTNN convolution configuration."""
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, torch_weights):
        """Load and convert PyTorch weights to TTNN format."""
        # PyTorch weights: [out_channels, in_channels, kernel_size]
        # For depthwise: [chan_out, 1, kernel_size] with groups=chan_in
        self.weight_tensor = ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = hidden_states.shape

        # Call TTNN conv1d
        [tt_output_tensor, out_length, [weights_device, _]] = ttnn.conv1d(
            input_tensor=hidden_states,
            weight_tensor=self.weight_tensor,
            in_channels=self.chan_in,
            out_channels=self.chan_out,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            batch_size=batch_size,
            input_length=length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.chan_in,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.reshape(tt_output_tensor, (batch_size, out_length, self.chan_out))
        tt_output_tensor = ttnn.to_device(tt_output_tensor, self.device)
        tt_output_tensor = ttnn.permute(tt_output_tensor, (0, 2, 1))
        return tt_output_tensor


class GraniteSpeechConformerBlockTTNN:
    """TTNN implementation of conformer block."""

    def __init__(self, device, config, include_layernorm):
        self.device = device
        self.config = config
        self.include_layernorm = include_layernorm

        # Initialize sub-modules
        self.ff1 = GraniteSpeechConformerFeedForwardTTNN(device, config)
        self.attn = GraniteSpeechConformerAttentionTTNN(device, config)
        self.conv = GraniteSpeechConformerConvModuleTTNN(device, config)
        self.ff2 = GraniteSpeechConformerFeedForwardTTNN(device, config)

        # Post norm
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if self.include_layernorm:
            self.layernorm_compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            self.layernorm_config = ttnn.LayerNormDefaultProgramConfig()

    def prepare_weights(self, ff1, ff2, attn, conv, post_norm):
        """Prepare all weights for sub-modules."""
        # FeedForward weights
        self.ff1.prepare_weights(
            ff1.pre_norm.weight,
            ff1.pre_norm.bias,
            ff1.up_proj.weight,
            ff1.up_proj.bias,
            ff1.down_proj.weight,
            ff1.down_proj.bias,
        )
        self.ff2.prepare_weights(
            ff2.pre_norm.weight,
            ff2.pre_norm.bias,
            ff2.up_proj.weight,
            ff2.up_proj.bias,
            ff2.down_proj.weight,
            ff2.down_proj.bias,
        )

        # Attention weights
        self.attn.prepare_weights(
            attn.pre_norm.weight,
            attn.pre_norm.bias,
            attn.to_q.weight,
            attn.to_kv.weight,
            attn.to_out.weight,
            attn.to_out.bias,
            attn.rel_pos_emb.weight,
        )

        # Conv module weights
        self.conv.prepare_weights(
            conv.norm.weight,
            conv.norm.bias,
            conv.up_conv.weight,
            conv.down_conv.weight,
            conv.batch_norm.weight,
            conv.batch_norm.bias,
            conv.batch_norm.running_mean,
            conv.batch_norm.running_var,
            conv.depth_conv.conv.weight,
        )

        # Prepare layer norm weights
        if self.include_layernorm:
            self.weight_tensor = ttnn.from_torch(
                post_norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            self.bias_tensor = ttnn.from_torch(
                post_norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

    def forward(self, hidden_states: ttnn.Tensor, attention_dists: torch.Tensor) -> ttnn.Tensor:
        # FF1 with residual
        ff1_out = self.ff1.forward(hidden_states)
        hidden_states = ttnn.add(ttnn.multiply(ff1_out, 0.5), hidden_states)

        # Attention with residual
        attn_out = self.attn.forward(hidden_states, attention_dists)
        hidden_states = ttnn.add(attn_out, hidden_states)

        # Conv module with residual
        conv_out = self.conv.forward(hidden_states)
        hidden_states = ttnn.add(conv_out, hidden_states)

        # FF2 with residual
        ff2_out = self.ff2.forward(hidden_states)
        hidden_states = ttnn.add(ttnn.multiply(ff2_out, 0.5), hidden_states)

        # Post norm
        if self.include_layernorm:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.weight_tensor,
                bias=self.bias_tensor,
                epsilon=1e-5,
                compute_kernel_config=self.layernorm_compute_config,
                program_config=self.layernorm_config,
            )

        return hidden_states


class GraniteSpeechCTCEncoderTTNN:
    """TTNN implementation of CTC Encoder."""

    def __init__(self, device, config, include_conformer_layernorm):
        self.device = device
        self.config = config
        self.num_layers = config.num_layers
        self.include_conformer_layernorm = include_conformer_layernorm

        # Precompute attention distances
        seq = torch.arange(config.context_size)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        self.attention_dists = torch.clamp(relpos_dist, -config.context_size, config.context_size) + config.max_pos_emb

        # Initialize conformer blocks
        self.layers = [
            GraniteSpeechConformerBlockTTNN(device, config, self.include_conformer_layernorm)
            for _ in range(config.num_layers)
        ]

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare_weights(self, encoder):
        """Prepare and load all weights from PyTorch encoder model."""
        # Input linear layer weights
        self.input_weight = ttnn.from_torch(
            encoder.input_linear.weight.transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.input_bias = ttnn.from_torch(
            encoder.input_linear.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Output linear layer weights
        self.out_weight = ttnn.from_torch(
            encoder.out.weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.out_bias = ttnn.from_torch(
            encoder.out.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Mid linear layer weights
        self.out_mid_weight = ttnn.from_torch(
            encoder.out_mid.weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.out_mid_bias = ttnn.from_torch(
            encoder.out_mid.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Prepare conformer block weights
        for i, layer in enumerate(self.layers):
            torch_layer = encoder.layers[i]
            layer.prepare_weights(
                torch_layer.ff1, torch_layer.ff2, torch_layer.attn, torch_layer.conv, torch_layer.post_norm
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with mid-layer residual connection."""
        batch_size, seq_len, input_dim = hidden_states.shape

        # Input linear projection
        tt_hidden_states = ttnn.linear(
            hidden_states, self.input_weight, bias=self.input_bias, compute_kernel_config=self.compute_config
        )

        # Process through conformer blocks
        for idx, layer in enumerate(self.layers, start=1):
            tt_hidden_states = layer.forward(tt_hidden_states, self.attention_dists)

            # Mid-layer residual connection
            if idx == self.num_layers // 2:
                tt_hidden_states_mid = ttnn.clone(tt_hidden_states)

                # Apply output projection
                tt_mid_out = ttnn.linear(
                    tt_hidden_states_mid, self.out_weight, bias=self.out_bias, compute_kernel_config=self.compute_config
                )

                tt_softmax_out = ttnn.softmax(tt_mid_out, dim=-1)

                # Apply mid projection
                tt_mid_residual = ttnn.linear(
                    tt_softmax_out,
                    self.out_mid_weight,
                    bias=self.out_mid_bias,
                    compute_kernel_config=self.compute_config,
                )

                # Add residual connection
                tt_hidden_states = ttnn.add(tt_hidden_states, tt_mid_residual)

        return tt_hidden_states
