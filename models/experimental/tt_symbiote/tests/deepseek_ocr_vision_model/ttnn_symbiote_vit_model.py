# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math

from models.experimental.tt_symbiote.core.module import TTNNModule
from typing import Optional


########## QUICK GELU ############
class QuickGELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TTNNQuickGelu(TTNNModule):
    """TTNN-accelerated Quick Gelu activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = QuickGELU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Quick Gelu activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scaled = ttnn.multiply(input_tensor, 1.702)
        sigmoid_output = ttnn.sigmoid(scaled)
        tt_output = ttnn.multiply(input_tensor, sigmoid_output)
        ttnn.deallocate(scaled)
        ttnn.deallocate(sigmoid_output)
        return tt_output


########## CLIP Vision Embeddings ############


class TTNNClipVisionEmbeddings(TTNNModule):
    """
    CLIP Vision Embeddings using TTNN operations.

    Converts image patches to embeddings with class token and positional embeddings.
    """

    def __init__(
        self,
        old_module,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        device: ttnn.Device = None,
    ):
        """
        Initialize CLIP vision embeddings.

        Args:
            hidden_size: Embedding dimension
            image_size: Input image size
            patch_size: Patch size
            num_channels: Number of input channels
            weights: PyTorch weights dict (optional, for loading pretrained)
            device: TTNN device
        """

        super().__init__()

        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.device = device
        self.torch_layer = old_module
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

    @classmethod
    def from_torch(cls, visionEmbedding):
        """Create TTNN module from PyTorch equivalent."""
        new_clip = cls()
        new_clip._fallback_torch_layer = visionEmbedding
        return new_clip

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load from pretrained weights
        self.class_embedding = ttnn.from_torch(
            self.torch_layer.embeddings.class_embedding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Patch embedding: Conv2d weight (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = (
            self.torch_layer.embeddings.patch_embedding.weight.data
        )  # (hidden_size, 3, patch_size, patch_size)
        conv_bias = self.torch_layer.embeddings.patch_embedding.bias.data

        # Convert Conv2d to linear format for TTNN
        # Flatten kernel: (hidden_size, 3, patch_size, patch_size) -> (hidden_size, 3*patch_size*patch_size)
        linear_weight = conv_weight.view(self.embed_dim, -1)  # (hidden_size, 3*patch_size*patch_size)
        linear_weight = linear_weight.T  # (3*patch_size*patch_size, hidden_size) for TTNN linear

        self.patch_embedding_weight = ttnn.from_torch(
            linear_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if conv_bias is not None:
            self.patch_embedding_bias = self.tensor_1d_to_2d_ttnn(conv_bias)
        else:
            self.patch_embedding_bias = None

        # Position embedding - shape (num_positions, embed_dim)
        position_embedding_weight = self.torch_layer.embeddings.position_embedding.weight.data
        # Reshape to (1, num_positions, embed_dim) for get_abs_pos_ttnn
        position_embedding_reshaped = position_embedding_weight.unsqueeze(0)
        self.position_embedding = ttnn.from_torch(
            position_embedding_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.class_embedding = ttnn.to_device(self.class_embedding, self.device)
        self.patch_embedding_weight = ttnn.to_device(self.patch_embedding_weight, self.device)
        self.position_embedding = ttnn.to_device(self.position_embedding, self.device)
        if self.patch_embedding_bias is not None:
            self.patch_embedding_bias = ttnn.to_device(self.patch_embedding_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.class_embedding)
        ttnn.deallocate(self.patch_embedding_weight)
        ttnn.deallocate(self.position_embedding)
        if self.patch_embedding_bias is not None:
            ttnn.deallocate(self.patch_embedding_bias)

    def tensor_1d_to_2d_ttnn(tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """
        Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

        Args:
            tensor_1d: 1D PyTorch tensor
            device: TTNN device
            dtype: TTNN data type

        Returns:
            2D TTNN tensor of shape (1, N)
        """
        tensor_2d = tensor_1d.unsqueeze(0)
        return ttnn.from_torch(
            tensor_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _unfold_patches(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Extract patches from image using TTNN operations.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, patch_size * patch_size * channels)
        """
        batch_size = pixel_values.shape[0]
        img_h = pixel_values.shape[2]
        img_w = pixel_values.shape[3]

        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        # Reshape to extract patches: (B, C, H, W) -> (B, C, patches_h, patch_size, patches_w, patch_size)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, self.num_channels, patches_h, self.patch_size, patches_w, self.patch_size)
        )

        # Permute to group patches: (B, patches_h, patches_w, patch_size, patch_size, C)
        pixel_values = ttnn.permute(pixel_values, (0, 2, 4, 1, 3, 5))

        # Flatten patches: (B, patches_h, patches_w, patch_size * patch_size * C)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, patches_h * patches_w, self.patch_size * self.patch_size * self.num_channels)
        )

        return pixel_values

    def get_abs_pos_ttnn(
        abs_pos: ttnn.Tensor,
        tgt_size: int,
        device: ttnn.Device,
    ) -> ttnn.Tensor:
        """
        Get absolute positional embeddings, interpolating if needed.

        Args:
            abs_pos: TTNN tensor of shape (1, L, C) with positional embeddings
            tgt_size: Target sequence size (excluding CLS token)
            device: TTNN device

        Returns:
            TTNN tensor of shape (1, tgt_size + 1, C) with interpolated positional embeddings
        """
        # Convert to torch for interpolation (TTNN doesn't have bicubic interpolation)
        abs_pos_torch = ttnn.to_torch(abs_pos)

        # Extract CLS token and position embeddings
        cls_token = abs_pos_torch[:, :1, :]  # (1, 1, C)
        old_pos_embed = abs_pos_torch[:, 1:, :]  # (1, L-1, C)

        src_size = int(math.sqrt(old_pos_embed.shape[1]))
        tgt_size_sqrt = int(math.sqrt(tgt_size))

        if src_size != tgt_size_sqrt:
            # Reshape for interpolation: (1, L-1, C) -> (1, C, src_size, src_size)
            old_pos_embed_2d = old_pos_embed.view(1, src_size, src_size, -1).permute(0, 3, 1, 2).contiguous()
            old_pos_embed_2d = old_pos_embed_2d.to(torch.float32)

            # Interpolate using PyTorch
            new_pos_embed_2d = torch.nn.functional.interpolate(
                old_pos_embed_2d,
                size=(tgt_size_sqrt, tgt_size_sqrt),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).to(old_pos_embed.dtype)

            # Reshape back: (1, C, tgt_size, tgt_size) -> (1, tgt_size, C)
            new_pos_embed = new_pos_embed_2d.permute(0, 2, 3, 1).contiguous()
            new_pos_embed = new_pos_embed.view(1, tgt_size, -1)

            # Concatenate CLS token
            vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=1)  # (1, tgt_size + 1, C)
        else:
            vision_pos_embed = abs_pos_torch

        # Convert back to TTNN
        return ttnn.from_torch(
            vision_pos_embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, pixel_values: ttnn.Tensor, patch_embeds: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Forward pass of CLIP vision embeddings.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)
            patch_embeds: Optional pre-computed patch embeddings (batch_size, num_patches, embed_dim)

        Returns:
            TTNN tensor (batch_size, num_patches + 1, embed_dim)
        """
        batch_size = pixel_values.shape[0]

        # Get patch embeddings
        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            # Extract patches
            patches = self._unfold_patches(pixel_values)

            # Apply linear projection
            patch_embeds = ttnn.linear(
                patches,
                self.patch_embedding_weight,
                bias=self.patch_embedding_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(patches)

        # Expand class embedding: (embed_dim) -> (batch_size, 1, embed_dim)
        class_embeds = ttnn.reshape(self.class_embedding, (1, 1, self.embed_dim))
        class_embeds = ttnn.repeat(class_embeds, (batch_size, 1, 1))

        # Concatenate class token and patch embeddings
        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(class_embeds)
        # Note: Don't deallocate patch_embeds here - it's either passed in (user's responsibility)
        # or was just created and will be used in the concat, so it's part of embeddings now

        # Get position embeddings (with interpolation if needed)
        # Position embedding is already in shape (1, num_positions, embed_dim)
        # We need to interpolate if sequence length doesn't match
        # Note: embeddings.size(1) is the actual sequence length (num_patches + 1)
        # but get_abs_pos_ttnn expects the number of patches (excluding CLS token)
        actual_seq_len = embeddings.shape[1]  # This is num_patches + 1
        num_patches_actual = actual_seq_len - 1  # Exclude CLS token
        pos_embeds = self.get_abs_pos_ttnn(
            self.position_embedding,
            num_patches_actual,
            self.device,
        )

        # Add position embeddings
        embeddings = ttnn.add(embeddings, pos_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_embeds)

        return embeddings


########## No Tensor Parallelism Attention ############
class TTNNNoTPAttention:
    """
    No Tensor Parallelism Attention using TTNN operations.

    Implements multi-head self-attention with QKV projection and scaled dot product attention.
    """

    def __init__(
        self,
        old_module,
        cfg,
        device: ttnn.Device = None,
    ):
        """
        Initialize attention layer.

        Args:
            cfg: Configuration dict with num_attention_heads, hidden_size, etc.
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.use_flash_attention = cfg.get("use_flash_attn", False)
        self.torch_layer = old_module

    @classmethod
    def from_torch(cls, NoTPAttention):
        """Create TTNN module from PyTorch equivalent."""
        new_Attn = cls()
        new_Attn._fallback_torch_layer = NoTPAttention
        return new_Attn

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load QKV projection weights
        qkv_weight = self.torch_layer.self_attn.qkv_proj.weight.data  # (hidden_size * 3, hidden_size)
        qkv_bias = self.torch_layer.self_attn.qkv_proj.bias.data

        # Split into Q, K, V
        q_weight = qkv_weight[: self.hidden_size, :].T  # (hidden_size, hidden_size)
        k_weight = qkv_weight[self.hidden_size : 2 * self.hidden_size, :].T
        v_weight = qkv_weight[2 * self.hidden_size :, :].T

        # Convert to TTNN
        self.q_weight = ttnn.from_torch(
            q_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_weight = ttnn.from_torch(
            k_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_weight = ttnn.from_torch(
            v_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if qkv_bias is not None:
            q_bias = qkv_bias[: self.hidden_size]
            k_bias = qkv_bias[self.hidden_size : 2 * self.hidden_size]
            v_bias = qkv_bias[2 * self.hidden_size :]

            self.q_bias = self.tensor_1d_to_2d_ttnn(q_bias)
            self.k_bias = self.tensor_1d_to_2d_ttnn(k_bias)
            self.v_bias = self.tensor_1d_to_2d_ttnn(v_bias)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        # Output projection
        out_weight = self.torch_layer.self_attn.out_proj.weight.data.T  # (hidden_size, hidden_size)
        self.out_weight = ttnn.from_torch(
            out_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out_bias = self.torch_layer.self_attn.out_proj.bias.data
        if out_bias is not None:
            self.out_bias = self.tensor_1d_to_2d_ttnn(out_bias)
        else:
            self.out_bias = None

    def tensor_1d_to_2d_ttnn(tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """
        Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

        Args:
            tensor_1d: 1D PyTorch tensor
            device: TTNN device
            dtype: TTNN data type

        Returns:
            2D TTNN tensor of shape (1, N)
        """
        tensor_2d = tensor_1d.unsqueeze(0)
        return ttnn.from_torch(
            tensor_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.q_weight = ttnn.to_device(self.q_weight, self.device)
        self.k_weight = ttnn.to_device(self.k_weight, self.device)
        self.v_weight = ttnn.to_device(self.v_weight, self.device)
        self.out_weight = ttnn.to_device(self.out_weight, self.device)

        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            self.q_bias = ttnn.to_device(self.q_bias, self.device)
            self.k_bias = ttnn.to_device(self.k_bias, self.device)
            self.v_bias = ttnn.to_device(self.v_bias, self.device)
        if self.out_bias is not None:
            self.out_bias = ttnn.to_device(self.out_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""

        ttnn.deallocate(self.q_weight)
        ttnn.deallocate(self.k_weight)
        ttnn.deallocate(self.v_weight)
        ttnn.deallocate(self.out_weight)
        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            ttnn.deallocate(self.q_bias)
            ttnn.deallocate(self.k_bias)
            ttnn.deallocate(self.v_bias)
        if self.out_bias is not None:
            ttnn.deallocate(self.out_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Pre-norm attention
        residual = ttnn.layer_norm(
            x,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.layernorm_epsilon,
        )
        residual = self.self_attn.forward(residual)
        h = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        # Pre-norm feedforward
        out = ttnn.layer_norm(
            h,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.layernorm_epsilon,
        )
        out = self.mlp.forward(out)
        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)

        return out


########## No Tensor Parallelism Feed Forward ############


class TTNNNoTPFeedForward:
    """
    No Tensor Parallelism Feed Forward using TTNN operations.

    Implements two linear layers with quick_gelu activation.
    """

    def __init__(
        self,
        old_module,
        dim: int,
        hidden_dim: int,
        device: ttnn.Device = None,
    ):
        """
        Initialize feedforward layer.

        Args:
            cfg: Configuration dict
            dim: Input/output dimension
            hidden_dim: Hidden dimension
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.torch_layer = old_module

    @classmethod
    def from_torch(cls, NoTPFeedForward):
        """Create TTNN module from PyTorch equivalent."""
        new_TPFeedForward = cls()
        new_TPFeedForward._fallback_torch_layer = NoTPFeedForward
        return new_TPFeedForward

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # FC1 weights
        fc1_weight = self.torch_layer.mlp.fc1.weight.data.T  # (hidden_dim, dim)
        self.fc1_weight = ttnn.from_torch(
            fc1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        fc1_bias = self.torch_layer.mlp.fc1.bias.data
        if fc1_bias is not None:
            self.fc1_bias = self.tensor_1d_to_2d_ttnn(fc1_bias)
        else:
            self.fc1_bias = None

        # FC2 weights
        fc2_weight = self.torch_layer.mlp.fc2.weight.data.T  # (dim, hidden_dim)
        self.fc2_weight = ttnn.from_torch(
            fc2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        fc2_bias = self.torch_layer.mlp.fc2.bias.data
        if fc2_bias is not None:
            self.fc2_bias = self.tensor_1d_to_2d_ttnn(fc2_bias)
        else:
            self.fc2_bias = None

    def tensor_1d_to_2d_ttnn(tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """
        Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

        Args:
            tensor_1d: 1D PyTorch tensor
            device: TTNN device
            dtype: TTNN data type

        Returns:
            2D TTNN tensor of shape (1, N)
        """
        tensor_2d = tensor_1d.unsqueeze(0)
        return ttnn.from_torch(
            tensor_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""

        self.fc1_weight = ttnn.to_device(self.fc1_weight, self.device)
        self.fc2_weight = ttnn.to_device(self.fc2_weight, self.device)
        if self.fc1_bias is not None:
            self.fc1_bias = ttnn.to_device(self.fc1_bias, self.device)
        if self.fc2_bias is not None:
            self.fc2_bias = ttnn.to_device(self.fc2_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""

        ttnn.deallocate(self.fc1_weight)
        ttnn.deallocate(self.fc2_weight)
        if self.fc1_bias is not None:
            ttnn.deallocate(self.fc1_bias)
        if self.fc2_bias is not None:
            ttnn.deallocate(self.fc2_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of feedforward layer.

        Args:
            x: TTNN tensor (batch_size, seq_len, dim)

        Returns:
            TTNN tensor (batch_size, seq_len, dim)
        """
        # FC1
        output = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Quick GELU
        output = self.quick_gelu_ttnn(output)

        # FC2
        output = ttnn.linear(
            output,
            self.fc2_weight,
            bias=self.fc2_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return output

    def quick_gelu_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Quick GELU activation: x * sigmoid(1.702 * x)

        Args:
            x: TTNN tensor

        Returns:
            TTNN tensor with quick_gelu applied
        """
        # Compute 1.702 * x
        scaled = ttnn.multiply(x, 1.702)
        # Compute sigmoid(1.702 * x)
        sigmoid_output = ttnn.sigmoid(scaled)
        # Compute x * sigmoid(1.702 * x)
        result = ttnn.multiply(x, sigmoid_output)

        ttnn.deallocate(scaled)
        ttnn.deallocate(sigmoid_output)

        return result


########## No Tensor Parallelism Transformer Block ############


class TTNNNoTPTransformerBlock:
    """
    No Tensor Parallelism Transformer Block using TTNN operations.

    Implements pre-norm transformer block with attention and feedforward.
    """

    def __init__(
        self,
        old_module,
        cfg,
        layer_id: int,
        device: ttnn.Device = None,
    ):
        """
        Initialize transformer block.

        Args:
            cfg: Configuration dict
            layer_id: Layer index
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.layer_id = layer_id
        self.device = device
        self.hidden_size = cfg.hidden_size
        self.layernorm_epsilon = cfg.layernorm_epsilon
        self.torch_layer = old_module
        self.self_attn = TTNNNoTPAttention.from_torch()
        self.mlp = TTNNNoTPFeedForward.from_torch()

    @classmethod
    def from_torch(cls, NoTPTransformer):
        """Create TTNN module from PyTorch equivalent."""
        new_TPTransformer = cls()
        new_TPTransformer._fallback_torch_layer = NoTPTransformer
        return new_TPTransformer

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        ln1_weight = self.torch_layer.layer_norm1.weight.data
        ln1_bias = self.torch_layer.layer_norm1.bias.data
        ln2_weight = self.torch_layer.layer_norm2.weight.data
        ln2_bias = self.torch_layer.layer_norm2.bias.data

        self.layer_norm1_weight = ttnn.from_torch(
            ln1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm1_bias = ttnn.from_torch(
            ln1_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_weight = ttnn.from_torch(
            ln2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_bias = ttnn.from_torch(
            ln2_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""

        self.layer_norm1_weight = ttnn.to_device(self.layer_norm1_weight, self.device)
        self.layer_norm1_bias = ttnn.to_device(self.layer_norm1_bias, self.device)
        self.layer_norm2_weight = ttnn.to_device(self.layer_norm2_weight, self.device)
        self.layer_norm2_bias = ttnn.to_device(self.layer_norm2_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.layer_norm1_weight)
        ttnn.deallocate(self.layer_norm1_bias)
        ttnn.deallocate(self.layer_norm2_weight)
        ttnn.deallocate(self.layer_norm2_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Pre-norm attention
        residual = ttnn.layer_norm(
            x,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.layernorm_epsilon,
        )
        residual = self.self_attn.forward(residual)
        h = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        # Pre-norm feedforward
        out = ttnn.layer_norm(
            h,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.layernorm_epsilon,
        )
        out = self.mlp.forward(out)
        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)

        return out
