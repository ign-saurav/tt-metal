# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule

from .window_attn import TTWindowAttention
from .mlp import TTMlp


class TTSwinTransformerBlock(LightweightModule):
    def __init__(
        self,
        parameters,
        device,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=None,
    ):
        super().__init__()
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Adjust window_size and shift_size if needed
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Initialize attention module
        self.attn = TTWindowAttention(
            parameters["attn"],
            device,
            dim,
            window_size,
            num_heads,
        )

        # Initialize MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            parameters=parameters["mlp"],  # Will use parameters instead
        )

        # # Set up MLP weights from parameters
        # self.mlp.fc1_weight = parameters["mlp"]["fc1"]["weight"]
        # self.mlp.fc1_bias = parameters["mlp"]["fc1"]["bias"]
        # self.mlp.fc2_weight = parameters["mlp"]["fc2"]["weight"]
        # self.mlp.fc2_bias = parameters["mlp"]["fc2"]["bias"]

        # Pre-compute attention mask if needed
        if self.shift_size > 0:
            self.attn_mask = self._compute_attention_mask()
        else:
            self.attn_mask = None

    def _compute_attention_mask(self):
        """Compute attention mask for shifted window attention"""
        H, W = self.input_resolution

        # Create mask on CPU first
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition into windows
        mask_windows, _ = self._window_partition_padding(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Convert to TTNN tensor
        return ttnn.from_torch(attn_mask, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config)

    def _window_partition_padding(self, x, window_size):
        """Partition into non-overlapping windows with padding if needed"""
        if isinstance(x, torch.Tensor):
            B, H, W, C = x.shape

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w

            x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows, (Hp, Wp)
        else:
            # TTNN tensor case
            B, H, W, C = x.shape

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                # x = ttnn.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
                x = ttnn.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), 0.0)
            Hp, Wp = H + pad_h, W + pad_w

            x = ttnn.reshape(x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C))
            x = ttnn.permute(x, (0, 1, 3, 2, 4, 5), memory_config=self.memory_config)
            windows = ttnn.reshape(x, (-1, window_size, window_size, C))
            return windows, (Hp, Wp)

    def _window_unpartition(self, windows, window_size, pad_hw, hw):
        """Window unpartition into original sequences and removing padding"""
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)

        x = ttnn.reshape(windows, (B, Hp // window_size, Wp // window_size, window_size, window_size, -1))
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5), memory_config=self.memory_config)
        x = ttnn.reshape(x, (B, Hp, Wp, -1))

        if Hp > H or Wp > W:
            # x = ttnn.slice(x, (slice(None), slice(0, H), slice(0, W), slice(None)), memory_config=self.memory_config)
            x = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], H, W, x.shape[3]], memory_config=self.memory_config)
        return x

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # Store shortcut connection
        shortcut = x

        # Layer normalization 1
        norm1_weight = self.parameters["norm1"]["weight"]
        norm1_bias = self.parameters["norm1"]["bias"]
        # norm1_weight = ttnn.from_torch(self.parameters["norm1"]["weight"], device=self.device, memory_config=self.memory_config)
        # norm1_bias = ttnn.from_torch(self.parameters["norm1"]["bias"], device=self.device, memory_config=self.memory_config)
        x = ttnn.layer_norm(x, weight=norm1_weight, bias=norm1_bias, memory_config=self.memory_config)

        # Reshape to spatial format
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.memory_config)
        x = ttnn.reshape(x, (B, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            # TTNN doesn't have direct roll operation, so we implement it with slicing and concatenation
            shifted_x = ttnn.roll(x, [-self.shift_size, -self.shift_size], [1, 2])
        else:
            shifted_x = x

        # Partition windows
        x_windows, pad_hw = self._window_partition_padding(shifted_x, self.window_size)
        x_windows = ttnn.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        x_windows = ttnn.to_layout(x_windows, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config)

        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = ttnn.to_layout(attn_windows, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.memory_config)
        attn_windows = ttnn.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = self._window_unpartition(attn_windows, self.window_size, pad_hw, (H, W))

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = ttnn.roll(shifted_x, [self.shift_size, self.shift_size], [1, 2])
        else:
            x = shifted_x

        # Reshape back to sequence format
        x = ttnn.reshape(x, (B, H * W, C))
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config)

        # First residual connection (no drop_path implementation in TTNN)
        x = ttnn.add(shortcut, x, memory_config=self.memory_config)

        # Store for second residual connection
        residual = x

        # Layer normalization 2
        norm2_weight = self.parameters["norm2"]["weight"]
        norm2_bias = self.parameters["norm2"]["bias"]
        x = ttnn.layer_norm(x, weight=norm2_weight, bias=norm2_bias, memory_config=self.memory_config)

        # MLP
        x = self.mlp(x)

        # Second residual connection
        x = ttnn.add(residual, x, memory_config=self.memory_config)

        # Cleanup intermediate tensors
        # ttnn.deallocate(norm1_weight)
        # ttnn.deallocate(norm1_bias)
        # ttnn.deallocate(norm2_weight)
        # ttnn.deallocate(norm2_bias)

        return x
