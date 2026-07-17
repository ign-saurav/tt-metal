# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule

from .CAB import TTCAB
from models.experimental.SSR.tt.common.mlp import TTMlp
from .window_attn_tr import TTWindowAttentionTR


class TTHAB(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dtype = dtype

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # Extract preprocessed parameters
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]
        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]
        self.conv_scale = parameters.get("conv_scale", 0.01)

        # Initialize sub-modules
        self.attn = TTWindowAttentionTR(
            device=device,
            parameters=parameters["attn"],
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            memory_config=memory_config,
            dtype=dtype,
        )

        self.conv_block = TTCAB(
            device=device, parameters=parameters["conv_block"], num_feat=dim, memory_config=memory_config, dtype=dtype
        )

        self.mlp = TTMlp(
            device=device,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            parameters=parameters["mlp"],
            dtype=dtype,
        )

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, seq_len, c = x.shape
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=self.dtype)
        shortcut = x

        # Layer norm 1
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)

        # Reshape to spatial format for conv and attention
        x = ttnn.reshape(x, [b, h, w, c])

        # Convolutional branch
        conv_x = self.conv_block(x)
        conv_x = ttnn.reshape(conv_x, [b, h * w, c])
        conv_x = ttnn.multiply(conv_x, self.conv_scale, dtype=self.dtype)

        # Attention branch - handle cyclic shifttt-metal
        if self.shift_size > 0:
            # Cyclic shift
            shifted_x = ttnn.roll(x, [-self.shift_size, -self.shift_size], [1, 2])
            current_attn_mask = attn_mask
        else:
            shifted_x = x
            current_attn_mask = None

        # Window partition
        if shifted_x.memory_config().buffer_type != ttnn.BufferType.L1:
            shifted_x = ttnn.to_memory_config(shifted_x, self.memory_config, dtype=self.dtype)
        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = ttnn.reshape(x_windows, [-1, self.window_size * self.window_size, c])

        # Window attention
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=current_attn_mask)

        # Window reverse
        attn_windows = ttnn.reshape(attn_windows, [-1, self.window_size, self.window_size, c])
        shifted_x = self._window_reverse(attn_windows, self.window_size, h, w)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = ttnn.roll(shifted_x, [self.shift_size, self.shift_size], [1, 2])
        else:
            attn_x = shifted_x

        if attn_x.memory_config().buffer_type != ttnn.BufferType.L1:
            attn_x = ttnn.to_memory_config(attn_x, ttnn.L1_MEMORY_CONFIG, dtype=self.dtype)
        attn_x = ttnn.reshape(attn_x, [b, h * w, c])

        # First residual connection
        x = ttnn.add(shortcut, attn_x, dtype=self.dtype)
        x = ttnn.add(x, conv_x, dtype=self.dtype)

        # MLP branch
        x_norm = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)
        mlp_out = self.mlp(x_norm)

        # Second residual connection
        x = ttnn.add(x, mlp_out)

        return x

    def _window_partition(self, x, window_size):
        """Partition into non-overlapping windows"""
        B, H, W, C = x.shape
        num_windows = (H // window_size) * (W // window_size)
        return ttnn.reshape(x, [B * num_windows, window_size, window_size, C], memory_config=self.memory_config)

    def _window_reverse(self, windows, window_size, H, W):
        B = windows.shape[0] // (H * W // window_size // window_size)
        return ttnn.reshape(windows, [B, H, W, -1], memory_config=self.memory_config)
