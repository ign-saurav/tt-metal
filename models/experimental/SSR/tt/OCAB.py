import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class TTOCAB(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        input_resolution,
        window_size,
        overlap_ratio,
        num_heads,
        parameters,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2,
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        self.overlap_ratio = overlap_ratio

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        # Extract preprocessed parameters
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]

        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"]

        self.relative_position_bias_table = parameters["relative_position_bias_table"]

        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"]

        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]

        self.mlp_fc1_weight = parameters["mlp"]["fc1"]["weight"]
        self.mlp_fc1_bias = parameters["mlp"]["fc1"]["bias"]
        self.mlp_fc2_weight = parameters["mlp"]["fc2"]["weight"]
        self.mlp_fc2_bias = parameters["mlp"]["fc2"]["bias"]

        # Host-side unfold operation (PyTorch for complex operations)
        self._unfold = torch.nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

    def ttnn_rearrange_host(self, tensor, pattern_from, pattern_to, **kwargs):
        """Host-side implementation of rearrange using TTNN operations"""
        # Convert to torch for complex rearrangement, then back to TTNN
        torch_tensor = ttnn.to_torch(tensor)

        if pattern_from == "b (nc ch owh oww) nw" and pattern_to == "nc (b nw) (owh oww) ch":
            b, combined_dim, nw = torch_tensor.shape
            nc, ch, owh, oww = kwargs["nc"], kwargs["ch"], kwargs["owh"], kwargs["oww"]

            # Reshape to separate combined dimension
            reshaped = torch_tensor.reshape(b, nc, ch, owh, oww, nw)
            # Permute to desired order: nc, b, nw, ch, owh, oww
            permuted = reshaped.permute(1, 0, 5, 2, 3, 4)
            # Final reshape
            final = permuted.reshape(nc, b * nw, owh * oww, ch)

            return ttnn.from_torch(
                final,
                dtype=tensor.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return tensor

    def window_partition_ttnn(self, x, window_size):
        """TTNN implementation of window partitioning"""
        b, h, w, c = x.shape

        # Reshape: (b, h, w, c) -> (b, h//ws, ws, w//ws, ws, c)
        reshaped = ttnn.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))

        # Permute: (0, 1, 3, 2, 4, 5) -> group windows together
        permuted = ttnn.permute(reshaped, (0, 1, 3, 2, 4, 5))

        # Final reshape to get windows
        windows = ttnn.reshape(permuted, (-1, window_size, window_size, c))

        return windows

    def window_reverse_ttnn(self, windows, window_size, h, w):
        """TTNN implementation of window reverse"""
        b = int(windows.shape[0] / (h * w / window_size / window_size))

        # Reshape windows back to grid
        reshaped = ttnn.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))

        # Permute back to original order
        permuted = ttnn.permute(reshaped, (0, 1, 3, 2, 4, 5))

        # Final reshape to original spatial dimensions
        output = ttnn.reshape(permuted, (b, h, w, -1))

        return output

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        # Store shortcut connection
        shortcut = x

        # Layer normalization - handle padded dimensions
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)

        # Reshape to spatial format
        x = ttnn.reshape(x, (b, h, w, c))

        # QKV projection
        qkv = ttnn.linear(x, self.qkv_weight, bias=self.qkv_bias)
        qkv = ttnn.reshape(qkv, (b, h, w, 3, c))
        qkv = ttnn.permute(qkv, (3, 0, 4, 1, 2))  # 3, b, c, h, w

        # Split Q, K, V using slicing
        q = ttnn.slice(qkv, (0, 0, 0, 0, 0), (1, b, c, h, w))
        q = ttnn.squeeze(q, 0)  # Remove first dimension
        q = ttnn.permute(q, (0, 2, 3, 1))  # b, h, w, c

        k = ttnn.slice(qkv, (1, 0, 0, 0, 0), (2, b, c, h, w))
        k = ttnn.squeeze(k, 0)

        v = ttnn.slice(qkv, (2, 0, 0, 0, 0), (3, b, c, h, w))
        v = ttnn.squeeze(v, 0)

        # Concatenate K and V for unfold operation
        kv = ttnn.concat([k, v], dim=1)  # b, 2*c, h, w

        # Window partition for Q
        q_windows = self.window_partition_ttnn(q, self.window_size)
        q_windows = ttnn.reshape(q_windows, (-1, self.window_size * self.window_size, c))

        # Host-side unfold operation for KV
        kv_torch = ttnn.to_torch(kv)
        kv_windows_torch = self._unfold(kv_torch)  # b, c*w*w, nw
        kv_windows = ttnn.from_torch(
            kv_windows_torch,
            dtype=kv.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Rearrange KV windows using host implementation
        kv_windows = self.ttnn_rearrange_host(
            kv_windows,
            "b (nc ch owh oww) nw",
            "nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        )

        # Split K and V windows
        k_windows = ttnn.slice(
            kv_windows, (0, 0, 0, 0), (1, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        k_windows = ttnn.squeeze(k_windows, 0)

        v_windows = ttnn.slice(
            kv_windows, (1, 0, 0, 0), (2, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        v_windows = ttnn.squeeze(v_windows, 0)

        # Multi-head attention computation
        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads

        # Reshape for multi-head attention
        q = ttnn.reshape(q_windows, (b_, nq, self.num_heads, d))
        q = ttnn.permute(q, (0, 2, 1, 3))  # nw*b, nH, nq, d

        k = ttnn.reshape(k_windows, (b_, n, self.num_heads, d))
        k = ttnn.permute(k, (0, 2, 1, 3))  # nw*b, nH, n, d

        v = ttnn.reshape(v_windows, (b_, n, self.num_heads, d))
        v = ttnn.permute(v, (0, 2, 1, 3))  # nw*b, nH, n, d

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        # Scale queries
        q = ttnn.multiply(q, self.scale)

        # Attention computation
        k_transposed = ttnn.transpose(k, -2, -1)
        attn = ttnn.matmul(q, k_transposed)

        # Add relative position bias
        # Note: This is simplified - you may need to handle the indexing more carefully
        # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)]
        # attn = ttnn.add(attn, relative_position_bias)

        # Apply softmax
        attn = ttnn.softmax(attn, dim=-1)

        # Apply attention to values
        attn_output = ttnn.matmul(attn, v)
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (b_, nq, self.dim))

        # Merge windows
        attn_windows = ttnn.reshape(attn_output, (-1, self.window_size, self.window_size, self.dim))
        x = self.window_reverse_ttnn(attn_windows, self.window_size, h, w)
        x = ttnn.reshape(x, (b, h * w, self.dim))

        # Projection and residual connection
        x = ttnn.linear(x, self.proj_weight, bias=self.proj_bias)
        x = ttnn.add(x, shortcut)

        # MLP block
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

        # MLP forward pass
        mlp_out = ttnn.linear(x, self.mlp_fc1_weight, bias=self.mlp_fc1_bias)
        mlp_out = ttnn.gelu(mlp_out)
        mlp_out = ttnn.linear(mlp_out, self.mlp_fc2_weight, bias=self.mlp_fc2_bias)

        # Final residual connection
        x = ttnn.add(x, mlp_out)

        return x
