# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.swin2sr.tt.tt_window_attention import TtSwin2SRWindowAttention
from models.experimental.swin2sr.tt.tt_mlp import TtSwin2SRMLP


def window_partition_ttnn(
    x: ttnn.Tensor, window_size: int, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> tuple[ttnn.Tensor, tuple[int, int]]:
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = ttnn.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), value=0.0, memory_config=memory_config)
    Hp, Wp = H + pad_h, W + pad_w

    x = ttnn.reshape(
        x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C), memory_config=memory_config
    )
    windows = ttnn.permute(x, (0, 1, 3, 2, 4, 5), memory_config=memory_config)
    windows = ttnn.reshape(windows, (-1, window_size, window_size, C), memory_config=memory_config)
    ttnn.deallocate(x)

    return windows, (Hp, Wp)


def window_reverse_ttnn(
    windows: ttnn.Tensor, window_size: int, H: int, W: int, Hp: int, Wp: int, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    C = windows.shape[-1]

    windows = ttnn.reshape(
        windows, (B, Hp // window_size, Wp // window_size, window_size, window_size, C), memory_config=memory_config
    )
    x = ttnn.permute(windows, (0, 1, 3, 2, 4, 5), memory_config=memory_config)
    ttnn.deallocate(windows)
    x = ttnn.reshape(x, (B, Hp, Wp, C), memory_config=memory_config)

    if Hp > H or Wp > W:
        x = ttnn.slice(x, [0, 0, 0, 0], [B, H, W, C], memory_config=memory_config)

    return x


class TtSwinTransformerBlock:
    def __init__(
        self,
        device,
        parameters,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.memory_config = memory_config
        self.l1 = ttnn.L1_MEMORY_CONFIG
        self.dram = ttnn.DRAM_MEMORY_CONFIG

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.attn = TtSwin2SRWindowAttention(
            device=device,
            parameters=parameters["attn"],
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            memory_config=memory_config,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)

        class ParamNamespace:
            pass

        mlp_params = ParamNamespace()
        mlp_params.fc1 = ParamNamespace()
        mlp_params.fc2 = ParamNamespace()
        mlp_params.fc1.weight = parameters["mlp"]["fc1"]["weight"]
        mlp_params.fc1.bias = parameters["mlp"]["fc1"].get("bias", None)
        mlp_params.fc2.weight = parameters["mlp"]["fc2"]["weight"]
        mlp_params.fc2.bias = parameters["mlp"]["fc2"].get("bias", None)

        self.mlp = TtSwin2SRMLP(
            device=device,
            parameters=mlp_params,
            activation="gelu",
            memory_config=memory_config,
        )

        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(self.input_resolution, Hp=None, Wp=None)
        else:
            self.attn_mask = None

    def calculate_mask(self, x_size: tuple[int, int], Hp: int = None, Wp: int = None) -> ttnn.Tensor:
        H, W = x_size
        if Hp is None:
            Hp = H
        if Wp is None:
            Wp = W
        img_mask = torch.zeros((1, Hp, Wp, 1))
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

        mask_windows = img_mask.view(
            1, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, 1
        )
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return ttnn.from_torch(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, x_size: tuple[int, int]) -> ttnn.Tensor:
        H, W = x_size
        B, L, C = x.shape
        if L < H * W:
            raise ValueError(f"Input sequence length {L} is less than H*W = {H*W}")
        shortcut = ttnn.reallocate(x, memory_config=self.dram)

        tensor_size = B * H * W * C
        reshape_memory_config = self.l1 if tensor_size < 1_000_000 else self.dram
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=reshape_memory_config, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, (B, H, W, C), memory_config=reshape_memory_config)

        if self.shift_size > 0:
            shifted_x = ttnn.roll(x, [-self.shift_size, -self.shift_size], [1, 2])
            shifted_x = ttnn.to_memory_config(shifted_x, reshape_memory_config)
        else:
            shifted_x = x

        x_windows, (Hp, Wp) = window_partition_ttnn(shifted_x, self.window_size, memory_config=self.dram)
        ttnn.deallocate(shifted_x)
        windows_size = x_windows.shape[0] * self.window_size * self.window_size * C * 2
        windows_memory_config = self.l1 if windows_size < 1_000_000 else self.dram
        x_windows = ttnn.reshape(
            x_windows, (-1, self.window_size * self.window_size, C), memory_config=windows_memory_config
        )
        x_windows = ttnn.to_layout(
            x_windows, layout=ttnn.TILE_LAYOUT, memory_config=windows_memory_config, dtype=ttnn.bfloat16
        )

        if self.shift_size > 0:
            if self.input_resolution == x_size and Hp == H and Wp == W:
                if self.attn_mask is None:
                    self.attn_mask = self.calculate_mask(x_size, Hp=Hp, Wp=Wp)
                attn_mask = self.attn_mask
            else:
                attn_mask = self.calculate_mask(x_size, Hp=Hp, Wp=Wp)
        else:
            attn_mask = None

        attn_windows = self.attn(x_windows, mask=attn_mask)
        ttnn.deallocate(x_windows)

        attn_windows = ttnn.to_layout(
            attn_windows, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=self.dram, dtype=ttnn.bfloat16
        )
        attn_windows = ttnn.reshape(attn_windows, (-1, self.window_size, self.window_size, C), memory_config=self.dram)
        shifted_x = window_reverse_ttnn(attn_windows, self.window_size, H, W, Hp, Wp, memory_config=self.dram)
        ttnn.deallocate(attn_windows)

        if self.shift_size > 0:
            x = ttnn.roll(shifted_x, [self.shift_size, self.shift_size], [1, 2])
            x = ttnn.to_memory_config(x, reshape_memory_config)
        else:
            x = shifted_x

        x = ttnn.reshape(x, (B, H * W, C), memory_config=reshape_memory_config)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, memory_config=reshape_memory_config, dtype=ttnn.bfloat16)

        norm1_weight = self.parameters["norm1"]["weight"]
        norm1_bias = self.parameters["norm1"]["bias"]
        x = ttnn.layer_norm(x, weight=norm1_weight, bias=norm1_bias, memory_config=self.l1)

        x = ttnn.add(shortcut, x, memory_config=self.l1, dtype=ttnn.bfloat16)
        ttnn.deallocate(shortcut)

        residual = ttnn.reallocate(x, memory_config=self.dram)

        x = self.mlp(x)

        norm2_weight = self.parameters["norm2"]["weight"]
        norm2_bias = self.parameters["norm2"]["bias"]
        x = ttnn.layer_norm(x, weight=norm2_weight, bias=norm2_bias, memory_config=self.l1)

        x = ttnn.add(residual, x, memory_config=self.l1, dtype=ttnn.bfloat16)
        ttnn.deallocate(residual)

        return x
