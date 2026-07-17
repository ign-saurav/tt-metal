# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.swin2sr.tt.tt_swin_transformer_block import TtSwinTransformerBlock


class TtBasicLayer:
    def __init__(
        self,
        device,
        parameters,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.memory_config = memory_config

        self.blocks = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block_params = parameters["blocks"][i] if "blocks" in parameters else parameters[f"blocks.{i}"]

            block = TtSwinTransformerBlock(
                device=device,
                parameters=block_params,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                memory_config=memory_config,
            )
            self.blocks.append(block)

        self.downsample = None

    def __call__(self, x: ttnn.Tensor, x_size: tuple[int, int]) -> ttnn.Tensor:
        for block in self.blocks:
            x = block(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
