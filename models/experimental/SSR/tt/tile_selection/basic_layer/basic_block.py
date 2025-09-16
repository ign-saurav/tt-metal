# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule

from .swin_transformer_block import TTSwinTransformerBlock
from .patch_merging import TTPatchMerging


class TTBasicLayer(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        downsample=None,
        memory_config=None,
    ):
        super().__init__()
        self.device = device
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build transformer blocks
        self.blocks = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2

            block = TTSwinTransformerBlock(
                device=device,
                parameters=parameters["blocks"][i],
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                dtype=self.dtype,
            )
            self.blocks.append(block)

        # Optional downsampling layer
        self.has_downsample = downsample is not None
        if self.has_downsample:
            self.downsample = TTPatchMerging(
                device=device,
                parameters=parameters["downsample"],
                input_resolution=input_resolution,
                dim=dim,
                memory_config=memory_config,
            )

    def forward(self, input_tensor):
        # Process through all transformer blocks
        x = input_tensor
        for block in self.blocks:
            x = block(x)

        # Apply downsampling if present
        if self.has_downsample:
            x = self.downsample(x)

        return x
