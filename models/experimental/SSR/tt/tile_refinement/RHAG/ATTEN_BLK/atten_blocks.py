# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from .HAB import TTHAB
from .OCAB import TTOCAB


class TTAttenBlocks(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        compress_ratio,
        squeeze_factor,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
        memory_config=None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.dtype = dtype

        # Build HAB blocks
        self.blocks = []
        for i in range(depth):
            # Calculate shift size: 0 for even indices, window_size // 2 for odd indices
            shift_size = 0 if (i % 2 == 0) else window_size // 2

            # Handle drop_path - can be a list or single value
            if isinstance(drop_path, list):
                current_drop_path = drop_path[i]
            else:
                current_drop_path = drop_path

            hab_block = TTHAB(
                device=device,
                parameters=parameters["blocks"][i],
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                memory_config=memory_config,
                dtype=dtype,
            )
            self.blocks.append(hab_block)

        # OCAB (Overlapping Cross Attention Block)
        self.overlap_attn = TTOCAB(
            device=device,
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            parameters=parameters["overlap_attn"],
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
        )

        self.downsample = None
        if downsample is not None:
            self.downsample = downsample

    def forward(self, x, x_size, params):
        """
        Forward pass through all attention blocks

        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            x_size: Tuple of (height, width) for spatial dimensions
            params: Dictionary containing:
                - "rpi_sa": Relative position index for self-attention
                - "attn_mask": Attention mask for shifted windows
                - "rpi_oca": Relative position index for overlapping cross attention
        """
        # Process through all HAB blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])

        # Apply overlapping cross attention
        x = self.overlap_attn(x, x_size, params["rpi_oca"])

        # Apply downsampling if present
        if self.downsample is not None:
            x = self.downsample(x)

        return x
