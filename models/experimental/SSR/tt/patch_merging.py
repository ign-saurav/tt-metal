# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
import torch
from models.demos.deepseek_v3.utils.config_helpers import matmul_config


class TTPatchMerging(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        input_resolution,
        dim,
        memory_config=None,
    ):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.input_resolution = input_resolution
        self.dim = dim

        # Extract weights from preprocessed parameters
        self.reduction_weight = parameters["reduction"]["weight"]
        self.norm_weight = parameters["norm"]["weight"]
        self.norm_bias = parameters["norm"]["bias"]

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: TTNN tensor with shape [B, H*W, C]
        Returns:
            TTNN tensor with shape [B, H/2*W/2, 2*C]
        """
        H, W = self.input_resolution
        B, L, C = input_tensor.shape

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # Reshape to spatial dimensions [B, H, W, C]
        input_tensor = ttnn.reshape(input_tensor, (B, H, W, C))
        x = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        kernel_top_left = torch.zeros(C, 1, 2, 2, dtype=torch.bfloat16)
        kernel_top_left[:, 0, 0, 0] = 1.0

        kernel_bottom_left = torch.zeros(C, 1, 2, 2, dtype=torch.bfloat16)
        kernel_bottom_left[:, 0, 1, 0] = 1.0

        kernel_top_right = torch.zeros(C, 1, 2, 2, dtype=torch.bfloat16)
        kernel_top_right[:, 0, 0, 1] = 1.0

        kernel_bottom_right = torch.zeros(C, 1, 2, 2, dtype=torch.bfloat16)
        kernel_bottom_right[:, 0, 1, 1] = 1.0

        # Convert to TTNN tensors
        tt_kernel_top_left = ttnn.from_torch(kernel_top_left, device=self.device)
        tt_kernel_bottom_left = ttnn.from_torch(kernel_bottom_left, device=self.device)
        tt_kernel_top_right = ttnn.from_torch(kernel_top_right, device=self.device)
        tt_kernel_bottom_right = ttnn.from_torch(kernel_bottom_right, device=self.device)

        # Apply grouped convolutions for each patch
        x0 = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=tt_kernel_top_left,
            in_channels=C,
            out_channels=C,
            device=self.device,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            groups=C,  # Grouped convolution
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=None,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x1 = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=tt_kernel_bottom_left,
            in_channels=C,
            out_channels=C,
            device=self.device,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            groups=C,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=None,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x2 = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=tt_kernel_top_right,
            in_channels=C,
            out_channels=C,
            device=self.device,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            groups=C,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=None,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x3 = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=tt_kernel_bottom_right,
            in_channels=C,
            out_channels=C,
            device=self.device,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            groups=C,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=None,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(x)

        # Concatenate along channel dimension [B, H/2, W/2, 4*C]
        merged = ttnn.concat([x0, x1, x2, x3], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Clean up intermediate tensors
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)

        # Reshape to sequence format [B, H/2*W/2, 4*C]
        merged = ttnn.reshape(merged, (B, (H // 2) * (W // 2), 4 * C), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Apply layer normalization
        normalized = ttnn.layer_norm(
            merged,
            weight=self.norm_weight,
            bias=self.norm_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(merged)

        # Apply linear reduction [B, H/2*W/2, 2*C]
        output_matmul_config = matmul_config(
            normalized.shape[-2], normalized.shape[-1], self.reduction_weight.shape[-2], (8, 8)
        )
        output = ttnn.linear(
            normalized, self.reduction_weight, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=output_matmul_config
        )
        ttnn.deallocate(normalized)

        return output
