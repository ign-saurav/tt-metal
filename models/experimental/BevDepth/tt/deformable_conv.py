# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN wrapper for Deformable Convolution.

NOTE: Native TTNN implementation of deformable convolution is not yet available.
TODO: Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/25526>

This module wraps the reference DeformConv2dPack implementation for use with TTNN tensors.
When native TTNN support is added, this module should be updated to use the native ops.
"""

import torch
import ttnn
from typing import Tuple, Optional
import logging

# Import the reference implementation
from models.experimental.BevDepth.reference.bevdepth.layers.heads.deform_conv import (
    DeformConv2d,
    DeformConv2dPack,
    DCN,
    _deform_conv2d_torchvision,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "DeformConv2d",
    "DeformConv2dPack",
    "DCN",
    "TtDeformConv2dPack",
    "_deform_conv2d_torchvision",
]


class TtDeformConv2dPack:
    """
    TTNN-compatible wrapper for DeformConv2dPack.

    This wraps the reference DeformConv2dPack for use with TTNN tensors,
    handling the conversion between TTNN (NHWC) and PyTorch (NCHW) formats.

    NOTE: This is a FALLBACK implementation using torchvision.
    TODO: Native TTNN implementation pending - https://github.com/tenstorrent/tt-metal/issues/25526

    Args:
        device: TTNN device
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride (default: 1)
        padding: Convolution padding (default: 0)
        dilation: Convolution dilation (default: 1)
        groups: Number of groups (default: 1)
        deform_groups: Number of deformable groups (default: 1)
        conv_offset_weight: Pre-trained offset conv weights (optional)
        conv_offset_bias: Pre-trained offset conv bias (optional)
        dcn_weight: Pre-trained DCN weights (optional)
        dcn_bias: Pre-trained DCN bias (optional)
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        conv_offset_weight: Optional[torch.Tensor] = None,
        conv_offset_bias: Optional[torch.Tensor] = None,
        dcn_weight: Optional[torch.Tensor] = None,
        dcn_bias: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create the reference DeformConv2dPack
        # Note: DeformConv2d base class doesn't support bias, but we handle it separately
        self.dcn = DeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False,  # DeformConv2d doesn't support bias in weight
        )

        # Load pre-trained weights if provided
        if dcn_weight is not None:
            self.dcn.weight.data = dcn_weight.float()

        if conv_offset_weight is not None:
            self.dcn.conv_offset.weight.data = conv_offset_weight.float()

        if conv_offset_bias is not None:
            self.dcn.conv_offset.bias.data = conv_offset_bias.float()

        # Store bias separately (applied after deform conv)
        self.bias = dcn_bias.float() if dcn_bias is not None else None

        # Set to eval mode
        self.dcn.eval()

        logger.info(
            f"TtDeformConv2dPack initialized: in={in_channels}, out={out_channels}, "
            f"kernel={kernel_size}, stride={stride}, padding={padding}, "
            f"deform_groups={deform_groups}, has_bias={self.bias is not None}"
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> Tuple[ttnn.Tensor, int, int]:
        """
        Forward pass with automatic offset generation.

        Args:
            x: Input TTNN tensor in NHWC format [B, H, W, C] or flattened format
            batch_size: Batch size
            height: Input height
            width: Input width

        Returns:
            Tuple of (output_tensor, output_height, output_width)
            output_tensor is in NHWC format [B, H_out, W_out, C_out]
        """
        # Convert TTNN tensor to PyTorch
        x_torch = ttnn.to_torch(x)

        # Handle various TTNN tensor formats -> [B, H, W, C]
        if len(x_torch.shape) == 4:
            if x_torch.shape[1] == 1 and x_torch.shape[2] == height * width:
                # Flattened: [B, 1, H*W, C] -> [B, H, W, C]
                x_torch = x_torch.reshape(batch_size, height, width, self.in_channels)
            elif x_torch.shape[0] == 1 and x_torch.shape[1] == 1:
                # [1, 1, B*H*W, C] -> [B, H, W, C]
                x_torch = x_torch.reshape(batch_size, height, width, self.in_channels)
            elif x_torch.shape[1] == height and x_torch.shape[2] == width:
                # Already [B, H, W, C]
                pass
        elif len(x_torch.shape) == 3:
            # [1, B*H*W, C] -> [B, H, W, C]
            x_torch = x_torch.reshape(batch_size, height, width, self.in_channels)

        # Convert NHWC -> NCHW for PyTorch
        x_torch = x_torch.permute(0, 3, 1, 2).contiguous().float()

        # Run deformable conv (reference implementation)
        with torch.no_grad():
            output = self.dcn(x_torch)

        # Apply bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        # Get output dimensions
        out_h, out_w = output.shape[2], output.shape[3]

        # Convert NCHW -> NHWC for TTNN
        output = output.permute(0, 2, 3, 1).contiguous()

        # Convert back to TTNN tensor
        output_ttnn = ttnn.from_torch(
            output,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return output_ttnn, out_h, out_w

    def load_state_dict(self, state_dict: dict, prefix: str = ""):
        """
        Load weights from a state dict.

        Args:
            state_dict: State dictionary containing weights
            prefix: Prefix for keys in state dict (e.g., "depth_conv.4.")
        """
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"
        offset_weight_key = f"{prefix}conv_offset.weight"
        offset_bias_key = f"{prefix}conv_offset.bias"

        if weight_key in state_dict:
            self.dcn.weight.data = state_dict[weight_key].float()

        if offset_weight_key in state_dict:
            self.dcn.conv_offset.weight.data = state_dict[offset_weight_key].float()

        if offset_bias_key in state_dict:
            self.dcn.conv_offset.bias.data = state_dict[offset_bias_key].float()

        if bias_key in state_dict and state_dict[bias_key] is not None:
            self.bias = state_dict[bias_key].float()

        logger.info(f"Loaded TtDeformConv2dPack weights from state_dict with prefix '{prefix}'")
