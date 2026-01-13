# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN ResNet50 backbone for MapTR using the tt_cnn format.
"""

from typing import List, Tuple

import torch
import ttnn

from models.experimental.mapTR.tt.common import (
    Conv2dConfig,
    MaxPool2dConfig,
    TtConv2d,
    TtMaxPool2d,
    fold_batch_norm2d_into_conv2d,
)
from models.experimental.mapTR.tt.ttcnn_bottleneck import TtBottleneck
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck


def create_conv_config_from_folded(
    conv: torch.nn.Conv2d,
    bn: torch.nn.BatchNorm2d,
    input_height: int,
    input_width: int,
    batch_size: int,
    activation: ttnn.UnaryWithParam = None,
    shard_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    act_block_h_override: int = 0,
    activation_dtype: ttnn.DataType = ttnn.bfloat16,
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> Conv2dConfig:
    """Create a Conv2dConfig from PyTorch conv+bn layers with folded weights."""
    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return Conv2dConfig(
        input_height=input_height,
        input_width=input_width,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        batch_size=batch_size,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
        weight=ttnn.from_torch(weight, dtype=ttnn.float32),
        bias=ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32),
        activation=activation,
        shard_layout=shard_layout,
        act_block_h_override=act_block_h_override,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
    )


def compute_output_size(input_h: int, input_w: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    """Compute output height and width after conv/pool operation."""
    out_h = (input_h + 2 * padding - kernel) // stride + 1
    out_w = (input_w + 2 * padding - kernel) // stride + 1
    return out_h, out_w


class TtResNet50:
    """ResNet50 backbone using tt_cnn format.

    This implementation uses the builder pattern for cleaner configuration
    and better separation of concerns.
    """

    def __init__(
        self,
        torch_model: ResNet,
        device: ttnn.Device,
        batch_size: int = 6,
        input_height: int = 384,
        input_width: int = 640,
    ):
        """Initialize the ResNet50 backbone.

        Args:
            torch_model: Pre-trained PyTorch ResNet model
            device: TTNN device
            batch_size: Batch size for inference
            input_height: Input image height
            input_width: Input image width
        """
        self.device = device
        self.batch_size = batch_size

        # Track spatial dimensions through the network
        h, w = input_height, input_width

        # Initial conv (7x7, stride 2) + BN + ReLU
        h_after_conv1, w_after_conv1 = compute_output_size(h, w, 7, 2, 3)
        self.conv1_config = create_conv_config_from_folded(
            torch_model.conv1,
            torch_model.bn1,
            h,
            w,
            batch_size,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=32,
        )
        self.conv1 = TtConv2d(self.conv1_config, device)

        # MaxPool (3x3, stride 2)
        h_after_pool, w_after_pool = compute_output_size(h_after_conv1, w_after_conv1, 3, 2, 1)
        self.maxpool_config = MaxPool2dConfig(
            input_height=h_after_conv1,
            input_width=w_after_conv1,
            channels=64,
            batch_size=batch_size,
        )
        self.maxpool = TtMaxPool2d(self.maxpool_config, device)

        # Build residual layers
        h, w = h_after_pool, w_after_pool

        # Layer 1: 3 blocks, no stride change
        self.layer1, h, w = self._make_layer(torch_model.layer1, h, w, batch_size, 64, 256, num_blocks=3, stride=1)

        # Layer 2: 4 blocks, stride 2
        self.layer2, h, w = self._make_layer(
            torch_model.layer2,
            h,
            w,
            batch_size,
            256,
            512,
            num_blocks=4,
            stride=2,
            use_bfloat8=True,
            block_sharded=True,
        )

        # Layer 3: 6 blocks, stride 2
        self.layer3, h, w = self._make_layer(
            torch_model.layer3,
            h,
            w,
            batch_size,
            512,
            1024,
            num_blocks=6,
            stride=2,
            use_bfloat8=True,
            block_sharded=True,
        )

        # Layer 4: 3 blocks, stride 2
        self.layer4, h, w = self._make_layer(
            torch_model.layer4,
            h,
            w,
            batch_size,
            1024,
            2048,
            num_blocks=3,
            stride=2,
            use_bfloat8=True,
            block_sharded=True,
            conv3_block_sharded=True,
        )

        self.output_height = h
        self.output_width = w

    def _make_layer(
        self,
        torch_layer: torch.nn.Sequential,
        input_h: int,
        input_w: int,
        batch_size: int,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_bfloat8: bool = False,
        block_sharded: bool = False,
        conv3_block_sharded: bool = False,
    ) -> Tuple[List[TtBottleneck], int, int]:
        """Create a residual layer with multiple bottleneck blocks."""
        blocks = []
        h, w = input_h, input_w
        expansion = 4  # Bottleneck expansion factor
        width = out_channels // expansion  # Intermediate width

        activation_dtype = ttnn.bfloat8_b if use_bfloat8 else ttnn.bfloat16
        relu_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        for block_idx in range(num_blocks):
            torch_block: Bottleneck = torch_layer[block_idx]
            is_first_block = block_idx == 0
            current_stride = stride if is_first_block else 1
            current_in_channels = in_channels if is_first_block else out_channels

            # Conv1: 1x1 reduce channels
            conv1_config = create_conv_config_from_folded(
                torch_block.conv1,
                torch_block.bn1,
                h,
                w,
                batch_size,
                activation=relu_activation,
            )

            # Conv2: 3x3 with stride
            h_after_conv1, w_after_conv1 = h, w  # 1x1 conv doesn't change spatial dims
            h_after_conv2, w_after_conv2 = compute_output_size(h_after_conv1, w_after_conv1, 3, current_stride, 1)

            conv2_config = create_conv_config_from_folded(
                torch_block.conv2,
                torch_block.bn2,
                h_after_conv1,
                w_after_conv1,
                batch_size,
                activation=relu_activation,
                act_block_h_override=32,
            )

            # Conv3: 1x1 expand channels (no activation)
            conv3_shard = (
                ttnn.TensorMemoryLayout.BLOCK_SHARDED if conv3_block_sharded else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            )
            conv3_config = create_conv_config_from_folded(
                torch_block.conv3,
                torch_block.bn3,
                h_after_conv2,
                w_after_conv2,
                batch_size,
                activation=None,
                shard_layout=conv3_shard,
            )

            # Downsample if needed
            downsample_config = None
            if torch_block.downsample is not None:
                ds_conv = torch_block.downsample[0]
                ds_bn = torch_block.downsample[1]
                ds_shard = (
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED if block_sharded else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                )
                downsample_config = create_conv_config_from_folded(
                    ds_conv,
                    ds_bn,
                    h,
                    w,
                    batch_size,
                    activation=None,
                    shard_layout=ds_shard,
                    activation_dtype=activation_dtype,
                )

            block = TtBottleneck(
                conv1_config=conv1_config,
                conv2_config=conv2_config,
                conv3_config=conv3_config,
                device=self.device,
                downsample_config=downsample_config,
                activation_dtype=activation_dtype if is_first_block else ttnn.bfloat16,
            )
            blocks.append(block)

            # Update spatial dimensions after first block
            if is_first_block:
                h, w = h_after_conv2, w_after_conv2

        return blocks, h, w

    def __call__(self, x: ttnn.Tensor) -> List[ttnn.Tensor]:
        """Execute the ResNet50 forward pass.

        Args:
            x: Input tensor in NHWC format, flattened to (1, 1, N*H*W, C)

        Returns:
            List of output feature maps (only layer4 output for out_indices=(3,))
        """
        # Initial conv + ReLU
        x, _, _ = self.conv1(x)
        x = ttnn.sharded_to_interleaved(x)

        # MaxPool with batch splitting for large batches
        if self.maxpool_config.batch_size > 1:
            x = self._split_maxpool(x)
        else:
            x = self.maxpool(x)

        # Layer 1
        for block in self.layer1:
            x = block(x)
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Layer 2
        for block in self.layer2:
            x = block(x)

        # Layer 3
        for block in self.layer3:
            x = block(x)

        # Layer 4
        for block in self.layer4:
            x = block(x)

        return [x]

    def _split_maxpool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply maxpool with batch splitting for memory efficiency."""
        config = self.maxpool_config
        split_point = config.batch_size // 2
        spatial_size = config.input_height * config.input_width

        # Split input
        x0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, split_point * spatial_size, x.shape[3]])
        x1 = ttnn.slice(
            x,
            [0, 0, split_point * spatial_size, 0],
            [1, 1, config.batch_size * spatial_size, x.shape[3]],
        )

        # Apply maxpool to each half
        x0 = ttnn.max_pool2d(
            input_tensor=x0,
            batch_size=split_point,
            input_h=config.input_height,
            input_w=config.input_width,
            channels=config.channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )
        x1 = ttnn.max_pool2d(
            input_tensor=x1,
            batch_size=config.batch_size - split_point,
            input_h=config.input_height,
            input_w=config.input_width,
            channels=config.channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        # Concatenate results
        return ttnn.concat((x0, x1), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
