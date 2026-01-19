# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from typing import List

import ttnn

from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration, AutoShardedStrategyConfiguration
from models.experimental.MapTR.tt.bottleneck import TtBottleneck


def create_conv_config_from_args(conv_args, conv_pth, activation=None):
    """Create Conv2dConfiguration from model args and weights."""
    # Get weight tensor
    weight = conv_pth.weight

    # Check if weight needs to be permuted from HWIO to OICHW format
    # Expected shape: (out_channels, in_channels, kernel_h, kernel_w)
    # If shape is (kernel_h, kernel_w, in_channels, out_channels), permute it
    if len(weight.shape) == 4:
        h, w, in_ch, out_ch = weight.shape
        expected_out_ch = conv_args.out_channels
        expected_in_ch = conv_args.in_channels // conv_args.groups
        expected_kernel_h, expected_kernel_w = conv_args.kernel_size

        # If shape matches HWIO format, permute to OICHW
        if h == expected_kernel_h and w == expected_kernel_w and in_ch == expected_in_ch and out_ch == expected_out_ch:
            # Permute from (H, W, I, O) to (O, I, H, W)
            weight = ttnn.permute(weight, (3, 2, 0, 1))

    return Conv2dConfiguration.from_model_args(
        conv2d_args=conv_args,
        weights=weight,
        bias=conv_pth.bias if hasattr(conv_pth, "bias") else None,
        activation=activation,
        sharding_strategy=AutoShardedStrategyConfiguration(),
    )


class TtResNet50:
    """ResNet50 backbone using"""

    def __init__(
        self,
        conv_args,
        conv_pth,
        device: ttnn.Device,
    ):
        """Initialize the ResNet50 backbone with preprocessed parameters.

        Args:
            conv_args: Preprocessed convolution arguments from infer_ttnn_module_args
            conv_pth: Preprocessed weights from custom_preprocessor
            device: TTNN device
        """
        self.device = device
        self.maxpool_args = conv_args.maxpool

        # Initial conv (7x7, stride 2) + BN + ReLU
        conv1_config = create_conv_config_from_args(
            conv_args.conv1,
            conv_pth.conv1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv1 = TtConv2d(conv1_config, device)

        # Layer 1 - use attribute access with string keys
        self.layer1_0 = TtBottleneck(
            getattr(conv_args.layer1, "0"),
            conv_pth.layer1_0,
            device=self.device,
            is_downsample=True,
        )
        self.layer1_1 = TtBottleneck(getattr(conv_args.layer1, "1"), conv_pth.layer1_1, device=self.device)
        self.layer1_2 = TtBottleneck(getattr(conv_args.layer1, "2"), conv_pth.layer1_2, device=self.device)

        # Layer 2
        self.layer2_0 = TtBottleneck(
            getattr(conv_args.layer2, "0"),
            conv_pth.layer2_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer2_1 = TtBottleneck(getattr(conv_args.layer2, "1"), conv_pth.layer2_1, device=self.device)
        self.layer2_2 = TtBottleneck(getattr(conv_args.layer2, "2"), conv_pth.layer2_2, device=self.device)
        self.layer2_3 = TtBottleneck(getattr(conv_args.layer2, "3"), conv_pth.layer2_3, device=self.device)

        # Layer 3
        self.layer3_0 = TtBottleneck(
            getattr(conv_args.layer3, "0"),
            conv_pth.layer3_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer3_1 = TtBottleneck(getattr(conv_args.layer3, "1"), conv_pth.layer3_1, device=self.device)
        self.layer3_2 = TtBottleneck(getattr(conv_args.layer3, "2"), conv_pth.layer3_2, device=self.device)
        self.layer3_3 = TtBottleneck(getattr(conv_args.layer3, "3"), conv_pth.layer3_3, device=self.device)
        self.layer3_4 = TtBottleneck(getattr(conv_args.layer3, "4"), conv_pth.layer3_4, device=self.device)
        self.layer3_5 = TtBottleneck(getattr(conv_args.layer3, "5"), conv_pth.layer3_5, device=self.device)

        # Layer 4
        self.layer4_0 = TtBottleneck(
            getattr(conv_args.layer4, "0"),
            conv_pth.layer4_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
            conv3_blk_sharded=True,
        )
        self.layer4_1 = TtBottleneck(
            getattr(conv_args.layer4, "1"),
            conv_pth.layer4_1,
            device=self.device,
            conv3_blk_sharded=True,
        )
        self.layer4_2 = TtBottleneck(
            getattr(conv_args.layer4, "2"),
            conv_pth.layer4_2,
            device=self.device,
            conv3_blk_sharded=True,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int = 1) -> List[ttnn.Tensor]:
        """Execute the ResNet50 forward pass.

        Args:
            x: Input tensor in NHWC format, flattened to (1, 1, N*H*W, C)
            batch_size: Batch size for processing

        Returns:
            List of output feature maps (only layer4 output for out_indices=(3,))
        """
        # Initial conv + ReLU
        x = self.conv1(x)
        x = ttnn.sharded_to_interleaved(x)

        # MaxPool with batch splitting for large batches
        if self.maxpool_args.batch_size > 1:
            x = self._split_maxpool(x)
        else:
            x = ttnn.max_pool2d(
                input_tensor=x,
                batch_size=self.maxpool_args.batch_size,
                input_h=self.maxpool_args.input_height,
                input_w=self.maxpool_args.input_width,
                channels=x.shape[3],
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ceil_mode=False,
            )

        # Layer 1
        x = self.layer1_0(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.layer1_1(x)
        x = self.layer1_2(x)

        # Layer 2
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        # Layer 3
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)

        # Layer 4
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return [x]

    def _split_maxpool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply maxpool with batch splitting for memory efficiency."""
        config = self.maxpool_args
        split_point = config.batch_size // 2
        spatial_size = config.input_height * config.input_width
        channels = x.shape[3]

        # Split input
        x0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, split_point * spatial_size, channels])
        x1 = ttnn.slice(
            x,
            [0, 0, split_point * spatial_size, 0],
            [1, 1, config.batch_size * spatial_size, channels],
        )

        # Apply maxpool to each half
        x0 = ttnn.max_pool2d(
            input_tensor=x0,
            batch_size=split_point,
            input_h=config.input_height,
            input_w=config.input_width,
            channels=channels,
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
            channels=channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        # Concatenate results
        return ttnn.concat((x0, x1), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
