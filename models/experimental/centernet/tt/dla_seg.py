# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import numpy as np
from models.common.lightweightmodule import LightweightModule
from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration, HeightShardedStrategyConfiguration
from models.experimental.centernet.tt.dlaup import TtDLAUp
from models.experimental.centernet.tt.dla import TtDLA
from models.experimental.centernet.tt.basic_block import TtBasicBlock


class TtDLASeg(LightweightModule):
    """TTNN implementation of DLASeg for segmentation tasks."""

    def __init__(
        self,
        heads,
        down_ratio=4,
        head_conv=256,
        parameters=None,
        device=None,
        layer_args=None,
    ):
        """
        Initialize TtDLASeg.

        Args:
            heads: Dictionary mapping head names to number of classes
            down_ratio: Downsampling ratio (2, 4, 8, or 16)
            head_conv: Number of channels in head convolution layers
            parameters: Preprocessed model parameters
            device: TTNN device
            layer_args: Layer arguments for configuration
        """
        super(TtDLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.device = device
        self.head_conv = head_conv

        # Initialize DLA backbone
        self.base = TtDLA(
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=TtBasicBlock,
            return_levels=True,
            parameters=parameters.base,
            device=device,
            layer_args=layer_args.base,
        )

        # Get channels from base
        channels = self.base.channels
        self.channels = channels

        # Initialize DLAUpsampling
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = TtDLAUp(
            channels=channels[self.first_level :],
            scales=scales,
            parameters=parameters.dla_up,
            layer_args=layer_args.dla_up,
            device=device,
        )

        # Create prediction heads
        for head in self.heads:
            classes = self.heads[head]
            head_params = getattr(parameters.heads, head)
            head_layer_args = layer_args[head]

            if head_conv > 0:
                fc = self._create_two_layer_head(
                    head_params,
                    head_layer_args,
                    channels[self.first_level],
                    head_conv,
                    classes,
                    head,
                )
            else:
                fc = self._create_single_layer_head(
                    head_params,
                    channels[self.first_level],
                    classes,
                    head,
                )

            self.__setattr__(head, fc)

    def _create_two_layer_head(self, parameters, layer_args, in_channels, head_conv, out_channels, head_name):
        """Create a two-layer head (Conv3x3 -> ReLU -> Conv1x1)."""

        conv1_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

        conv1 = TtConv2d(
            Conv2dConfiguration(
                input_height=layer_args["0"].input_height,
                input_width=layer_args["0"].input_width,
                in_channels=in_channels,
                out_channels=head_conv,
                batch_size=layer_args["0"].batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                weight=parameters.conv1.weight,
                bias=parameters.conv1.bias,
                activation=conv1_config.activation,
                weights_dtype=conv1_config.weights_dtype,
                output_dtype=ttnn.bfloat16,
                sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
                deallocate_activation=conv1_config.deallocate_activation,
            ),
            self.device,
        )

        conv2_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
        )
        conv2 = TtConv2d(
            Conv2dConfiguration(
                input_height=layer_args["2"].input_height,
                input_width=layer_args["2"].input_width,
                in_channels=head_conv,
                out_channels=out_channels,
                batch_size=layer_args["2"].batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=parameters.conv2.weight,
                bias=parameters.conv2.bias,
                activation=None,
                weights_dtype=conv2_config.weights_dtype,
                output_dtype=ttnn.bfloat16,
                sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                deallocate_activation=conv2_config.deallocate_activation,
            ),
            self.device,
        )

        if "hm" in head_name:
            pass

        return [conv1, conv2]

    def _create_single_layer_head(self, parameters, in_channels, out_channels, head_name):
        """Create a single-layer head (Conv1x1)."""

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
        )

        conv = TtConv2d(
            Conv2dConfiguration(
                input_height=parameters.input_height,
                input_width=parameters.input_width,
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=parameters.batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=parameters.weight,
                bias=parameters.bias,
                activation=None,
                weights_dtype=conv_config.weights_dtype,
                output_dtype=ttnn.bfloat16,
                sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                deallocate_activation=conv_config.deallocate_activation,
            ),
            self.device,
        )

        if "hm" in head_name:
            pass

        return conv

    def forward(self, x):
        """Forward pass through TtDLASeg."""
        # Extract features using DLA backbone
        x = self.base(x)

        # Upsample features
        x = self.dla_up(x[self.first_level :])

        # Apply each prediction head
        ret = {}
        for head in self.heads:
            head_module = getattr(self, head)

            if isinstance(head_module, list):
                x_temp = head_module[0](x)  # First conv with ReLU
                ret[head] = head_module[1](x_temp)  # Second conv
            else:
                ret[head] = head_module(x)

        return [ret]
