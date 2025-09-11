# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from .channel_attention import TTChannelAttention


class TTCAB(LightweightModule):
    def __init__(self, device, parameters, num_feat, compress_ratio=3, squeeze_factor=30, memory_config=None):
        super().__init__()

        self.device = device
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.num_feat = num_feat
        self.compress_ratio = compress_ratio
        self.squeeze_factor = squeeze_factor

        # Extract preprocessed parameters for convolutions
        self.conv1_weight = parameters["conv1"]["weight"]
        self.conv1_bias = parameters["conv1"]["bias"]
        self.conv2_weight = parameters["conv2"]["weight"]
        self.conv2_bias = parameters["conv2"]["bias"]

        # Initialize channel attention module
        self.channel_attention = TTChannelAttention(
            device=device,
            parameters=parameters["channel_attention"],
            num_feat=num_feat,
            squeeze_factor=squeeze_factor,
            memory_config=memory_config,
        )

    def forward(self, x):
        # Store original input shape for convolutions
        batch_size, height, width, channels = x.shape
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            output_layout=ttnn.TILE_LAYOUT,
            activation="gelu",
        )
        # First 3x3 convolution (compression)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight,
            bias_tensor=self.conv1_bias,
            device=self.device,
            in_channels=self.num_feat,
            out_channels=self.num_feat // self.compress_ratio,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            memory_config=self.memory_config,
            conv_config=conv_config,
            compute_config=ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
        )

        # Reshape from flattened conv output back to spatial format
        x = ttnn.reshape(x, [batch_size, height, width, self.num_feat // self.compress_ratio])

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            output_layout=ttnn.TILE_LAYOUT,
        )
        # Second 3x3 convolution (expansion)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            device=self.device,
            in_channels=self.num_feat // self.compress_ratio,
            out_channels=self.num_feat,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            memory_config=self.memory_config,
            compute_config=ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            conv_config=conv_config,
        )

        # Reshape from flattened conv output back to spatial format
        x = ttnn.reshape(x, [batch_size, height, width, self.num_feat], memory_config=self.memory_config)

        # Apply channel attention
        x = self.channel_attention(x)

        return x
