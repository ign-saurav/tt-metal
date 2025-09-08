# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTChannelAttention(LightweightModule):
    def __init__(self, device, parameters, num_feat, squeeze_factor=16, memory_config=None):
        super().__init__()

        self.device = device
        # self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.num_feat = num_feat
        self.squeeze_factor = squeeze_factor

        # Extract preprocessed parameters
        self.conv1_weight = parameters["conv1"]["weight"]
        self.conv1_bias = parameters["conv1"]["bias"]
        self.conv2_weight = parameters["conv2"]["weight"]
        self.conv2_bias = parameters["conv2"]["bias"]

    def forward(self, x):
        # Store original input for multiplication
        original_x = x
        original_shape = x.shape

        # NOTE: if the input is not in L1, convert it to L1, convertion takes 20 us hence only makes sense for the input to be already in L1
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # Global Average Pooling (AdaptiveAvgPool2d(1) equivalent)
        x = ttnn.global_avg_pool2d(x, memory_config=self.memory_config)

        # 180 -> 192 -> 180
        # if math.log(original_shape[-1], 2) != 0:
        if original_shape[-1] == 180:
            x = ttnn.slice(
                x,
                starts=[0, 0, 0, 0],  # Start indices for each dimension
                ends=[original_shape[0], 1, 1, 180],  # End indices - slice to 180 in last dim
                steps=[1, 1, 1, 1],  # Step size for each dimension
            )
        # TODO: find ways to generalise for all inputs, setting program config messes up the multi batch runs..
        # Matrix multiplication 1:  [1, 180] @ [180, 6] = [1, 6]
        # program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=(10, 1),  # or (5, 2) to accommodate 10 batches
        #     in0_block_w=6,  # Keep same as your calculation
        #     out_subblock_h=1,
        #     out_subblock_w=1,
        #     per_core_M=1,  # Each core handles 1 batch worth of M dimension
        #     per_core_N=1,
        #     fuse_batch=True,
        #     fused_activation=None,
        #     mcast_in0=False,
        # )
        x = ttnn.linear(
            x,
            self.conv1_weight,
            bias=self.conv1_bias,
            memory_config=self.memory_config,
            # program_config=program_config,
            activation="relu",
            # compute_kernel_config=compute_kernel_config,  # set to HiFi2 to improve accuracy
        )

        # Matrix multiplication 2:  [1, 6] @ [6, 180] = [1, 180]
        # program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=(6, 6),
        #     in0_block_w=1, # 32(6 padded) / 32(1 tile size) = 1
        #     out_subblock_h=1,
        #     out_subblock_w=1,
        #     per_core_M=1,
        #     per_core_N=1,
        #     fuse_batch=True,
        #     fused_activation=None,
        #     mcast_in0=False,
        # )
        x = ttnn.linear(
            x,
            self.conv2_weight,
            bias=self.conv2_bias,
            memory_config=self.memory_config,
            # program_config=program_config,
            # compute_kernel_config=compute_kernel_config,  # set to HiFi2 to improve accuracy
        )

        # Sigmoid activation
        x = ttnn.sigmoid(x)  # sigmoid in conv2d config, return invalid activation fn error

        batch_size, height, width, channels = original_shape
        attention_weights = ttnn.reshape(x, [batch_size, 1, 1, channels])
        # NOTE: float16 or float32 only supported for repeat
        attention_weights = ttnn.repeat(attention_weights, [1, height, width, 1], memory_config=self.memory_config)

        # Element-wise multiplication with original input
        # setting the dtype of the output to bfloat16 is not improving the performance, maybe affected by BF16 => BFP4 overhead
        output = ttnn.multiply(original_x, attention_weights, memory_config=self.memory_config)

        return output
