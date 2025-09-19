# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.panoptic_deeplab.tt.utils import (
    TTConv2D,
    TTUpsample,
    TTDepthwiseSeparableConv2D,
    DepthwiseSeparableOptimizer,
)


aspp_optimisations = [
    DepthwiseSeparableOptimizer(
        depthwise={
            "act_block_h": 64,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "reshard_if_not_optimal": True,
        },
        pointwise={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reshard_if_not_optimal": True,
        },
    ),
    DepthwiseSeparableOptimizer(
        depthwise={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "reshard_if_not_optimal": True,
        },
        pointwise={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reshard_if_not_optimal": True,
        },
    ),
    DepthwiseSeparableOptimizer(
        depthwise={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "reshard_if_not_optimal": True,
        },
        pointwise={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reshard_if_not_optimal": True,
        },
    ),
]


class TTASPP:
    def __init__(self, parameters, model_config) -> None:
        self.model_config = model_config

        dilations = [6, 12, 18]
        in_channels = 2048
        activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        self.convs = []

        # conv 1x1
        self.convs.append(
            TTConv2D(
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                parameters=parameters["convs"][0],
                kernel_fidelity=model_config,
                activation=activation,
                act_block_h=32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                reshard_if_not_optimal=True,
            )
        )
        # atrous convs
        for index, dilation in enumerate(dilations):
            self.convs.append(
                TTDepthwiseSeparableConv2D(
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    groups=in_channels,
                    parameters=parameters["convs"][index + 1],
                    model_config=model_config,
                    activation=activation,
                    optimisations=aspp_optimisations[index],
                )
            )

        # image pooling
        self.pooling_conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters["convs"][4][1],
            kernel_fidelity=model_config,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
        )

        # Upsample
        self.upsample = TTUpsample(
            scale_factor=(32, 64),
            mode="bilinear",
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # Projection conv
        self.project = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.project,
            kernel_fidelity=model_config,
            activation=activation,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=True,
        )

    def __call__(
        self,
        x,
        device,
    ):
        res = []
        for conv in self.convs:
            out, _ = conv(device, x, (1, 32, 64, 2048))
            res.append(out)

        x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])
        out = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=32,
            input_w=64,
            channels=2048,
            kernel_size=(32, 64),
            stride=(1, 1),
            padding=(0, 0),
            applied_shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            in_place_halo=True,
            deallocate_input=True,
            reallocate_halo_output=True,
        )
        ttnn.deallocate(x, force=True)

        out, shape = self.pooling_conv(device, out, (1, 1, 1, 2048))
        out = self.upsample(device, out, [1, 1, 1, 256], reshape_output=True, dtype=ttnn.bfloat8_b)
        res.append(out)

        aspp_concat = ttnn.concat(res, dim=3)
        for res_out in res:
            ttnn.deallocate(res_out, force=True)

        shape = (1, 32, 64, 1280)
        out, shape = self.project(device, aspp_concat, shape)

        return out
