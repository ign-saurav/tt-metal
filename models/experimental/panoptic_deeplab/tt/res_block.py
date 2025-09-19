# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Dict, Any
from dataclasses import dataclass
from models.experimental.panoptic_deeplab.tt.utils import (
    TTConv2D,
    TTUpsample,
    TTDepthwiseSeparableConv2D,
    DepthwiseSeparableOptimizer,
)


@dataclass
class ResOptimizer:
    project_conv: Dict[Any, Any]
    fuse_conv: DepthwiseSeparableOptimizer
    shape: tuple


res_layer_optimisations = {
    "default": ResOptimizer(
        project_conv={"act_block_h": 32, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        fuse_conv=DepthwiseSeparableOptimizer(
            depthwise={
                "act_block_h": 32,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
            pointwise={
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "deallocate_activation": True,
            },
        ),
        shape=(0, 0, 0, 0),
    ),
    "instance_decoder.res3": ResOptimizer(
        project_conv={
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
        },
        fuse_conv=DepthwiseSeparableOptimizer(
            depthwise={
                "act_block_h": 256,
                "memory_config": ttnn.L1_MEMORY_CONFIG,
                "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
                "enable_act_double_buffer": True,
                "enable_weights_double_buffer": True,
                "reshard_if_not_optimal": True,
            },
            pointwise={
                "act_block_h": 32,
                "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
        ),
        shape=(1, 64, 128, 512),
    ),
    "instance_decoder.res2": ResOptimizer(
        project_conv={
            "act_block_h": 128,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fuse_conv=DepthwiseSeparableOptimizer(
            depthwise={
                "act_block_h": 32,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
            pointwise={
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
        ),
        shape=(1, 128, 256, 256),
    ),
    "semantic_decoder.res3": ResOptimizer(
        project_conv={
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        fuse_conv=DepthwiseSeparableOptimizer(
            depthwise={
                "act_block_h": 256,
                "memory_config": ttnn.L1_MEMORY_CONFIG,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
                "enable_act_double_buffer": True,
                "enable_weights_double_buffer": True,
            },
            pointwise={
                "act_block_h": 32,
                "memory_config": ttnn.L1_MEMORY_CONFIG,
                "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
        ),
        shape=(1, 64, 128, 512),
    ),
    "semantic_decoder.res2": ResOptimizer(
        project_conv={
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        fuse_conv=DepthwiseSeparableOptimizer(
            depthwise={
                "act_block_h": 160,
                "memory_config": ttnn.L1_MEMORY_CONFIG,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
                "enable_act_double_buffer": True,
                "enable_weights_double_buffer": True,
            },
            pointwise={
                "act_block_h": 32,
                "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "deallocate_activation": True,
                "reallocate_halo_output": True,
            },
        ),
        shape=(1, 128, 256, 256),
    ),
}


class TTRes:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=res_layer_optimisations["default"],
    ) -> None:
        # upsample
        self.upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # Project conv
        self.project_conv = TTConv2D(
            kernel_size=parameters.conv_args["project_conv"].kernel_size,
            stride=parameters.conv_args["project_conv"].stride,
            padding=parameters.conv_args["project_conv"].padding,
            dilation=parameters.conv_args["project_conv"].dilation,
            groups=parameters.conv_args["project_conv"].groups,
            parameters=parameters.project_conv,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **layer_optimisations.project_conv,
        )

        # Fuse conv
        self.fuse_conv = TTDepthwiseSeparableConv2D(
            kernel_size=parameters.conv_args["fuse_conv"]["depthwise"].kernel_size,
            stride=parameters.conv_args["fuse_conv"]["depthwise"].stride,
            padding=parameters.conv_args["fuse_conv"]["depthwise"].padding,
            groups=parameters.conv_args["fuse_conv"]["depthwise"].groups,
            dilation=1,
            parameters=parameters.fuse_conv,
            model_config=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            optimisations=layer_optimisations.fuse_conv,
        )
        self.shape = layer_optimisations.shape

    def __call__(
        self,
        x,
        res,
        upsample_channels,
        device,
    ):
        shape = [self.shape[-4], self.shape[-3] // 2, self.shape[-2] // 2, upsample_channels]

        out = self.upsample(device, x, shape, sent_to_dram=True, reshape_output=True)

        out_res, shape = self.project_conv(device, res, self.shape)

        out = ttnn.concat([out_res, out], dim=3)

        shape = (self.shape[-4], self.shape[-3], self.shape[-2], upsample_channels + shape[-1])

        out, shape = self.fuse_conv(device, out, shape)
        return out
