# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
import tt_lib.fallback_ops as fallback_ops
import os
from models.experimental.retinanet.tt.utils import _create_conv_config_from_params
from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import (
    HeightShardedStrategyConfiguration,
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
)


@dataclass
class RetinaNetHeadOptimizer:
    fpn0_conv_blocks: dict
    fpn1_conv_blocks: dict
    fpn2_conv_blocks: dict
    fpn3_conv_blocks: dict
    fpn4_conv_blocks: dict

    fpn0_final_conv: dict
    fpn1_final_conv: dict
    fpn2_final_conv: dict
    fpn3_final_conv: dict
    fpn4_final_conv: dict


retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        fpn0_conv_blocks={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn0_final_conv={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_conv_blocks={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_final_conv={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


class Conv2dNormActivation:
    def __init__(
        self,
        parameters: dict,
        input_height: int,
        input_width: int,
        device: ttnn.Device,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        num_groups: int = 32,
        grid_size: Optional[ttnn.CoreGrid] = None,
        input_mask: Optional[ttnn.Tensor] = None,
        model_config: dict = None,
        compute_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
        conv_config: dict = None,  # Changed to dict for optimizer config
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.model_config = model_config
        self.compute_config = compute_config

        self.conv_weight = parameters["weight"]
        self.conv_bias = parameters["bias"]
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]

        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "1") == "1"
        self.grid_size = grid_size if grid_size is not None else ttnn.CoreGrid(y=8, x=8)
        self.input_mask = input_mask

        base_conv_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            batch_size=1,
            parameters=parameters,
            stride=stride,
            padding=padding,
            sharding_strategy=HeightShardedStrategyConfiguration()
            if input_height == 4
            else AutoShardedStrategyConfiguration(),
        )

        config_dict = base_conv_config.__dict__.copy()

        if conv_config:  # conv_config is a dict
            for key, value in conv_config.items():
                if key in config_dict:
                    config_dict[key] = value

        self.conv_config = Conv2dConfiguration(**config_dict)
        self.conv = TtConv2d(self.conv_config, device)

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
        fpn_level: int = None,
        conv_block_idx: int = None,
    ) -> ttnn.Tensor:
        prefix = (
            f"[FPN{fpn_level}][Conv{conv_block_idx}]"
            if fpn_level is not None and conv_block_idx is not None
            else "[Conv]"
        )

        x, [H_out, W_out] = self.conv(x, return_output_dim=True)
        N, H_out, W_out, C = x.shape

        if self.fallback_on_groupnorm:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (N, input_height, input_width, C))
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.permute(x, (0, 3, 1, 2))

            x = fallback_ops.group_norm(
                x,
                num_groups=self.num_groups,
                weight=self.norm_weight,
                bias=self.norm_bias,
            )
            x = x.to(self.device)
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        else:
            logger.debug(f"{prefix} Using TTNN native GroupNorm")
            spatial_size = H_out * W_out
            required_size = ((spatial_size + self.grid_size.y * 32 - 1) // (self.grid_size.y * 32)) * (
                self.grid_size.y * 32
            )

            if spatial_size != required_size:
                pad_amount = required_size - spatial_size
                x_flat = ttnn.reshape(x, (N, 1, spatial_size, C))
                x_padded = ttnn.pad(x_flat, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
            else:
                x_padded = ttnn.reshape(x, (N, 1, spatial_size, C))

            x_normalized = ttnn.group_norm(
                x_padded,
                num_groups=self.num_groups,
                input_mask=self.input_mask,
                weight=self.norm_weight,
                bias=self.norm_bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.grid_size,
                inplace=False,
                compute_kernel_config=self.compute_config,
            )

            if spatial_size != required_size:
                x_normalized = x_normalized[:, :, :spatial_size, :]

            x = ttnn.reshape(x_normalized, (N, input_height, input_width, C))

        H_out = input_height
        W_out = input_width

        x = ttnn.relu(x)

        return x


class TtnnRetinaNetRegressionHead:
    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        num_anchors: int = 9,
        batch_size: int = 1,
        input_shapes: List[tuple] = None,
        model_config: dict = None,
        optimization_profile: str = "optimized",
    ):
        self.device = device
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.input_shapes = input_shapes
        self.model_config = model_config
        self.optimization_profile = optimization_profile

        self.parameters = parameters
        self.opt_config = retinanet_head_optimizations[optimization_profile]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
            fp32_dest_acc_en=model_config.get("FP32_DEST_ACC_EN", True),
            packer_l1_acc=model_config.get("PACKER_L1_ACC", False),
        )

        self.grid_size = ttnn.CoreGrid(y=8, x=8)
        input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, self.grid_size.y)
        self.input_mask_tensor = input_mask_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)

        self.fpn_optimizer_configs = {
            0: (self.opt_config.fpn0_conv_blocks, self.opt_config.fpn0_final_conv),
            1: (self.opt_config.fpn1_conv_blocks, self.opt_config.fpn1_final_conv),
            2: (self.opt_config.fpn2_conv_blocks, self.opt_config.fpn2_final_conv),
            3: (self.opt_config.fpn3_conv_blocks, self.opt_config.fpn3_final_conv),
            4: (self.opt_config.fpn4_conv_blocks, self.opt_config.fpn4_final_conv),
        }

        self.conv_blocks_by_fpn = {}

        for fpn_idx in range(5):
            conv_opt, final_opt = self.fpn_optimizer_configs[fpn_idx]

            conv_blocks = []
            for conv_idx in range(4):
                conv_block = Conv2dNormActivation(
                    parameters=self.parameters["conv"].get(str(conv_idx), {}).get("0", None),
                    device=self.device,
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    num_groups=32,
                    grid_size=self.grid_size,
                    input_mask=self.input_mask_tensor,
                    model_config=self.model_config,
                    compute_config=self.compute_config,
                    conv_config=conv_opt,  # Apply optimizer config
                    input_height=64,  # Default, will be updated in forward
                    input_width=64,  # Default, will be updated in forward
                )
                conv_blocks.append(conv_block)

            self.conv_blocks_by_fpn[fpn_idx] = conv_blocks

            self.final_conv_configs = {}
            self.final_conv_configs[fpn_idx] = final_opt

    def forward(
        self,
        feature_maps: List[ttnn.Tensor],
        batch_size: Optional[int] = None,
        input_shapes: Optional[List[tuple]] = None,
    ) -> ttnn.Tensor:
        current_batch_size = batch_size if batch_size is not None else self.batch_size
        current_input_shapes = input_shapes if input_shapes is not None else self.input_shapes

        if current_input_shapes is None:
            current_input_shapes = [(fm.shape[1], fm.shape[2]) for fm in feature_maps]

        all_bbox_regression = []

        for fpn_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, current_input_shapes)):
            conv_blocks = self.conv_blocks_by_fpn[fpn_idx]

            x = feature_map
            for conv_idx, conv_block in enumerate(conv_blocks):
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                x = conv_block(
                    x,
                    batch_size=current_batch_size,
                    input_height=H,
                    input_width=W,
                    fpn_level=fpn_idx,
                    conv_block_idx=conv_idx,
                )

            final_conv_optimizer = self.final_conv_configs[fpn_idx]

            conv_final_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
                in_channels=self.in_channels,
                out_channels=self.num_anchors * 4,
                kernel_size=(3, 3),
                batch_size=1,
                parameters=self.parameters["bbox_reg"],
                stride=(1, 1),
                padding=(1, 1),
                sharding_strategy=HeightShardedStrategyConfiguration(),
            )

            if final_conv_optimizer:
                config_dict = conv_final_config.__dict__.copy()
                for key, value in final_conv_optimizer.items():
                    if key in config_dict:
                        config_dict[key] = value
                conv_final_config = Conv2dConfiguration(**config_dict)

            conv_final = TtConv2d(conv_final_config, self.device)
            bbox_regression, shape = conv_final(x, return_output_dim=True)

            N, H_out, W_out, _ = bbox_regression.shape
            H_out, W_out = shape
            bbox_regression = ttnn.sharded_to_interleaved(bbox_regression, ttnn.DRAM_MEMORY_CONFIG)
            bbox_regression = ttnn.reshape(bbox_regression, (N, H_out, W_out, self.num_anchors, 4))
            bbox_regression = ttnn.reshape(bbox_regression, (N, H_out * W_out * self.num_anchors, 4))

            all_bbox_regression.append(bbox_regression)

        output = ttnn.concat(all_bbox_regression, dim=1)
        return output

    def __call__(self, feature_maps: List[ttnn.Tensor], **kwargs) -> ttnn.Tensor:
        return self.forward(feature_maps, **kwargs)
