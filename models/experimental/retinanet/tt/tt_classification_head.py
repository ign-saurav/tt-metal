# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.retinanet.tt.tt_regression_head import Conv2dNormActivation
from dataclasses import dataclass
from models.experimental.retinanet.tt.utils import _create_conv_config_from_params
from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
)


@dataclass
class RetinaNetClassificationHeadOptimizer:
    """Optimization configuration for RetinaNet classification head conv blocks"""

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


# Define optimization configurations
retinanet_classification_head_optimizations = {
    "optimized": RetinaNetClassificationHeadOptimizer(
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


def ttnn_retinanet_classification_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    num_classes: int = 91,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
    model_config: dict = None,
    optimization_profile: str = "optimized",
) -> ttnn.Tensor:
    """
    TTNN implementation of RetinaNet classification head.
    """
    if input_shapes is None:
        input_shapes = [(fm.shape[1], fm.shape[2]) for fm in feature_maps]

    opt_config = retinanet_classification_head_optimizations[optimization_profile]

    grid_size = ttnn.CoreGrid(y=8, x=8)

    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = input_mask_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
        math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
        fp32_dest_acc_en=model_config.get("FP32_DEST_ACC_EN", True),
        packer_l1_acc=model_config.get("PACKER_L1_ACC", False),
    )

    # Process each FPN level
    all_cls_logits = []
    for fpn_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, input_shapes)):
        fpn_conv_config_dict = getattr(opt_config, f"fpn{fpn_idx}_conv_blocks")
        fpn_final_config_dict = getattr(opt_config, f"fpn{fpn_idx}_final_conv")

        fpn_conv_config = ttnn.Conv2dConfig(**fpn_conv_config_dict) if fpn_conv_config_dict else None
        fpn_final_config = ttnn.Conv2dConfig(**fpn_final_config_dict) if fpn_final_config_dict else None

        x = feature_map
        for conv_idx in range(4):
            conv_block = Conv2dNormActivation(
                parameters=parameters["conv"].get(str(conv_idx), {}).get("0", None),
                device=device,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                num_groups=32,
                grid_size=grid_size,
                input_mask=input_mask_tensor,
                model_config=model_config,
                compute_config=compute_config,
                conv_config=fpn_conv_config,
                input_height=H,
                input_width=W,
            )
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = conv_block(
                x, batch_size=batch_size, input_height=H, input_width=W, fpn_level=fpn_idx, conv_block_idx=conv_idx
            )

        conv_final_config = _create_conv_config_from_params(
            input_height=H,
            input_width=W,
            in_channels=in_channels,
            out_channels=num_anchors * num_classes,
            kernel_size=(3, 3),
            batch_size=1,
            parameters=parameters["cls_logits"],
            stride=(1, 1),
            padding=(1, 1),
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        conv_final = TtConv2d(conv_final_config, device)
        cls_logits, shape = conv_final(x, return_output_dim=True)

        N, H_out, W_out, _ = cls_logits.shape
        H_out, W_out = shape
        cls_logits = ttnn.sharded_to_interleaved(cls_logits, ttnn.L1_MEMORY_CONFIG)
        cls_logits = ttnn.reshape(cls_logits, (N, H_out, W_out, num_anchors, num_classes))

        cls_logits = ttnn.reshape(cls_logits, (N, H_out * W_out * num_anchors, num_classes))

        all_cls_logits.append(cls_logits)

    output = ttnn.concat(all_cls_logits, dim=1)
    return output
