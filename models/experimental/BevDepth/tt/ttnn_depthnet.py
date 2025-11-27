# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


class BasicBlock_TTNN:
    def __init__(self, device, parameters, in_channels, out_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x, batch_size, height, width):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        identity = x

        # Conv1: 3x3
        out = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.conv1_weight,
            bias_tensor=self.params.conv1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )

        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, height, width, self.out_channels))

        # Conv2: 3x3 (no activation)
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.params.conv2_weight,
            bias_tensor=self.params.conv2_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            **self.model_config,
        )

        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, height, width, self.out_channels))

        # Add + ReLU
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out


class ASPP_TTNN:
    def __init__(self, device, parameters, in_channels, mid_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.model_config = model_config
        self.params = parameters

    def __call__(self, x, batch_size, height, width):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        # Branch 1: 1x1 conv, dilation=1
        x1 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp1_weight,
            bias_tensor=self.params.aspp1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )
        if len(x1.shape) == 3:
            x1 = ttnn.reshape(x1, (batch_size, height, width, self.mid_channels))

        # Branch 2-4: 3x3 conv with dilation (simplified - use padding instead)
        x2 = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.aspp2_weight,
            bias_tensor=self.params.aspp2_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(6, 6),  # dilation=6
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )
        if len(x2.shape) == 3:
            x2 = ttnn.reshape(x2, (batch_size, height, width, self.mid_channels))

        # Global pooling branch
        x5 = ttnn.global_avg_pool2d(x)
        x5 = ttnn_conv2d(
            input_tensor=x5,
            weight_tensor=self.params.global_weight,
            bias_tensor=self.params.global_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=1,
            input_width=1,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )
        x5 = ttnn.upsample(x5, (batch_size, height, width, self.mid_channels))

        # Concatenate (simplified - use only 3 branches)
        out = ttnn.concat([x1, x2, x5], dim=-1)

        # Final conv
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.params.conv1_weight,
            bias_tensor=self.params.conv1_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels * 3,
            out_channels=self.mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )

        return out


class DepthNet_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels=512,
        mid_channels=256,
        context_channels=512,
        depth_channels=118,
        model_config=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.params = parameters

        # Initialize sub-modules
        self.block1 = BasicBlock_TTNN(device, parameters.block1, mid_channels, mid_channels, self.model_config)
        self.block2 = BasicBlock_TTNN(device, parameters.block2, mid_channels, mid_channels, self.model_config)
        self.block3 = BasicBlock_TTNN(device, parameters.block3, mid_channels, mid_channels, self.model_config)
        self.aspp = ASPP_TTNN(device, parameters.aspp, mid_channels, mid_channels, self.model_config)

        logger.info(f"DepthNet init: in={in_channels}, mid={mid_channels}, depth={depth_channels}")

    def __call__(self, x, batch_size=1):
        from models.experimental.BevDepth.tt.utils import ttnn_conv2d

        height, width = x.shape[1], x.shape[2]

        # Reduce conv
        x = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.reduce_weight,
            bias_tensor=self.params.reduce_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **self.model_config,
        )

        if len(x.shape) == 3:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))

        # Context branch
        context = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.params.context_weight,
            bias_tensor=self.params.context_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels,
            out_channels=self.context_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,
            **self.model_config,
        )

        if len(context.shape) == 3:
            context = ttnn.reshape(context, (batch_size, height, width, self.context_channels))

        # Depth branch: 3 BasicBlocks
        depth = self.block1(x, batch_size, height, width)
        depth = self.block2(depth, batch_size, height, width)
        depth = self.block3(depth, batch_size, height, width)

        # ASPP
        depth = self.aspp(depth, batch_size, height, width)

        # DCN (Deformable Conv) - using regular grouped conv as approximation
        # TODO: Replace with actual DCN when available in TTNN
        depth = ttnn_conv2d(
            input_tensor=depth,
            weight_tensor=self.params.dcn_weight,
            bias_tensor=self.params.dcn_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation=None,
            **self.model_config,
        )

        if len(depth.shape) == 3:
            depth = ttnn.reshape(depth, (batch_size, height, width, self.mid_channels))

        # Final depth conv
        depth = ttnn_conv2d(
            input_tensor=depth,
            weight_tensor=self.params.final_weight,
            bias_tensor=self.params.final_bias,
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.mid_channels,
            out_channels=self.depth_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation=None,
            **self.model_config,
        )

        if len(depth.shape) == 3:
            depth = ttnn.reshape(depth, (batch_size, height, width, self.depth_channels))

        # Concatenate depth and context
        out = ttnn.concat([depth, context], dim=-1)

        return out


def prepare_depthnet_parameters(state_dict, in_channels=512, mid_channels=256, depth_channels=118):
    class Parameters:
        pass

    params = Parameters()

    # Find the actual prefix used in this checkpoint
    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.depth_net.",
        "img_backbone.depth_net.",
        "backbone.depth_net.",
        "depth_net.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        # No depth_net found, return the full state dict for debugging
        logger.error(f"Could not find depth_net prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No depth_net keys found in checkpoint")

    logger.info(f"Using DepthNet prefix: {prefix}")

    # Reduce conv
    try:
        params.reduce_weight = state_dict[f"{prefix}reduce_conv.0.weight"].to(torch.bfloat16)
        params.reduce_bias = state_dict.get(f"{prefix}reduce_conv.0.bias", None)
        if params.reduce_bias is not None:
            params.reduce_bias = params.reduce_bias.to(torch.bfloat16)
    except KeyError as e:
        logger.error(f"Failed to load reduce_conv: {e}")
        logger.info(f"Available depth_net keys: {[k for k in all_keys if prefix in k][:20]}")
        raise

    # Context conv
    params.context_weight = state_dict[f"{prefix}context_conv.weight"].to(torch.bfloat16)
    params.context_bias = state_dict.get(f"{prefix}context_conv.bias", None)
    if params.context_bias is not None:
        params.context_bias = params.context_bias.to(torch.bfloat16)

    # BasicBlocks (depth_conv.0, depth_conv.1, depth_conv.2)
    for i in range(3):
        block = Parameters()
        block.conv1_weight = state_dict[f"{prefix}depth_conv.{i}.conv1.weight"].to(torch.bfloat16)
        block.conv1_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv1.bias", None)
        block.conv2_weight = state_dict[f"{prefix}depth_conv.{i}.conv2.weight"].to(torch.bfloat16)
        block.conv2_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv2.bias", None)
        if block.conv1_bias is not None:
            block.conv1_bias = block.conv1_bias.to(torch.bfloat16)
        if block.conv2_bias is not None:
            block.conv2_bias = block.conv2_bias.to(torch.bfloat16)
        setattr(params, f"block{i+1}", block)

    # ASPP (depth_conv.3)
    params.aspp = Parameters()
    params.aspp.aspp1_weight = state_dict[f"{prefix}depth_conv.3.aspp1.atrous_conv.weight"].to(torch.bfloat16)
    params.aspp.aspp1_bias = state_dict.get(f"{prefix}depth_conv.3.aspp1.atrous_conv.bias", None)
    params.aspp.aspp2_weight = state_dict[f"{prefix}depth_conv.3.aspp2.atrous_conv.weight"].to(torch.bfloat16)
    params.aspp.aspp2_bias = state_dict.get(f"{prefix}depth_conv.3.aspp2.atrous_conv.bias", None)
    params.aspp.global_weight = state_dict[f"{prefix}depth_conv.3.global_avg_pool.1.weight"].to(torch.bfloat16)
    params.aspp.global_bias = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.1.bias", None)
    params.aspp.conv1_weight = state_dict[f"{prefix}depth_conv.3.conv1.weight"].to(torch.bfloat16)
    params.aspp.conv1_bias = state_dict.get(f"{prefix}depth_conv.3.conv1.bias", None)

    # DCN layer (depth_conv.4)
    params.dcn_weight = state_dict[f"{prefix}depth_conv.4.weight"].to(torch.bfloat16)
    params.dcn_bias = state_dict.get(f"{prefix}depth_conv.4.bias", None)
    if params.dcn_bias is not None:
        params.dcn_bias = params.dcn_bias.to(torch.bfloat16)

    # Final conv (depth_conv.5)
    params.final_weight = state_dict[f"{prefix}depth_conv.5.weight"].to(torch.bfloat16)
    params.final_bias = state_dict.get(f"{prefix}depth_conv.5.bias", None)
    if params.final_bias is not None:
        params.final_bias = params.final_bias.to(torch.bfloat16)

    logger.info("Successfully prepared DepthNet parameters")
    return params
