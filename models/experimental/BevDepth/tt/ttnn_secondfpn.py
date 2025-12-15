# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from dataclasses import dataclass
from loguru import logger
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
)


@dataclass
class SECONDFPNOptimizations:
    conv_transpose: dict
    conv2d: dict


secondfpn_optimizations = SECONDFPNOptimizations(
    conv_transpose={
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    conv2d={
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)


@dataclass
class SECONDFPNHeadOptimizations:
    conv_transpose: dict


secondfpn_head_optimizations = SECONDFPNHeadOptimizations(
    conv_transpose={
        "deallocate_activation": False,
    },
)


@dataclass
class ConvTransposeConfig:
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    input_height: int
    input_width: int
    batch_size: int
    weight: ttnn.Tensor
    bias: ttnn.Tensor


class SECONDFPN_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels,
        out_channels,
        upsample_strides,
        model_config,
        input_shapes=None,
        batch_size=1,
        optimizations=None,
        use_torch_fallback=False,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_levels = len(in_channels)
        self.model_config = model_config
        self.batch_size = batch_size
        self.optimizations = optimizations or secondfpn_optimizations
        self.deblocks = parameters.deblocks
        self.use_torch_fallback = use_torch_fallback

        if input_shapes is None:
            input_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]
        self.input_shapes = input_shapes

        self.conv_layers = []
        self.conv_transpose_configs = []
        self._torch_weights = []

        for i in range(self.num_levels):
            kernel_size = self.deblocks[i].kernel_size
            stride = self.upsample_strides[i]
            use_conv_transpose = stride >= 1
            input_h, input_w = self.input_shapes[i]

            weight = self.deblocks[i].conv_weight
            bias = self.deblocks[i].conv_bias

            if isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight)
            if isinstance(bias, np.ndarray):
                bias = torch.from_numpy(bias)

            weight = weight.float()
            if bias is not None:
                bias = bias.float()

            self._torch_weights.append((weight.clone(), bias.clone() if bias is not None else None))

            if use_conv_transpose:
                weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
                bias_ttnn = None
                if bias is not None:
                    if len(bias.shape) == 1:
                        bias = bias.view(1, 1, 1, -1)
                    bias_ttnn = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

                config = ConvTransposeConfig(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size,
                    stride=(int(stride), int(stride)),
                    input_height=input_h,
                    input_width=input_w,
                    batch_size=batch_size,
                    weight=weight_ttnn,
                    bias=bias_ttnn,
                )
                self.conv_transpose_configs.append(config)
                self.conv_layers.append(None)
            else:
                conv_stride = int(np.round(1 / stride))

                if not use_torch_fallback:
                    weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)
                    config = Conv2dConfiguration(
                        input_height=input_h,
                        input_width=input_w,
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        batch_size=batch_size,
                        kernel_size=kernel_size,
                        stride=(conv_stride, conv_stride),
                        padding=(0, 0),
                        weight=weight_ttnn,
                        bias=bias_ttnn,
                        activation_dtype=model_config["ACTIVATIONS_DTYPE"],
                        weights_dtype=model_config["WEIGHTS_DTYPE"],
                        output_dtype=model_config["ACTIVATIONS_DTYPE"],
                        math_fidelity=model_config["MATH_FIDELITY"],
                        sharding_strategy=AutoShardedStrategyConfiguration(),
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                        **self.optimizations.conv2d,
                    )
                    self.conv_layers.append(TtConv2d(config, device))
                else:
                    self.conv_layers.append(None)
                self.conv_transpose_configs.append(None)

    def _create_conv_config(self):
        return ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            shard_layout=None,
            deallocate_activation=self.optimizations.conv_transpose.get("deallocate_activation", False),
            reallocate_halo_output=self.optimizations.conv_transpose.get("reallocate_halo_output", False),
            enable_act_double_buffer=self.optimizations.conv_transpose.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=self.optimizations.conv_transpose.get("enable_weights_double_buffer", False),
            output_layout=ttnn.TILE_LAYOUT,
        )

    def _create_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, batch_size=1):
        ups = []
        target_height = None
        target_width = None

        conv_config = self._create_conv_config()
        compute_config = self._create_compute_config()

        for i in range(self.num_levels):
            feat = x[i]
            height, width = feat.shape[1], feat.shape[2]
            stride = self.upsample_strides[i]
            kernel_size = self.deblocks[i].kernel_size
            use_conv_transpose = stride >= 1

            if feat.is_sharded():
                feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
            elif feat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            if feat.layout != ttnn.TILE_LAYOUT:
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)

            if use_conv_transpose:
                int_stride = int(stride)
                conv_out_height = (height - 1) * int_stride + kernel_size[0]
                conv_out_width = (width - 1) * int_stride + kernel_size[1]

                if target_height is None:
                    target_height = conv_out_height
                    target_width = conv_out_width

                config = self.conv_transpose_configs[i]
                feat, [conv_out_height, conv_out_width] = ttnn.conv_transpose2d(
                    input_tensor=feat,
                    weight_tensor=config.weight,
                    bias_tensor=config.bias,
                    device=self.device,
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    batch_size=batch_size,
                    input_height=height,
                    input_width=width,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=(0, 0),
                    output_padding=(0, 0),
                    dilation=(1, 1),
                    groups=1,
                    conv_config=conv_config,
                    compute_config=compute_config,
                    return_output_dim=True,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )

                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                feat = ttnn.relu(feat)
            else:
                conv_stride = int(np.round(1 / stride))
                conv_out_height = (height - kernel_size[0]) // conv_stride + 1
                conv_out_width = (width - kernel_size[1]) // conv_stride + 1

                if target_height is None:
                    target_height = conv_out_height
                    target_width = conv_out_width

                if self.use_torch_fallback:
                    feat_torch = ttnn.to_torch(feat).float()
                    if len(feat_torch.shape) == 4:
                        feat_torch = feat_torch.permute(0, 3, 1, 2).contiguous()
                    weight, bias = self._torch_weights[i]
                    feat_torch = torch.nn.functional.conv2d(
                        feat_torch, weight, bias.squeeze() if bias is not None else None, stride=conv_stride, padding=0
                    )
                    feat_torch = torch.nn.functional.relu(feat_torch)
                    feat_torch = feat_torch.permute(0, 2, 3, 1).contiguous()
                    feat = ttnn.from_torch(
                        feat_torch.to(torch.bfloat16),
                        dtype=self.model_config["ACTIVATIONS_DTYPE"],
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    feat = self.conv_layers[i](feat)
                    if feat.is_sharded():
                        feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                    feat = ttnn.relu(feat)

            if len(feat.shape) == 4:
                if feat.shape[1] == 1 and feat.shape[2] != conv_out_height:
                    feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))
            elif len(feat.shape) == 3:
                feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))

            ups.append(feat)

        processed_ups = []
        for up_tensor in ups:
            if isinstance(up_tensor, ttnn.Tensor):
                if up_tensor.is_sharded():
                    up_tensor = ttnn.sharded_to_interleaved(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
                if up_tensor.layout != ttnn.TILE_LAYOUT:
                    up_tensor = ttnn.to_layout(up_tensor, ttnn.TILE_LAYOUT)
            processed_ups.append(up_tensor)

        if len(processed_ups) > 1:
            out = ttnn.concat(processed_ups, dim=3)
        else:
            out = processed_ups[0]

        return [out]


class SECONDFPN_Head_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels,
        out_channels,
        upsample_strides,
        model_config,
        input_shapes=None,
        batch_size=1,
        use_slicing=True,
        optimizations=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_levels = len(in_channels)
        self.use_slicing = use_slicing
        self.model_config = model_config
        self.batch_size = batch_size
        self.optimizations = optimizations or secondfpn_head_optimizations
        self.deblocks = parameters.deblocks

        if input_shapes is None:
            input_shapes = [(128, 128), (64, 64), (32, 32), (16, 16)]
        self.input_shapes = input_shapes

        self.conv_transpose_configs = []

        for i in range(self.num_levels):
            kernel_size = self.deblocks[i].kernel_size
            stride = int(self.upsample_strides[i])
            input_h, input_w = self.input_shapes[i]

            weight = self.deblocks[i].conv_weight
            bias = self.deblocks[i].conv_bias

            if isinstance(weight, torch.Tensor):
                weight = weight.clone().float()
            elif isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight).float()

            if isinstance(bias, torch.Tensor):
                bias = bias.clone().float()
            elif isinstance(bias, np.ndarray):
                bias = torch.from_numpy(bias).float()

            weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            bias_ttnn = None
            if bias is not None:
                if len(bias.shape) == 1:
                    bias = bias.view(1, 1, 1, -1)
                bias_ttnn = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

            config = ConvTransposeConfig(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_size,
                stride=(stride, stride),
                input_height=input_h,
                input_width=input_w,
                batch_size=batch_size,
                weight=weight_ttnn,
                bias=bias_ttnn,
            )
            self.conv_transpose_configs.append(config)

        logger.info(f"SECONDFPN_Head_TTNN init: {self.num_levels} levels")

    def _create_conv_config(self):
        return ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            shard_layout=None,
            deallocate_activation=self.optimizations.conv_transpose.get("deallocate_activation", False),
            output_layout=ttnn.TILE_LAYOUT,
        )

    def _create_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x_list, batch_size=1):
        ups = []
        target_height = None
        target_width = None

        conv_config = self._create_conv_config()
        compute_config = self._create_compute_config()

        for i in range(self.num_levels):
            feat = x_list[i]
            height, width = feat.shape[1], feat.shape[2]
            stride = int(self.upsample_strides[i])
            kernel_size = self.deblocks[i].kernel_size

            if feat.is_sharded():
                feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
            elif feat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            if feat.layout != ttnn.TILE_LAYOUT:
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)

            conv_out_height = (height - 1) * stride + kernel_size[0]
            conv_out_width = (width - 1) * stride + kernel_size[1]

            l1_estimate = height * width * self.in_channels[i]
            slice_config = None
            if self.use_slicing and l1_estimate > 100000:
                num_slices = 8
                slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices)

            config = self.conv_transpose_configs[i]

            feat, [conv_out_height, conv_out_width] = ttnn.conv_transpose2d(
                input_tensor=feat,
                weight_tensor=config.weight,
                bias_tensor=config.bias,
                device=self.device,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=(0, 0),
                output_padding=(0, 0),
                dilation=(1, 1),
                conv_config=conv_config,
                compute_config=compute_config,
                dram_slice_config=slice_config,
                return_output_dim=True,
                return_weights_and_bias=False,
                mirror_kernel=True,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )

            if feat.is_sharded():
                feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
            feat = ttnn.relu(feat)

            if target_height is None:
                target_height = conv_out_height
                target_width = conv_out_width

            if len(feat.shape) == 4:
                if feat.shape[1] == 1 and feat.shape[2] != conv_out_height:
                    feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))
            elif len(feat.shape) == 3:
                feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))

            ups.append(feat)

        processed_ups = []
        for up_tensor in ups:
            if isinstance(up_tensor, ttnn.Tensor):
                if up_tensor.is_sharded():
                    up_tensor = ttnn.sharded_to_interleaved(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
                if up_tensor.layout != ttnn.TILE_LAYOUT:
                    up_tensor = ttnn.to_layout(up_tensor, ttnn.TILE_LAYOUT)
            processed_ups.append(up_tensor)

        if len(processed_ups) > 1:
            out = ttnn.concat(processed_ups, dim=3)
        else:
            out = processed_ups[0]

        return [out]


def prepare_secondfpn_parameters(
    state_dict,
    in_channels=[256, 512, 1024, 2048],
    out_channels=[128, 128, 128, 128],
    upsample_strides=[0.25, 0.5, 1, 2],
):
    from models.experimental.BevDepth.tests.test_resnet50_backbone import fuse_conv_bn_weights

    class Parameters:
        pass

    params = Parameters()
    params.deblocks = []

    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.backbone.img_neck.",
        "backbone.img_neck.",
        "img_neck.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find SECONDFPN prefix. Available keys: {all_keys[:20]}")
        raise KeyError("No img_neck keys found in checkpoint")

    logger.info(f"Using SECONDFPN prefix: {prefix}")

    for i in range(len(in_channels)):
        deblock = Parameters()
        weight = state_dict[f"{prefix}deblocks.{i}.0.weight"].clone()

        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        deblock.kernel_size = (kernel_h, kernel_w)

        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
        bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

        stride = upsample_strides[i]
        is_transposed = stride >= 1

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            if is_transposed:
                conv_weight_for_fusion = weight.permute(1, 0, 2, 3).contiguous().float()
            else:
                conv_weight_for_fusion = weight.float()

            eps = 1e-3
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight_for_fusion,
                bn_weight.float(),
                bn_bias.float() if bn_bias is not None else torch.zeros_like(bn_weight),
                bn_mean.float(),
                bn_var.float(),
                eps=eps,
            )

            if is_transposed:
                fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()

            deblock.conv_weight = fused_weight.to(torch.bfloat16)
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            deblock.conv_weight = weight.to(torch.bfloat16)
            bias_key = f"{prefix}deblocks.{i}.0.bias"
            if bias_key in state_dict:
                deblock.conv_bias = state_dict[bias_key].to(torch.bfloat16)
            else:
                deblock.conv_bias = None

        params.deblocks.append(deblock)

    logger.info(f"Prepared SECONDFPN parameters for {len(in_channels)} levels")
    return params


def prepare_secondfpn_head_parameters(
    state_dict,
    in_channels=[160, 160, 320, 640],
    out_channels=[64, 64, 64, 64],
    upsample_strides=[1, 2, 4, 8],
):
    from models.experimental.BevDepth.tests.test_resnet50_backbone import fuse_conv_bn_weights

    class Parameters:
        pass

    params = Parameters()
    params.deblocks = []

    all_keys = list(state_dict.keys())
    possible_prefixes = [
        "model.head.neck.",
        "head.neck.",
        "neck.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find head neck prefix. Available keys: {all_keys[:20]}")
        raise KeyError("No head neck keys found in checkpoint")

    logger.info(f"Using head SECONDFPN prefix: {prefix}")

    for i in range(len(in_channels)):
        deblock = Parameters()
        weight = state_dict[f"{prefix}deblocks.{i}.0.weight"].clone()

        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        deblock.kernel_size = (kernel_h, kernel_w)

        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
        bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            conv_weight_for_fusion = weight.permute(1, 0, 2, 3).contiguous().float()
            eps = 1e-3
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight_for_fusion,
                bn_weight.float(),
                bn_bias.float() if bn_bias is not None else torch.zeros_like(bn_weight),
                bn_mean.float(),
                bn_var.float(),
                eps=eps,
            )
            fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()
            deblock.conv_weight = fused_weight.to(torch.bfloat16)
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            deblock.conv_weight = weight.to(torch.bfloat16)
            bias_key = f"{prefix}deblocks.{i}.0.bias"
            if bias_key in state_dict:
                deblock.conv_bias = state_dict[bias_key].to(torch.bfloat16)
            else:
                deblock.conv_bias = None

        params.deblocks.append(deblock)

    logger.info(f"Prepared head SECONDFPN parameters for {len(in_channels)} levels")
    return params
