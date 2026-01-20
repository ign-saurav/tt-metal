# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from dataclasses import dataclass

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.BevDepth.tt.utils import (
    create_conv2d_config,
    post_process_conv_output,
)


@dataclass
class ConvTransposeConfig:
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    weight: ttnn.Tensor
    bias: ttnn.Tensor


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


class TtSecondFpnBackbone:
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

        self._torch_weights = []
        self._conv_transpose_configs = []

        for i in range(self.num_levels):
            kernel_size = self.deblocks[i].kernel_size
            stride = self.upsample_strides[i]
            use_conv_transpose = stride >= 1

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
                    weight=weight_ttnn,
                    bias=bias_ttnn,
                )
                self._conv_transpose_configs.append(config)
            else:
                self._conv_transpose_configs.append(None)

    def _get_conv(self, level_idx, batch_size, height, width):
        stride = self.upsample_strides[level_idx]
        conv_stride = int(np.round(1 / stride))
        kernel_size = self.deblocks[level_idx].kernel_size
        weight, bias = self._torch_weights[level_idx]

        conv_config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=self.in_channels[level_idx],
            out_channels=self.out_channels[level_idx],
            batch_size=batch_size,
            kernel_size=kernel_size,
            weight=weight,
            bias=bias,
            stride=(conv_stride, conv_stride),
            padding=(0, 0),
            model_config=self.model_config,
            conv_config=self.optimizations.conv2d,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        return TtConv2d(conv_config, self.device)

    def __call__(self, x, batch_size=1):
        ups = []
        target_height = None
        target_width = None

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            shard_layout=None,
            deallocate_activation=self.optimizations.conv_transpose.get("deallocate_activation", False),
            reallocate_halo_output=self.optimizations.conv_transpose.get("reallocate_halo_output", False),
            enable_act_double_buffer=self.optimizations.conv_transpose.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=self.optimizations.conv_transpose.get("enable_weights_double_buffer", False),
            output_layout=ttnn.TILE_LAYOUT,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

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

                config = self._conv_transpose_configs[i]
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

                feat = post_process_conv_output(feat, batch_size, conv_out_height, conv_out_width, self.out_channels[i])
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
                    tt_conv = self._get_conv(i, batch_size, height, width)
                    feat, (conv_out_height, conv_out_width) = tt_conv(feat, return_output_dim=True)
                    feat = post_process_conv_output(
                        feat, batch_size, conv_out_height, conv_out_width, self.out_channels[i]
                    )
                    feat = ttnn.relu(feat)

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


class TtSecondFpnHead:
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

        self._conv_transpose_configs = []

        for i in range(self.num_levels):
            kernel_size = self.deblocks[i].kernel_size
            stride = int(self.upsample_strides[i])

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
                weight=weight_ttnn,
                bias=bias_ttnn,
            )
            self._conv_transpose_configs.append(config)

    def __call__(self, x_list, batch_size=1):
        ups = []
        target_height = None
        target_width = None

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            shard_layout=None,
            deallocate_activation=self.optimizations.conv_transpose.get("deallocate_activation", False),
            output_layout=ttnn.TILE_LAYOUT,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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

            config = self._conv_transpose_configs[i]

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

            feat = post_process_conv_output(feat, batch_size, conv_out_height, conv_out_width, self.out_channels[i])

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
