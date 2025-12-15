# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from loguru import logger


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
        "output_layout": ttnn.TILE_LAYOUT,
    },
)


class SECONDFPN_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels,
        out_channels,
        upsample_strides,
        model_config,
        use_torch_conv2d_fallback=False,
        optimizations=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_levels = len(in_channels)
        self.use_torch_conv2d_fallback = use_torch_conv2d_fallback
        self.model_config = model_config
        self.optimizations = optimizations or secondfpn_optimizations

        self.deblocks = parameters.deblocks
        self._original_weights = []
        for i in range(self.num_levels):
            w = self.deblocks[i].conv_weight.detach().float().cpu().numpy().copy()
            b = (
                self.deblocks[i].conv_bias.detach().float().cpu().numpy().copy()
                if self.deblocks[i].conv_bias is not None
                else None
            )
            self._original_weights.append((w, b))

    def __call__(self, x, batch_size=1):
        ups = []
        target_height = None
        target_width = None

        for i in range(self.num_levels):
            feat = x[i]
            height, width = feat.shape[1], feat.shape[2]
            stride = self.upsample_strides[i]
            kernel_size = self.deblocks[i].kernel_size

            # stride >= 1 uses ConvTranspose2d, stride < 1 uses Conv2d
            use_conv_transpose = stride >= 1

            if feat.is_sharded():
                feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
            elif feat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            if feat.layout != ttnn.TILE_LAYOUT:
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)

            # Create fresh weight tensors each call from stored numpy arrays
            orig_weight_np, orig_bias_np = self._original_weights[i]
            weight_tensor = ttnn.from_torch(
                torch.from_numpy(orig_weight_np.copy()).to(torch.bfloat16),
                dtype=self.model_config["WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            bias_tensor = None
            if orig_bias_np is not None:
                b = torch.from_numpy(orig_bias_np.copy()).to(torch.bfloat16)
                if len(b.shape) == 1:
                    b = b.view(1, 1, 1, -1)
                bias_tensor = ttnn.from_torch(
                    b,
                    dtype=self.model_config["WEIGHTS_DTYPE"],
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

            conv_config = ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=None,
                deallocate_activation=False,
                reallocate_halo_output=False,
                enable_act_double_buffer=False,
                enable_weights_double_buffer=False,
                output_layout=ttnn.TILE_LAYOUT,
            )

            compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            if use_conv_transpose:
                # TTNN conv_transpose2d path
                int_stride = int(stride)
                conv_out_height = (height - 1) * int_stride + kernel_size[0]
                conv_out_width = (width - 1) * int_stride + kernel_size[1]

                if target_height is None:
                    target_height = conv_out_height
                    target_width = conv_out_width

                feat, [conv_out_height, conv_out_width] = ttnn.conv_transpose2d(
                    input_tensor=feat,
                    weight_tensor=weight_tensor,
                    bias_tensor=bias_tensor,
                    device=self.device,
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    batch_size=batch_size,
                    input_height=height,
                    input_width=width,
                    kernel_size=kernel_size,
                    stride=(int_stride, int_stride),
                    padding=(0, 0),
                    output_padding=(0, 0),
                    dilation=(1, 1),
                    conv_config=conv_config,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                    mirror_kernel=True,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )

                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                feat = ttnn.relu(feat)

            else:
                # Conv2d path (stride < 1)
                conv_stride = int(np.round(1 / stride)) if stride < 1 else 1
                conv_out_height = (height - kernel_size[0]) // conv_stride + 1
                conv_out_width = (width - kernel_size[1]) // conv_stride + 1

                if target_height is None:
                    target_height = conv_out_height
                    target_width = conv_out_width

                kernel_equals_stride = kernel_size[0] == conv_stride and kernel_size[1] == conv_stride
                use_pytorch_conv2d = kernel_equals_stride and self.use_torch_conv2d_fallback

                if use_pytorch_conv2d:
                    if isinstance(feat, ttnn.Tensor):
                        feat_torch = ttnn.to_torch(feat)
                        if len(feat_torch.shape) == 4:
                            feat_torch = feat_torch.permute(0, 3, 1, 2).contiguous()
                    else:
                        feat_torch = feat

                    weight = self.deblocks[i].conv_weight
                    bias = self.deblocks[i].conv_bias

                    weight = weight.float()
                    if bias is not None:
                        bias = bias.float()
                        if len(bias.shape) > 1:
                            bias = bias.squeeze()

                    out = F.conv2d(
                        feat_torch.float(),
                        weight,
                        bias,
                        stride=conv_stride,
                        padding=0,
                    )
                    out = F.relu(out)

                    out_nhwc = out.permute(0, 2, 3, 1).contiguous()
                    feat = ttnn.from_torch(
                        out_nhwc.to(torch.bfloat16),
                        dtype=self.model_config["ACTIVATIONS_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    conv_out_height = out.shape[2]
                    conv_out_width = out.shape[3]
                else:
                    result = ttnn.conv2d(
                        input_tensor=feat,
                        weight_tensor=weight_tensor,
                        bias_tensor=bias_tensor,
                        device=self.device,
                        in_channels=self.in_channels[i],
                        out_channels=self.out_channels[i],
                        batch_size=batch_size,
                        input_height=height,
                        input_width=width,
                        kernel_size=kernel_size,
                        stride=(conv_stride, conv_stride),
                        padding=(0, 0),
                        conv_config=conv_config,
                        compute_config=compute_config,
                        return_output_dim=True,
                        return_weights_and_bias=False,
                        dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    )
                    feat = result[0]
                    conv_out_height = result[1][0]
                    conv_out_width = result[1][1]

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
            if isinstance(up_tensor, ttnn.Tensor) and up_tensor.is_sharded():
                up_tensor = ttnn.sharded_to_interleaved(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
            processed_ups.append(up_tensor)

        out = ttnn.concat(processed_ups, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        "img_backbone.img_neck.",
        "backbone.img_neck.",
        "img_neck.",
    ]

    prefix = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in all_keys):
            prefix = p
            break

    if prefix is None:
        logger.error(f"Could not find img_neck prefix. Available keys: {all_keys[:10]}")
        raise KeyError("No img_neck keys found in checkpoint")

    logger.info(f"Using SECONDFPN prefix: {prefix}")

    for i in range(len(in_channels)):
        deblock = Parameters()
        weight = state_dict[f"{prefix}deblocks.{i}.0.weight"].clone()

        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        deblock.kernel_size = (kernel_h, kernel_w)

        stride = upsample_strides[i]
        is_conv_transpose = stride >= 1

        bn_weight = state_dict.get(f"{prefix}deblocks.{i}.1.weight", None)
        bn_bias = state_dict.get(f"{prefix}deblocks.{i}.1.bias", None)
        bn_mean = state_dict.get(f"{prefix}deblocks.{i}.1.running_mean", None)
        bn_var = state_dict.get(f"{prefix}deblocks.{i}.1.running_var", None)

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            if is_conv_transpose:
                conv_weight_for_fusion = weight.permute(1, 0, 2, 3).contiguous().float()
            else:
                conv_weight_for_fusion = weight.float()

            eps = 1e-3
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_weight_for_fusion,
                bn_weight.float(),
                bn_bias.float(),
                bn_mean.float(),
                bn_var.float(),
                eps=eps,
            )

            if is_conv_transpose:
                fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()

            deblock.conv_weight = fused_weight.to(torch.bfloat16)
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            deblock.conv_weight = weight.to(torch.bfloat16)
            deblock.conv_bias = None

        params.deblocks.append(deblock)

    logger.info(f"Prepared SECONDFPN parameters for {len(in_channels)} levels")
    return params


class SECONDFPN_Head_TTNN:
    def __init__(
        self,
        device,
        parameters,
        in_channels,
        out_channels,
        upsample_strides,
        model_config,
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
        self.optimizations = optimizations or secondfpn_head_optimizations
        self.deblocks = parameters.deblocks
        logger.info(f"SECONDFPN_Head_TTNN init: {self.num_levels} levels")

    def __call__(self, x_list, batch_size=1):
        ups = []
        target_height = None
        target_width = None

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

            weight_tensor = self.deblocks[i].conv_weight
            if isinstance(weight_tensor, torch.Tensor):
                weight_tensor = ttnn.from_torch(
                    weight_tensor.clone(),
                    dtype=self.model_config["WEIGHTS_DTYPE"],
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

            bias_tensor = self.deblocks[i].conv_bias
            if bias_tensor is not None and isinstance(bias_tensor, torch.Tensor):
                if len(bias_tensor.shape) == 1:
                    bias_tensor = bias_tensor.view(1, 1, 1, -1)
                bias_tensor = ttnn.from_torch(
                    bias_tensor.clone(),
                    dtype=self.model_config["WEIGHTS_DTYPE"],
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

            conv_config = ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=None,
                deallocate_activation=False,
                output_layout=ttnn.TILE_LAYOUT,
            )

            compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            # Use DRAM slicing for large inputs to avoid L1 overflow
            l1_estimate = height * width * self.in_channels[i]
            slice_config = None
            # use slicing if the l1 estimate is greater than 100000 elements
            if self.use_slicing and l1_estimate > 100000:
                num_slices = 8
                slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices)

            conv_out_height = (height - 1) * stride + kernel_size[0]
            conv_out_width = (width - 1) * stride + kernel_size[1]

            feat, [conv_out_height, conv_out_width] = ttnn.conv_transpose2d(
                input_tensor=feat,
                weight_tensor=weight_tensor,
                bias_tensor=bias_tensor,
                device=self.device,
                in_channels=self.in_channels[i],
                out_channels=self.out_channels[i],
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                kernel_size=kernel_size,
                stride=(stride, stride),
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

            if len(feat.shape) == 4:
                if feat.shape[1] == 1 and feat.shape[2] != conv_out_height:
                    feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))
            elif len(feat.shape) == 3:
                feat = ttnn.reshape(feat, (batch_size, conv_out_height, conv_out_width, self.out_channels[i]))

            if target_height is None:
                target_height = conv_out_height
                target_width = conv_out_width

            ups.append(feat)

        processed_ups = []
        for up_tensor in ups:
            if isinstance(up_tensor, ttnn.Tensor) and up_tensor.is_sharded():
                up_tensor = ttnn.sharded_to_interleaved(up_tensor, ttnn.DRAM_MEMORY_CONFIG)
            processed_ups.append(up_tensor)

        out = ttnn.concat(processed_ups, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out


def prepare_secondfpn_head_parameters(
    state_dict,
    in_channels=[160, 160, 320, 640],
    out_channels=[64, 64, 64, 64],
    upsample_strides=[1, 2, 4, 8],
):
    """Prepare parameters for head's SecondFPN (bev_neck)."""
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
                bn_bias.float(),
                bn_mean.float(),
                bn_var.float(),
                eps=eps,
            )
            fused_weight = fused_weight.permute(1, 0, 2, 3).contiguous()
            deblock.conv_weight = fused_weight.to(torch.bfloat16)
            deblock.conv_bias = fused_bias.to(torch.bfloat16)
        else:
            deblock.conv_weight = weight.to(torch.bfloat16)
            deblock.conv_bias = None

        params.deblocks.append(deblock)

    logger.info(f"Prepared head SECONDFPN parameters for {len(in_channels)} levels")
    return params
