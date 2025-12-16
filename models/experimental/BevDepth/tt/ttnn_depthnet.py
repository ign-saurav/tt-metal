# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from dataclasses import dataclass
from loguru import logger

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.BevDepth.tt.utils import (
    create_conv2d_config,
    post_process_conv_output,
)
from models.experimental.BevDepth.tt.deformable_conv import TtDeformConv2dPack


@dataclass
class DepthNetOptimizations:
    reduce_conv: dict
    mlp: dict
    se_layer: dict
    basic_block: dict
    aspp: dict
    context_conv: dict
    depth_conv: dict


depthnet_optimizations = DepthNetOptimizations(
    reduce_conv={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "packer_l1_acc": False,
    },
    mlp={},
    se_layer={
        "shard_layout": None,
        "packer_l1_acc": False,
    },
    basic_block={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "packer_l1_acc": False,
    },
    aspp={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "packer_l1_acc": False,
    },
    context_conv={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "packer_l1_acc": False,
    },
    depth_conv={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "packer_l1_acc": False,
    },
)


class MLP_TTNN:
    def __init__(self, device, parameters, in_features, hidden_features, out_features, model_config):
        self.device = device
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.model_config = model_config

        self.fc1_weight = ttnn.from_torch(
            parameters.fc1_weight.T.contiguous(),
            dtype=model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc1_bias = ttnn.from_torch(
            parameters.fc1_bias.unsqueeze(0),
            dtype=model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc2_weight = ttnn.from_torch(
            parameters.fc2_weight.T.contiguous(),
            dtype=model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc2_bias = ttnn.from_torch(
            parameters.fc2_bias.unsqueeze(0),
            dtype=model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias)
        x = ttnn.relu(x)
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias)
        return x


class SELayer_TTNN:
    """Squeeze-and-Excitation Layer using TtConv2d builder API."""

    def __init__(self, device, parameters, channels, model_config):
        self.device = device
        self.channels = channels
        self.model_config = model_config
        self.params = parameters
        self._conv_reduce_cache = {}
        self._conv_expand_cache = {}

    def _get_conv_reduce(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv_reduce_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.channels,
                out_channels=self.channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.params.conv_reduce_weight,
                bias=self.params.conv_reduce_bias,
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=None,
                packer_l1_acc=False,
            )
            self._conv_reduce_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv_reduce_cache[cache_key]

    def _get_conv_expand(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv_expand_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.channels,
                out_channels=self.channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.params.conv_expand_weight,
                bias=self.params.conv_expand_bias,
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                packer_l1_acc=False,
            )
            self._conv_expand_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv_expand_cache[cache_key]

    def __call__(self, x, x_se, batch_size=None, height=None, width=None):
        """
        Forward pass:
        x: TTNN tensor [batch, height, width, channels]
        x_se: TTNN tensor [batch, height, width, channels] (from MLP output broadcasted)
        batch_size, height, width: Optional explicit dimensions to use for reshaping
        """
        if x_se.is_sharded():
            x_se = ttnn.sharded_to_interleaved(x_se, ttnn.DRAM_MEMORY_CONFIG)
        if x_se.layout != ttnn.TILE_LAYOUT:
            x_se = ttnn.to_layout(x_se, ttnn.TILE_LAYOUT)

        # Ensure x is in interleaved DRAM
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Get dimensions from input or explicit parameters
        if batch_size is None or height is None or width is None:
            # Try to infer from x shape
            if len(x.shape) == 4:
                batch_size = x.shape[0]
                height = x.shape[1]
                width = x.shape[2]
                channels = x.shape[3]
            else:
                raise ValueError(f"Cannot infer dimensions from x.shape={x.shape}")
        else:
            channels = self.channels

        # Ensure x has proper shape
        x = post_process_conv_output(x, batch_size, height, width, channels)
        # Ensure x_se has proper shape
        x_se = post_process_conv_output(x_se, batch_size, height, width, channels)

        # conv_reduce: 1x1 conv (channels -> channels) with ReLU
        conv_reduce = self._get_conv_reduce(batch_size, height, width)
        x_se, (out_h, out_w) = conv_reduce(x_se, return_output_dim=True)
        x_se = post_process_conv_output(x_se, batch_size, out_h, out_w, channels)

        # conv_expand: 1x1 conv (channels -> channels)
        conv_expand = self._get_conv_expand(batch_size, height, width)
        x_se, (out_h, out_w) = conv_expand(x_se, return_output_dim=True)
        x_se = post_process_conv_output(x_se, batch_size, out_h, out_w, channels)

        # Apply sigmoid (gate)
        x_se = ttnn.sigmoid(x_se)

        # Ensure both tensors are in DRAM with interleaved memory config and TILE_LAYOUT
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x_se.is_sharded():
            x_se = ttnn.sharded_to_interleaved(x_se, ttnn.DRAM_MEMORY_CONFIG)

        # Ensure both are in TILE_LAYOUT
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if x_se.layout != ttnn.TILE_LAYOUT:
            x_se = ttnn.to_layout(x_se, ttnn.TILE_LAYOUT)

        # Ensure matching memory configs
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x_se = ttnn.to_memory_config(x_se, ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiply: x * gate(x_se)
        result = ttnn.multiply(x, x_se)

        return result


class BasicBlock_TTNN:
    """BasicBlock using TtConv2d builder API."""

    def __init__(self, device, parameters, in_channels, out_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config
        self.params = parameters
        self._conv1_cache = {}
        self._conv2_cache = {}

    def _get_conv1(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                weight=self.params.conv1_weight,
                bias=self.params.conv1_bias,
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._conv1_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv1_cache[cache_key]

    def _get_conv2(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv2_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                weight=self.params.conv2_weight,
                bias=self.params.conv2_bias,
                model_config=self.model_config,
                activation=None,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._conv2_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv2_cache[cache_key]

    def __call__(self, x, batch_size, height, width):
        # Ensure tensor is in interleaved DRAM (not sharded) for stability
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Ensure x is properly shaped before saving as identity
        x = post_process_conv_output(x, batch_size, height, width, self.in_channels)
        identity = x

        # Conv1: 3x3 with ReLU fused
        conv1 = self._get_conv1(batch_size, height, width)
        out, (out_h, out_w) = conv1(x, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, out_h, out_w, self.out_channels)

        # Ensure out is in DRAM before conv2
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            if out.is_sharded():
                out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        # Conv2: 3x3 (no activation)
        conv2 = self._get_conv2(batch_size, height, width)
        out_conv2, (out_h2, out_w2) = conv2(out, return_output_dim=True)
        out_conv2 = post_process_conv_output(out_conv2, batch_size, out_h2, out_w2, self.out_channels)

        # Add + ReLU
        out = ttnn.add(out_conv2, identity)
        out = ttnn.relu(out)

        return out


class ASPP_TTNN:
    """ASPP module using TtConv2d builder API."""

    def __init__(self, device, parameters, in_channels, mid_channels, model_config):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.model_config = model_config
        self.params = parameters
        # Conv caches for each branch
        self._aspp1_cache = {}
        self._aspp2_cache = {}
        self._aspp3_cache = {}
        self._aspp4_cache = {}
        self._global_cache = {}
        self._final_slice_caches = [{}, {}, {}, {}]

    def _get_aspp1(self, batch_size, height, width):
        """Branch 1: 1x1 conv with ReLU."""
        cache_key = (batch_size, height, width)
        if cache_key not in self._aspp1_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=self.params.aspp1_weight,
                bias=self.params.aspp1_bias,
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._aspp1_cache[cache_key] = TtConv2d(config, self.device)
        return self._aspp1_cache[cache_key]

    def _get_dilated_conv(self, cache, batch_size, height, width, dilation, weight, bias):
        """Get dilated conv for ASPP branches 2-4."""
        cache_key = (batch_size, height, width)
        if cache_key not in cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=(dilation, dilation),
                weight=weight,
                bias=bias,
                model_config=self.model_config,
                activation=None,  # Apply ReLU after
                shard_layout=None,
                packer_l1_acc=False,
            )
            cache[cache_key] = TtConv2d(config, self.device)
        return cache[cache_key]

    def _get_global_conv(self, batch_size):
        """Global pooling branch: 1x1 conv."""
        cache_key = (batch_size, 1, 1)
        if cache_key not in self._global_cache:
            config = create_conv2d_config(
                input_height=1,
                input_width=1,
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=self.params.global_weight,
                bias=self.params.global_bias,
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._global_cache[cache_key] = TtConv2d(config, self.device)
        return self._global_cache[cache_key]

    def __call__(self, x, batch_size, height, width):
        import torch

        # Ensure input is in DRAM before conv2d
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Branch 1: 1x1 conv with ReLU
        aspp1_conv = self._get_aspp1(batch_size, height, width)
        x1, (out_h1, out_w1) = aspp1_conv(x, return_output_dim=True)
        x1 = post_process_conv_output(x1, batch_size, out_h1, out_w1, self.mid_channels)

        # Branch 2: 3x3 conv with dilation=6
        aspp2_conv = self._get_dilated_conv(
            self._aspp2_cache, batch_size, height, width, 6, self.params.aspp2_weight, self.params.aspp2_bias
        )
        x2, (out_h2, out_w2) = aspp2_conv(x, return_output_dim=True)
        x2 = post_process_conv_output(x2, batch_size, out_h2, out_w2, self.mid_channels)
        x2 = ttnn.relu(x2)

        # Branch 3: 3x3 conv with dilation=12
        aspp3_conv = self._get_dilated_conv(
            self._aspp3_cache, batch_size, height, width, 12, self.params.aspp3_weight, self.params.aspp3_bias
        )
        x3, (out_h3, out_w3) = aspp3_conv(x, return_output_dim=True)
        x3 = post_process_conv_output(x3, batch_size, out_h3, out_w3, self.mid_channels)
        x3 = ttnn.relu(x3)

        # Branch 4: 3x3 conv with dilation=18
        aspp4_conv = self._get_dilated_conv(
            self._aspp4_cache, batch_size, height, width, 18, self.params.aspp4_weight, self.params.aspp4_bias
        )
        x4, (out_h4, out_w4) = aspp4_conv(x, return_output_dim=True)
        x4 = post_process_conv_output(x4, batch_size, out_h4, out_w4, self.mid_channels)
        x4 = ttnn.relu(x4)

        # Global pooling branch
        x5 = ttnn.global_avg_pool2d(x)
        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)
        if x5.layout != ttnn.TILE_LAYOUT:
            x5 = ttnn.to_layout(x5, ttnn.TILE_LAYOUT)
            if x5.is_sharded():
                x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)

        global_conv = self._get_global_conv(batch_size)
        x5, (out_h5, out_w5) = global_conv(x5, return_output_dim=True)

        # Convert sharded to interleaved before upsample
        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)

        # Reshape x5 to [batch, 1, 1, channels]
        if len(x5.shape) == 4 and x5.shape[0] == 1 and x5.shape[1] == 1:
            if x5.shape[2] == batch_size:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            elif x5.shape[2] == 1:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
        elif len(x5.shape) == 3 and x5.shape[0] == 1:
            if x5.shape[1] == batch_size:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            elif x5.shape[1] == 1:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
        elif len(x5.shape) != 4 or x5.shape[0] != batch_size or x5.shape[1] != 1 or x5.shape[2] != 1:
            expected_elements = batch_size * 1 * 1 * self.mid_channels
            actual_elements = 1
            for dim in x5.shape:
                actual_elements *= dim
            if actual_elements == expected_elements:
                x5 = ttnn.reshape(x5, (batch_size, 1, 1, self.mid_channels))
            else:
                raise RuntimeError(
                    f"Cannot reshape x5: shape={x5.shape}, expected elements={expected_elements}, actual={actual_elements}"
                )

        # Upsample from 1x1 to height x width using TtUpsample with channel slicing
        from models.tt_cnn.tt.builder import TtUpsample, UpsampleConfiguration, ChannelSliceStrategyConfiguration

        if x5.is_sharded():
            x5 = ttnn.sharded_to_interleaved(x5, ttnn.DRAM_MEMORY_CONFIG)
        x5 = ttnn.to_layout(x5, ttnn.ROW_MAJOR_LAYOUT)

        output_bytes = self.mid_channels * height * width * 2
        l1_bank_size = 1363712
        num_slices = max(4, (output_bytes // l1_bank_size) + 1)
        upsample_config = UpsampleConfiguration(
            input_height=1,
            input_width=1,
            channels=self.mid_channels,
            batch_size=batch_size,
            scale_factor=(height, width),
            mode="bilinear",
            slice_strategy=ChannelSliceStrategyConfiguration(num_slices=num_slices),
        )
        upsample_layer = TtUpsample(upsample_config, self.device)
        x5 = upsample_layer(x5)

        x5 = ttnn.to_layout(x5, ttnn.TILE_LAYOUT)
        x5 = ttnn.to_memory_config(x5, ttnn.DRAM_MEMORY_CONFIG)

        # Concatenate all 5 branches
        out = ttnn.concat([x1, x2, x3, x4, x5], dim=-1)

        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.is_allocated() and out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)

        # Final conv - 2560->512, 1x1 kernel: Use channel slicing (4x 640->512)
        num_slices = 4
        channels_per_slice = (self.mid_channels * 5) // num_slices

        out_slices = []
        for i in range(num_slices):
            start_ch = i * channels_per_slice
            end_ch = (i + 1) * channels_per_slice if i < num_slices - 1 else self.mid_channels * 5
            out_slices.append(ttnn.slice(out, [0, 0, 0, start_ch], [batch_size, height, width, end_ch]))

        weight_torch = (
            self.params.conv1_weight
            if isinstance(self.params.conv1_weight, torch.Tensor)
            else ttnn.to_torch(self.params.conv1_weight)
        )
        weight_slices = []
        for i in range(num_slices):
            start_ch = i * channels_per_slice
            end_ch = (i + 1) * channels_per_slice if i < num_slices - 1 else self.mid_channels * 5
            weight_slices.append(weight_torch[:, start_ch:end_ch, :, :])

        # Run each slice separately using TtConv2d
        out_accum = None
        for i in range(num_slices):
            cache_key = (batch_size, height, width)
            if cache_key not in self._final_slice_caches[i]:
                config = create_conv2d_config(
                    input_height=height,
                    input_width=width,
                    in_channels=weight_slices[i].shape[1],
                    out_channels=self.mid_channels,
                    batch_size=batch_size,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    weight=weight_slices[i],
                    bias=None,
                    model_config=self.model_config,
                    activation=None,
                    shard_layout=None,
                    packer_l1_acc=False,
                )
                self._final_slice_caches[i][cache_key] = TtConv2d(config, self.device)

            conv_layer = self._final_slice_caches[i][cache_key]
            out_i, (out_h_i, out_w_i) = conv_layer(out_slices[i], return_output_dim=True)
            out_i = post_process_conv_output(out_i, batch_size, out_h_i, out_w_i, self.mid_channels)

            if out_accum is None:
                out_accum = out_i
            else:
                out_accum = ttnn.add(out_accum, out_i)

            out_slices[i].deallocate(True)

        out = out_accum

        # Apply bias if it exists
        if self.params.conv1_bias is not None:
            bias_ttnn = self.params.conv1_bias
            if isinstance(bias_ttnn, torch.Tensor):
                if len(bias_ttnn.shape) == 1:
                    bias_ttnn = bias_ttnn.view(1, 1, 1, -1)
                bias_ttnn = ttnn.from_torch(
                    bias_ttnn,
                    device=self.device,
                    dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            out = ttnn.add(out, bias_ttnn)

        out = ttnn.relu(out)

        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)

        return out


class DepthNet_TTNN:
    """DepthNet using TtConv2d builder API."""

    def __init__(
        self,
        device,
        parameters,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        model_config,
        optimizations=None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.model_config = model_config
        self.optimizations = optimizations or depthnet_optimizations
        self.params = parameters

        # Conv caches for TtConv2d instances
        self._reduce_slice_caches = [{}, {}]  # 2 slices for reduce_conv
        self._context_conv_cache = {}
        self._dcn_fallback_cache = {}
        self._final_conv_cache = {}

        # Initialize sub-modules
        self.block1 = BasicBlock_TTNN(device, parameters.block1, mid_channels, mid_channels, self.model_config)
        self.block2 = BasicBlock_TTNN(device, parameters.block2, mid_channels, mid_channels, self.model_config)
        self.block3 = BasicBlock_TTNN(device, parameters.block3, mid_channels, mid_channels, self.model_config)
        self.aspp = ASPP_TTNN(device, parameters.aspp, mid_channels, mid_channels, self.model_config)

        # Initialize MLP and SELayer if parameters are available
        # Check actual input size from MLP weights (fc1_weight shape)
        if hasattr(parameters, "depth_mlp") and hasattr(parameters.depth_mlp, "fc1_weight"):
            mlp_in_features = parameters.depth_mlp.fc1_weight.shape[1]  # Get from checkpoint
        else:
            mlp_in_features = 31  # Default: 31 features (15 from stack + 16 from sensor2ego 4x4)

        if hasattr(parameters, "depth_mlp") and hasattr(parameters, "context_mlp"):
            self.depth_mlp = MLP_TTNN(
                device, parameters.depth_mlp, mlp_in_features, mid_channels, mid_channels, self.model_config
            )
            self.context_mlp = MLP_TTNN(
                device, parameters.context_mlp, mlp_in_features, mid_channels, mid_channels, self.model_config
            )
        else:
            self.depth_mlp = None
            self.context_mlp = None

        if hasattr(parameters, "depth_se") and hasattr(parameters, "context_se"):
            self.depth_se = SELayer_TTNN(device, parameters.depth_se, mid_channels, self.model_config)
            self.context_se = SELayer_TTNN(device, parameters.context_se, mid_channels, self.model_config)
        else:
            self.depth_se = None
            self.context_se = None

        # BN for MLP input (27 features: 15 from intrins/ida/bda + 12 from sensor2ego 3x4)
        if hasattr(parameters, "mlp_bn"):
            self.mlp_bn = parameters.mlp_bn
        else:
            self.mlp_bn = None

        # Initialize DCN (Deformable Conv) using wrapper
        # TODO: Native TTNN implementation pending - https://github.com/tenstorrent/tt-metal/issues/25526
        if hasattr(parameters, "dcn_weight"):
            self.dcn = TtDeformConv2dPack(
                device=device,
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                deform_groups=1,
                conv_offset_weight=parameters.dcn_conv_offset.weight.data
                if hasattr(parameters, "dcn_conv_offset")
                else None,
                conv_offset_bias=parameters.dcn_conv_offset.bias.data
                if hasattr(parameters, "dcn_conv_offset")
                else None,
                dcn_weight=parameters.dcn_weight,
                dcn_bias=parameters.dcn_bias,
            )
        else:
            self.dcn = None

        logger.info(f"DepthNet init: in={in_channels}, mid={mid_channels}, depth={depth_channels}")

    def _get_reduce_conv_slice(self, slice_idx, batch_size, height, width, weight_slice):
        """Get cached TtConv2d for reduce_conv slice."""
        cache_key = (batch_size, height, width)
        channels_per_slice = self.in_channels // 2
        if cache_key not in self._reduce_slice_caches[slice_idx]:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=channels_per_slice,
                out_channels=self.mid_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                weight=weight_slice,
                bias=None,
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                packer_l1_acc=False,
            )
            self._reduce_slice_caches[slice_idx][cache_key] = TtConv2d(config, self.device)
        return self._reduce_slice_caches[slice_idx][cache_key]

    def _get_context_conv(self, batch_size, height, width):
        """Get cached TtConv2d for context_conv."""
        cache_key = (batch_size, height, width)
        if cache_key not in self._context_conv_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.mid_channels,
                out_channels=self.context_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=self.params.context_weight,
                bias=self.params.context_bias,
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                packer_l1_acc=False,
            )
            self._context_conv_cache[cache_key] = TtConv2d(config, self.device)
        return self._context_conv_cache[cache_key]

    def _get_dcn_fallback_conv(self, batch_size, height, width):
        """Get cached TtConv2d for DCN fallback."""
        cache_key = (batch_size, height, width)
        if cache_key not in self._dcn_fallback_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                weight=self.params.dcn_weight,
                bias=self.params.dcn_bias,
                model_config=self.model_config,
                activation=None,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._dcn_fallback_cache[cache_key] = TtConv2d(config, self.device)
        return self._dcn_fallback_cache[cache_key]

    def _get_final_conv(self, batch_size, height, width):
        """Get cached TtConv2d for final depth conv."""
        cache_key = (batch_size, height, width)
        if cache_key not in self._final_conv_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.mid_channels,
                out_channels=self.depth_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=self.params.final_weight,
                bias=self.params.final_bias,
                model_config=self.model_config,
                activation=None,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                packer_l1_acc=False,
            )
            self._final_conv_cache[cache_key] = TtConv2d(config, self.device)
        return self._final_conv_cache[cache_key]

    def __call__(self, x, batch_size=1, mats_dict=None):
        """
        Forward pass for DepthNet

        Args:
            x: TTNN tensor [batch, height, width, in_channels]
            batch_size: Batch size
            mats_dict: Optional dict with camera matrices. If None, uses identity matrices.
                Required keys: intrin_mats, ida_mats, sensor2ego_mats, bda_mat
        """
        import torch

        height, width = x.shape[1], x.shape[2]

        # Compute MLP input from camera matrices (for SELayer)
        # If mats_dict is None, use identity matrices (for test compatibility)
        if mats_dict is None:
            # Create identity matrices for test
            num_cams = 1
            intrin_mats = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_cams, 1, 1)
            ida_mats = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_cams, 1, 1)
            # Slice to :3, : to match checkpoint (27 features: 15 from stack + 12 from sensor2ego 3x4)
            sensor2ego_mats = (
                torch.eye(4)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, num_cams, 1, 1)[:, 0:1, ..., :3, :]
            )
            bda_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            intrin_mats = mats_dict["intrin_mats"][:, 0:1, ...]
            ida_mats = mats_dict["ida_mats"][:, 0:1, ...]
            # Use :3, : to match checkpoint (27 features: 15 from stack + 12 from sensor2ego 3x4)
            sensor2ego_mats = mats_dict["sensor2ego_mats"][:, 0:1, ..., :3, :]
            bda_mat = mats_dict["bda_mat"]

        # Compute MLP input following reference implementation
        intrins = intrin_mats[..., :3, :3]  # [B, 1, num_cams, 3, 3]
        actual_batch_size = intrins.shape[0]  # Use actual batch size from mats, not passed batch_size
        num_cams = intrins.shape[2]
        bda = bda_mat.view(actual_batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida_mats[:, 0:1, ..., 0, 0],
                        ida_mats[:, 0:1, ..., 0, 1],
                        ida_mats[:, 0:1, ..., 0, 3],
                        ida_mats[:, 0:1, ..., 1, 0],
                        ida_mats[:, 0:1, ..., 1, 1],
                        ida_mats[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego_mats.view(actual_batch_size, 1, num_cams, -1),  # [B, 1, num_cams, 12] (3x4 matrix)
            ],
            -1,
        )  # [B, 1, num_cams, 27] (15 from stack + 12 from sensor2ego)

        # Apply BN to MLP input
        if self.mlp_bn is not None:
            mlp_input = mlp_input.reshape(-1, mlp_input.shape[-1])  # [B*num_cams, 27]
            # BN: (x - running_mean) / sqrt(running_var + eps) * weight + bias
            if self.mlp_bn.running_mean is not None and self.mlp_bn.running_var is not None:
                mlp_input = (mlp_input - self.mlp_bn.running_mean) / torch.sqrt(
                    self.mlp_bn.running_var + self.mlp_bn.eps
                )
            if self.mlp_bn.weight is not None:
                mlp_input = mlp_input * self.mlp_bn.weight
            if self.mlp_bn.bias is not None:
                mlp_input = mlp_input + self.mlp_bn.bias
            mlp_input = mlp_input.reshape(actual_batch_size, 1, num_cams, -1)  # [B, 1, num_cams, 27]

        mlp_input_flat = mlp_input.reshape(-1, mlp_input.shape[-1])
        if self.depth_mlp is not None:
            depth_se_mlp = self.depth_mlp(mlp_input_flat)
            depth_se_mlp = ttnn.to_torch(depth_se_mlp).view(actual_batch_size, 1, num_cams, -1)
        else:
            depth_se_mlp = None

        if self.context_mlp is not None:
            context_se_mlp = self.context_mlp(mlp_input_flat)
            context_se_mlp = ttnn.to_torch(context_se_mlp).view(actual_batch_size, 1, num_cams, -1)
        else:
            context_se_mlp = None

        if x.is_sharded():
            # Convert sharded to interleaved DRAM
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        elif x.is_allocated() and x.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        num_slices = 2
        channels_per_slice = self.in_channels // num_slices  # 256

        # Split input along channel dimension
        x_slice1 = ttnn.slice(x, [0, 0, 0, 0], [batch_size, height, width, channels_per_slice])
        x_slice2 = ttnn.slice(x, [0, 0, 0, channels_per_slice], [batch_size, height, width, self.in_channels])

        # Split weights: [out_channels, in_channels, kernel_h, kernel_w] = [512, 512, 3, 3]
        weight_torch = (
            self.params.reduce_weight
            if isinstance(self.params.reduce_weight, torch.Tensor)
            else ttnn.to_torch(self.params.reduce_weight)
        )
        weight_slice1_torch = weight_torch[:, 0:channels_per_slice, :, :]  # [512, 256, 3, 3]
        weight_slice2_torch = weight_torch[:, channels_per_slice:, :, :]  # [512, 256, 3, 3]

        # Run each slice separately using TtConv2d - each produces ALL output channels
        conv_slice1 = self._get_reduce_conv_slice(0, batch_size, height, width, weight_slice1_torch)
        out_slice1, (out_h1, out_w1) = conv_slice1(x_slice1, return_output_dim=True)
        out_slice1 = post_process_conv_output(out_slice1, batch_size, out_h1, out_w1, self.mid_channels)

        conv_slice2 = self._get_reduce_conv_slice(1, batch_size, height, width, weight_slice2_torch)
        out_slice2, (out_h2, out_w2) = conv_slice2(x_slice2, return_output_dim=True)
        out_slice2 = post_process_conv_output(out_slice2, batch_size, out_h2, out_w2, self.mid_channels)

        # SUM the outputs (not concatenate) - each output channel depends on all input channels
        x = ttnn.add(out_slice1, out_slice2)

        # Apply bias if it exists
        if self.params.reduce_bias is not None:
            bias_ttnn = self.params.reduce_bias
            if isinstance(bias_ttnn, torch.Tensor):
                if len(bias_ttnn.shape) == 1:
                    bias_ttnn = bias_ttnn.view(1, 1, 1, -1)
                bias_ttnn = ttnn.from_torch(
                    bias_ttnn,
                    device=self.device,
                    dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            x = ttnn.add(x, bias_ttnn)

        # Apply ReLU activation
        x = ttnn.relu(x)

        # Clean up intermediate tensors
        x_slice1.deallocate(True)
        x_slice2.deallocate(True)
        out_slice1.deallocate(True)
        out_slice2.deallocate(True)

        # Ensure x is in correct format
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Reshape flattened tensor to [batch, height, width, channels]
        if len(x.shape) == 4 and x.shape[0] == 1 and x.shape[1] == 1:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
        elif len(x.shape) == 3 and x.shape[0] == 1:
            x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
        elif len(x.shape) != 4 or x.shape[0] != batch_size or x.shape[1] != height or x.shape[2] != width:
            expected_elements = batch_size * height * width * self.mid_channels
            actual_elements = 1
            for dim in x.shape:
                actual_elements *= dim
            if actual_elements == expected_elements:
                x = ttnn.reshape(x, (batch_size, height, width, self.mid_channels))
            else:
                raise RuntimeError(
                    f"Cannot reshape x: shape={x.shape}, expected={expected_elements}, actual={actual_elements}"
                )

        # Ensure tensor is in correct state
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Log PCC after reduce_conv

        # Context branch: Apply SELayer before context_conv
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        # Apply SELayer if available
        if self.context_se is not None and context_se_mlp is not None:
            # context_se_mlp has shape [actual_batch_size, 1, num_cams, mid_channels]
            # We need to expand it to [batch_size, H, W, mid_channels] where batch_size = actual_batch_size * num_cams
            # Reshape: [actual_B, 1, num_cams, C] -> [actual_B * num_cams, C] -> [batch_size, 1, 1, C] -> expand
            actual_B = context_se_mlp.shape[0]
            num_cams_mlp = context_se_mlp.shape[2]
            context_se_flat = context_se_mlp[:, 0, :, :]  # [actual_B, num_cams, C]
            context_se_flat = context_se_flat.reshape(actual_B * num_cams_mlp, -1)  # [actual_B * num_cams, C]
            context_se_torch = context_se_flat.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, C]
            context_se_torch = context_se_torch.expand(batch_size, height, width, self.mid_channels).contiguous()

            # Convert to TTNN tensor
            context_se_ttnn = ttnn.from_torch(
                context_se_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Apply SELayer: x * gate(conv_expand(relu(conv_reduce(x_se))))
            x_context = self.context_se(x, context_se_ttnn, batch_size=batch_size, height=height, width=width)
            # Log PCC after context SELayer
        else:
            x_context = x

        context_conv = self._get_context_conv(batch_size, height, width)
        context, (out_h_ctx, out_w_ctx) = context_conv(x_context, return_output_dim=True)
        context = post_process_conv_output(context, batch_size, out_h_ctx, out_w_ctx, self.context_channels)

        # Depth branch: Apply SELayer before depth_conv
        if x.shape[-1] != self.mid_channels:
            raise RuntimeError(f"Wrong shape before depth SELayer: expected {self.mid_channels}, got {x.shape[-1]}")

        if not x.is_allocated() and x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Apply SELayer if available
        if self.depth_se is not None and depth_se_mlp is not None:
            # depth_se_mlp has shape [actual_batch_size, 1, num_cams, mid_channels]
            # Reshape: [actual_B, 1, num_cams, C] -> [actual_B * num_cams, C] -> expand
            actual_B = depth_se_mlp.shape[0]
            num_cams_mlp = depth_se_mlp.shape[2]
            depth_se_flat = depth_se_mlp[:, 0, :, :]  # [actual_B, num_cams, C]
            depth_se_flat = depth_se_flat.reshape(actual_B * num_cams_mlp, -1)  # [actual_B * num_cams, C]
            depth_se_torch = depth_se_flat.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, C]
            depth_se_torch = depth_se_torch.expand(batch_size, height, width, self.mid_channels).contiguous()

            # Convert to TTNN tensor
            depth_se_ttnn = ttnn.from_torch(
                depth_se_torch,
                device=self.device,
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Apply SELayer: x * gate(conv_expand(relu(conv_reduce(x_se))))
            x_depth = self.depth_se(x, depth_se_ttnn, batch_size=batch_size, height=height, width=width)
            # Log PCC after depth SELayer
        else:
            x_depth = x

        # BasicBlocks
        depth = self.block1(x_depth, batch_size, height, width)
        depth = self.block2(depth, batch_size, height, width)
        depth = self.block3(depth, batch_size, height, width)

        # ASPP
        depth = self.aspp(depth, batch_size, height, width)

        # Ensure depth is in DRAM before DCN conv
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)
            if depth.is_sharded():
                depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)

        # DCN (Deformable Conv) - using TtDeformConv2dPack wrapper
        # TODO: Native TTNN implementation pending - https://github.com/tenstorrent/tt-metal/issues/25526
        if self.dcn is not None:
            depth, _, _ = self.dcn(depth, batch_size, height, width)
        else:
            # Fallback to regular conv if DCN not initialized
            logger.warning("DCN not initialized, using regular Conv2d as approximation")
            dcn_fallback = self._get_dcn_fallback_conv(batch_size, height, width)
            depth, (out_h_dcn, out_w_dcn) = dcn_fallback(depth, return_output_dim=True)
            depth = post_process_conv_output(depth, batch_size, out_h_dcn, out_w_dcn, self.mid_channels)

        # Ensure depth is in DRAM before final conv
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        # Otherwise, assume it's already in DRAM
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)
            if depth.is_sharded():
                depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)

        # Final depth conv - using TtConv2d
        final_conv = self._get_final_conv(batch_size, height, width)
        depth, (out_h_final, out_w_final) = final_conv(depth, return_output_dim=True)
        depth = post_process_conv_output(depth, batch_size, out_h_final, out_w_final, self.depth_channels)

        # Log PCC after final depth conv

        # Convert both tensors to INTERLEAVED DRAM before concat
        # ttnn.concat requires INTERLEAVED layout when inputs are sharded
        if depth.is_sharded():
            depth = ttnn.sharded_to_interleaved(depth, ttnn.DRAM_MEMORY_CONFIG)
        if depth.is_allocated() and depth.memory_config().buffer_type != ttnn.BufferType.DRAM:
            depth = ttnn.to_memory_config(depth, ttnn.DRAM_MEMORY_CONFIG)
        if depth.layout != ttnn.TILE_LAYOUT:
            depth = ttnn.to_layout(depth, ttnn.TILE_LAYOUT)

        if context.is_sharded():
            context = ttnn.sharded_to_interleaved(context, ttnn.DRAM_MEMORY_CONFIG)
        if context.is_allocated() and context.memory_config().buffer_type != ttnn.BufferType.DRAM:
            context = ttnn.to_memory_config(context, ttnn.DRAM_MEMORY_CONFIG)
        if context.layout != ttnn.TILE_LAYOUT:
            context = ttnn.to_layout(context, ttnn.TILE_LAYOUT)

        # Concatenate depth and context (both in INTERLEAVED DRAM)
        out = ttnn.concat([depth, context], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out


def fuse_conv_bn_weights_unified(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Unified function to fuse BatchNorm into conv weights for inference.
    Handles both conv with bias and conv without bias.

    Formula verification:
    - BN(x) = (x - mean) / sqrt(var + eps) * weight + bias
    - For conv without bias: BN(conv(x)) = (conv(x) - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = conv(x) * (bn_weight / sqrt(var + eps)) - mean * (bn_weight / sqrt(var + eps)) + bn_bias
      = conv(x) * scale + (bn_bias - mean * scale)
    - For conv with bias: BN(conv(x) + conv_bias) = ((conv(x) + conv_bias) - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = (conv(x) + conv_bias - mean) / sqrt(var + eps) * bn_weight + bn_bias
      = conv(x) * scale + (conv_bias - mean) * scale + bn_bias
      = conv(x) * scale + bn_bias + (conv_bias - mean) * scale

    Args:
        conv_weight: [out_channels, in_channels, kH, kW]
        conv_bias: [out_channels] or None
        bn_weight: [out_channels] (gamma)
        bn_bias: [out_channels] (beta)
        bn_mean: [out_channels] (running_mean)
        bn_var: [out_channels] (running_var)
        eps: BN epsilon

    Returns:
        fused_weight, fused_bias
    """
    # Ensure all inputs are float32 for precision during fusion
    conv_weight = conv_weight.float() if conv_weight.dtype != torch.float32 else conv_weight
    bn_weight = (
        bn_weight.float() if isinstance(bn_weight, torch.Tensor) and bn_weight.dtype != torch.float32 else bn_weight
    )
    bn_bias = bn_bias.float() if isinstance(bn_bias, torch.Tensor) and bn_bias.dtype != torch.float32 else bn_bias
    bn_mean = bn_mean.float() if isinstance(bn_mean, torch.Tensor) and bn_mean.dtype != torch.float32 else bn_mean
    bn_var = bn_var.float() if isinstance(bn_var, torch.Tensor) and bn_var.dtype != torch.float32 else bn_var

    # Calculate scale factor from BN: scale = bn_weight / sqrt(bn_var + eps)
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std

    # Fuse into conv weight: multiply each output channel by its scale
    # Shape: [out_channels, in_channels, kH, kW] * [out_channels, 1, 1, 1]
    fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

    # Fuse into bias
    # Handle conv_bias: if None, treat as zero
    if conv_bias is not None:
        conv_bias = conv_bias.float() if conv_bias.dtype != torch.float32 else conv_bias
    else:
        # Create zero bias tensor matching conv_weight device
        conv_bias = torch.zeros(conv_weight.shape[0], dtype=torch.float32, device=conv_weight.device)

    # Handle bn_bias: if None, treat as zero
    if bn_bias is not None:
        bn_bias_val = bn_bias
    else:
        bn_bias_val = torch.zeros_like(bn_mean)

    # Fused bias formula: bn_bias + (conv_bias - bn_mean) * scale
    # This works for both cases:
    # - conv_bias = 0: fused_bias = bn_bias - bn_mean * scale (matches standard formula)
    # - conv_bias != 0: fused_bias = bn_bias + (conv_bias - bn_mean) * scale
    fused_bias = bn_bias_val + scale * (conv_bias - bn_mean)

    return fused_weight, fused_bias


def prepare_depthnet_parameters(state_dict, in_channels=512, mid_channels=256, depth_channels=112):
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

    # Reduce conv: reduce_conv.0 (conv) -> reduce_conv.1 (BN) -> reduce_conv.2 (ReLU)
    try:
        reduce_conv_weight = state_dict[f"{prefix}reduce_conv.0.weight"].float()  # Keep in float32 for fusion
        reduce_conv_bias = state_dict.get(f"{prefix}reduce_conv.0.bias", None)

        # Load BN parameters (reduce_conv.1)
        reduce_bn_weight = state_dict.get(f"{prefix}reduce_conv.1.weight", None)
        reduce_bn_bias = state_dict.get(f"{prefix}reduce_conv.1.bias", None)
        reduce_bn_mean = state_dict.get(f"{prefix}reduce_conv.1.running_mean", None)
        reduce_bn_var = state_dict.get(f"{prefix}reduce_conv.1.running_var", None)

        # Fuse BN into reduce_conv
        if reduce_bn_weight is not None and reduce_bn_mean is not None and reduce_bn_var is not None:
            # Get BN eps from state dict if available, otherwise use default
            reduce_bn_eps = state_dict.get(f"{prefix}reduce_conv.1.eps", 1e-5)
            if isinstance(reduce_bn_eps, torch.Tensor):
                reduce_bn_eps = reduce_bn_eps.item()
            # Use unified fusion function
            fused_reduce_weight, fused_reduce_bias = fuse_conv_bn_weights_unified(
                reduce_conv_weight,
                reduce_conv_bias,
                reduce_bn_weight,
                reduce_bn_bias,
                reduce_bn_mean,
                reduce_bn_var,
                eps=reduce_bn_eps,
            )
            params.reduce_weight = fused_reduce_weight.to(torch.bfloat16)
            params.reduce_bias = fused_reduce_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use original weights
            params.reduce_weight = reduce_conv_weight.to(torch.bfloat16)
            params.reduce_bias = reduce_conv_bias.to(torch.bfloat16) if reduce_conv_bias is not None else None
    except KeyError as e:
        logger.error(f"Failed to load reduce_conv: {e}")
        logger.info(f"Available depth_net keys: {[k for k in all_keys if prefix in k][:20]}")
        raise

    # MLP and SELayer for camera-aware features
    # MLP: 27 -> mid_channels -> mid_channels
    params.depth_mlp = Parameters()
    params.depth_mlp.fc1_weight = state_dict[f"{prefix}depth_mlp.fc1.weight"].to(torch.bfloat16)
    params.depth_mlp.fc1_bias = state_dict.get(f"{prefix}depth_mlp.fc1.bias", None)
    if params.depth_mlp.fc1_bias is not None:
        params.depth_mlp.fc1_bias = params.depth_mlp.fc1_bias.to(torch.bfloat16)
    params.depth_mlp.fc2_weight = state_dict[f"{prefix}depth_mlp.fc2.weight"].to(torch.bfloat16)
    params.depth_mlp.fc2_bias = state_dict.get(f"{prefix}depth_mlp.fc2.bias", None)
    if params.depth_mlp.fc2_bias is not None:
        params.depth_mlp.fc2_bias = params.depth_mlp.fc2_bias.to(torch.bfloat16)

    params.context_mlp = Parameters()
    params.context_mlp.fc1_weight = state_dict[f"{prefix}context_mlp.fc1.weight"].to(torch.bfloat16)
    params.context_mlp.fc1_bias = state_dict.get(f"{prefix}context_mlp.fc1.bias", None)
    if params.context_mlp.fc1_bias is not None:
        params.context_mlp.fc1_bias = params.context_mlp.fc1_bias.to(torch.bfloat16)
    params.context_mlp.fc2_weight = state_dict[f"{prefix}context_mlp.fc2.weight"].to(torch.bfloat16)
    params.context_mlp.fc2_bias = state_dict.get(f"{prefix}context_mlp.fc2.bias", None)
    if params.context_mlp.fc2_bias is not None:
        params.context_mlp.fc2_bias = params.context_mlp.fc2_bias.to(torch.bfloat16)

    # SELayer: conv_reduce, conv_expand
    params.depth_se = Parameters()
    params.depth_se.conv_reduce_weight = state_dict[f"{prefix}depth_se.conv_reduce.weight"].to(torch.bfloat16)
    params.depth_se.conv_reduce_bias = state_dict.get(f"{prefix}depth_se.conv_reduce.bias", None)
    if params.depth_se.conv_reduce_bias is not None:
        params.depth_se.conv_reduce_bias = params.depth_se.conv_reduce_bias.to(torch.bfloat16)
    params.depth_se.conv_expand_weight = state_dict[f"{prefix}depth_se.conv_expand.weight"].to(torch.bfloat16)
    params.depth_se.conv_expand_bias = state_dict.get(f"{prefix}depth_se.conv_expand.bias", None)
    if params.depth_se.conv_expand_bias is not None:
        params.depth_se.conv_expand_bias = params.depth_se.conv_expand_bias.to(torch.bfloat16)

    params.context_se = Parameters()
    params.context_se.conv_reduce_weight = state_dict[f"{prefix}context_se.conv_reduce.weight"].to(torch.bfloat16)
    params.context_se.conv_reduce_bias = state_dict.get(f"{prefix}context_se.conv_reduce.bias", None)
    if params.context_se.conv_reduce_bias is not None:
        params.context_se.conv_reduce_bias = params.context_se.conv_reduce_bias.to(torch.bfloat16)
    params.context_se.conv_expand_weight = state_dict[f"{prefix}context_se.conv_expand.weight"].to(torch.bfloat16)
    params.context_se.conv_expand_bias = state_dict.get(f"{prefix}context_se.conv_expand.bias", None)
    if params.context_se.conv_expand_bias is not None:
        params.context_se.conv_expand_bias = params.context_se.conv_expand_bias.to(torch.bfloat16)

    # BN for MLP input (27 features)
    params.mlp_bn = Parameters()
    params.mlp_bn.weight = state_dict.get(f"{prefix}bn.weight", None)
    params.mlp_bn.bias = state_dict.get(f"{prefix}bn.bias", None)
    params.mlp_bn.running_mean = state_dict.get(f"{prefix}bn.running_mean", None)
    params.mlp_bn.running_var = state_dict.get(f"{prefix}bn.running_var", None)
    params.mlp_bn.eps = 1e-5  # Default BN eps

    # Context conv
    params.context_weight = state_dict[f"{prefix}context_conv.weight"].to(torch.bfloat16)
    params.context_bias = state_dict.get(f"{prefix}context_conv.bias", None)
    if params.context_bias is not None:
        params.context_bias = params.context_bias.to(torch.bfloat16)

    # BasicBlocks (depth_conv.0, depth_conv.1, depth_conv.2)
    # BasicBlock structure: conv1 -> norm1 (BN) -> ReLU -> conv2 -> norm2 (BN) -> add -> ReLU
    # Need to fuse BN layers into conv weights

    for i in range(3):
        block = Parameters()

        # Load conv1 weight and BN1 parameters
        conv1_weight = state_dict[f"{prefix}depth_conv.{i}.conv1.weight"].float()  # Keep in float32 for fusion
        conv1_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv1.bias", None)
        if conv1_bias is not None:
            conv1_bias = conv1_bias.float()

        # Load BN1 parameters - checkpoint uses "bn1" not "norm1"
        # Try both formats: "bn1" (checkpoint format) and "norm1" (PyTorch model format)
        bn1_key_weight = f"{prefix}depth_conv.{i}.bn1.weight"
        bn1_key_bias = f"{prefix}depth_conv.{i}.bn1.bias"
        bn1_key_mean = f"{prefix}depth_conv.{i}.bn1.running_mean"
        bn1_key_var = f"{prefix}depth_conv.{i}.bn1.running_var"

        # If bn1 not found, try norm1 (for models that use norm1)
        if bn1_key_weight not in state_dict:
            bn1_key_weight = f"{prefix}depth_conv.{i}.norm1.weight"
            bn1_key_bias = f"{prefix}depth_conv.{i}.norm1.bias"
            bn1_key_mean = f"{prefix}depth_conv.{i}.norm1.running_mean"
            bn1_key_var = f"{prefix}depth_conv.{i}.norm1.running_var"

        bn1_weight = state_dict.get(bn1_key_weight, None)
        bn1_bias = state_dict.get(bn1_key_bias, None)
        bn1_mean = state_dict.get(bn1_key_mean, None)
        bn1_var = state_dict.get(bn1_key_var, None)

        # Debug: Check if BN parameters exist for first block
        if i == 0:
            logger.info(
                f"Block {i} BN1 parameter keys: weight={bn1_key_weight} (exists={bn1_key_weight in state_dict}), "
                f"bias={bn1_key_bias} (exists={bn1_key_bias in state_dict}), "
                f"mean={bn1_key_mean} (exists={bn1_key_mean in state_dict}), "
                f"var={bn1_key_var} (exists={bn1_key_var in state_dict})"
            )
            if bn1_key_weight not in state_dict:
                # Try alternative key formats
                alt_keys = [k for k in state_dict.keys() if f"depth_conv.{i}" in k and ("norm1" in k or "bn1" in k)]
                logger.warning(f"Block {i} BN1 weight key not found! Alternative keys: {alt_keys[:10]}")

        # Store unfused weights and BN parameters for PyTorch fallback
        block.conv1_weight_unfused = conv1_weight.clone()
        block.conv1_bias_unfused = conv1_bias.clone() if conv1_bias is not None else None
        block.norm1_weight = bn1_weight.clone() if bn1_weight is not None else None
        block.norm1_bias = bn1_bias.clone() if bn1_bias is not None else None
        block.norm1_mean = bn1_mean.clone() if bn1_mean is not None else None
        block.norm1_var = bn1_var.clone() if bn1_var is not None else None

        # Debug: Log BN parameter loading for first block
        if i == 0:
            logger.info(
                f"Block {i} loaded BN1: weight={bn1_weight is not None}, bias={bn1_bias is not None}, "
                f"mean={bn1_mean is not None}, var={bn1_var is not None}"
            )

        # Fuse BN1 into conv1
        if bn1_weight is not None and bn1_mean is not None and bn1_var is not None:
            # Debug: Log BN parameters for first block
            if i == 0:
                bn1_weight_norm = bn1_weight.norm().item() if bn1_weight is not None else 0.0
                bn1_bias_norm = bn1_bias.norm().item() if bn1_bias is not None else 0.0
                bn1_mean_norm = bn1_mean.norm().item() if bn1_mean is not None else 0.0
                bn1_var_norm = bn1_var.norm().item() if bn1_var is not None else 0.0
                logger.info(
                    f"Block {i} BN1 params: weight_norm={bn1_weight_norm:.6f}, "
                    f"bias_norm={bn1_bias_norm:.6f}, "
                    f"mean_norm={bn1_mean_norm:.6f}, "
                    f"var_norm={bn1_var_norm:.6f}"
                )
            # Get BN eps from state dict if available, otherwise use default
            # Try both bn1 and norm1 formats
            bn1_eps = state_dict.get(f"{prefix}depth_conv.{i}.bn1.eps", None)
            if bn1_eps is None:
                bn1_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm1.eps", 1e-5)
            if isinstance(bn1_eps, torch.Tensor):
                bn1_eps = bn1_eps.item()
            # Use unified fusion function (handles conv with or without bias)
            fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
                conv1_weight,
                conv1_bias,
                bn1_weight,
                bn1_bias,
                bn1_mean,
                bn1_var,
                eps=bn1_eps,
            )
            # Debug: Verify fusion for first block
            if i == 0:
                logger.info(
                    f"Block {i} conv1 fusion (float32): weight_norm={fused_conv1_weight.norm().item():.6f}, "
                    f"bias_norm={fused_conv1_bias.norm().item():.6f}, bias_mean={fused_conv1_bias.mean().item():.6f}, "
                    f"bias_min={fused_conv1_bias.min().item():.6f}, bias_max={fused_conv1_bias.max().item():.6f}"
                )
            block.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
            block.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
            # Debug: Check if bias is lost in conversion
            if i == 0:
                bias_after_convert = block.conv1_bias
                logger.info(
                    f"Block {i} conv1_bias immediately after assignment: type={type(bias_after_convert)}, "
                    f"is_none={bias_after_convert is None}"
                )
                if isinstance(bias_after_convert, torch.Tensor):
                    logger.info(
                        f"Block {i} conv1 bias after bfloat16 conversion: norm={bias_after_convert.float().norm().item():.6f}, "
                        f"mean={bias_after_convert.float().mean().item():.6f}, "
                        f"min={bias_after_convert.float().min().item():.6f}, max={bias_after_convert.float().max().item():.6f}"
                    )
                else:
                    logger.warning(f"Block {i} conv1_bias is not a tensor after conversion: {type(bias_after_convert)}")
        else:
            # No BN to fuse, use original weights
            if i == 0:
                logger.warning(
                    f"Block {i} BN1 fusion skipped: bn1_weight={bn1_weight is not None}, "
                    f"bn1_mean={bn1_mean is not None}, bn1_var={bn1_var is not None}"
                )
            block.conv1_weight = conv1_weight.to(torch.bfloat16)
            block.conv1_bias = conv1_bias.to(torch.bfloat16) if conv1_bias is not None else None

        # Load conv2 weight and BN2 parameters
        conv2_weight = state_dict[f"{prefix}depth_conv.{i}.conv2.weight"].float()  # Keep in float32 for fusion
        conv2_bias = state_dict.get(f"{prefix}depth_conv.{i}.conv2.bias", None)
        if conv2_bias is not None:
            conv2_bias = conv2_bias.float()

        # Load BN2 parameters - checkpoint uses "bn2" not "norm2"
        # Try both formats: "bn2" (checkpoint format) and "norm2" (PyTorch model format)
        bn2_key_weight = f"{prefix}depth_conv.{i}.bn2.weight"
        bn2_key_bias = f"{prefix}depth_conv.{i}.bn2.bias"
        bn2_key_mean = f"{prefix}depth_conv.{i}.bn2.running_mean"
        bn2_key_var = f"{prefix}depth_conv.{i}.bn2.running_var"

        # If bn2 not found, try norm2 (for models that use norm2)
        if bn2_key_weight not in state_dict:
            bn2_key_weight = f"{prefix}depth_conv.{i}.norm2.weight"
            bn2_key_bias = f"{prefix}depth_conv.{i}.norm2.bias"
            bn2_key_mean = f"{prefix}depth_conv.{i}.norm2.running_mean"
            bn2_key_var = f"{prefix}depth_conv.{i}.norm2.running_var"

        bn2_weight = state_dict.get(bn2_key_weight, None)
        bn2_bias = state_dict.get(bn2_key_bias, None)
        bn2_mean = state_dict.get(bn2_key_mean, None)
        bn2_var = state_dict.get(bn2_key_var, None)

        # Store unfused weights and BN parameters for PyTorch fallback
        block.conv2_weight_unfused = conv2_weight.clone()
        block.conv2_bias_unfused = conv2_bias.clone() if conv2_bias is not None else None
        block.norm2_weight = bn2_weight.clone() if bn2_weight is not None else None
        block.norm2_bias = bn2_bias.clone() if bn2_bias is not None else None
        block.norm2_mean = bn2_mean.clone() if bn2_mean is not None else None
        block.norm2_var = bn2_var.clone() if bn2_var is not None else None

        # Fuse BN2 into conv2
        if bn2_weight is not None and bn2_mean is not None and bn2_var is not None:
            # Get BN eps from state dict if available, otherwise use default
            # Try both bn2 and norm2 formats
            bn2_eps = state_dict.get(f"{prefix}depth_conv.{i}.bn2.eps", None)
            if bn2_eps is None:
                bn2_eps = state_dict.get(f"{prefix}depth_conv.{i}.norm2.eps", 1e-5)
            if isinstance(bn2_eps, torch.Tensor):
                bn2_eps = bn2_eps.item()
            # Use unified fusion function (handles conv with or without bias)
            fused_conv2_weight, fused_conv2_bias = fuse_conv_bn_weights_unified(
                conv2_weight,
                conv2_bias,
                bn2_weight,
                bn2_bias,
                bn2_mean,
                bn2_var,
                eps=bn2_eps,
            )
            block.conv2_weight = fused_conv2_weight.to(torch.bfloat16)
            block.conv2_bias = fused_conv2_bias.to(torch.bfloat16)
        else:
            # No BN to fuse, use original weights
            block.conv2_weight = conv2_weight.to(torch.bfloat16)
            block.conv2_bias = conv2_bias.to(torch.bfloat16) if conv2_bias is not None else None

        setattr(params, f"block{i+1}", block)
        # Debug: Verify block1 parameters are stored correctly
        if i == 0:
            stored_block = getattr(params, f"block{i+1}", None)
            if stored_block is not None:
                logger.info(
                    f"Block {i} stored in params.block{i+1}: conv1_weight type={type(stored_block.conv1_weight)}, "
                    f"conv1_bias type={type(stored_block.conv1_bias)}, conv1_bias is_none={stored_block.conv1_bias is None}"
                )
                if stored_block.conv1_bias is not None:
                    if hasattr(stored_block.conv1_bias, "float"):
                        bias_norm = stored_block.conv1_bias.float().norm().item()
                        logger.info(f"Block {i} stored conv1_bias: norm={bias_norm:.6f}")
                    else:
                        logger.info(
                            f"Block {i} stored conv1_bias: type={type(stored_block.conv1_bias)} (cannot compute norm)"
                        )
            else:
                logger.warning(f"Block {i} not found in params.block{i+1}!")

    # ASPP (depth_conv.3)
    # ASPP structure: Each branch has atrous_conv -> bn -> relu, final conv1 -> bn1 -> relu
    params.aspp = Parameters()

    # Fuse BN for aspp1-aspp4 branches
    for branch_idx, branch_name in enumerate(["aspp1", "aspp2", "aspp3", "aspp4"], 1):
        atrous_weight = state_dict[f"{prefix}depth_conv.3.{branch_name}.atrous_conv.weight"].float()
        # Load BN parameters - checkpoint uses "bn" format
        bn_key_weight = f"{prefix}depth_conv.3.{branch_name}.bn.weight"
        bn_key_bias = f"{prefix}depth_conv.3.{branch_name}.bn.bias"
        bn_key_mean = f"{prefix}depth_conv.3.{branch_name}.bn.running_mean"
        bn_key_var = f"{prefix}depth_conv.3.{branch_name}.bn.running_var"

        bn_weight = state_dict.get(bn_key_weight, None)
        bn_bias = state_dict.get(bn_key_bias, None)
        bn_mean = state_dict.get(bn_key_mean, None)
        bn_var = state_dict.get(bn_key_var, None)

        # Debug: Check if BN parameters exist for first branch
        if branch_idx == 1:
            logger.info(
                f"ASPP {branch_name} BN parameter keys: weight={bn_key_weight} (exists={bn_key_weight in state_dict}), "
                f"bias={bn_key_bias} (exists={bn_key_bias in state_dict}), "
                f"mean={bn_key_mean} (exists={bn_key_mean in state_dict}), "
                f"var={bn_key_var} (exists={bn_key_var in state_dict})"
            )
            if bn_key_weight not in state_dict:
                # Try alternative key formats
                alt_keys = [
                    k for k in state_dict.keys() if f"depth_conv.3.{branch_name}" in k and ("bn" in k or "norm" in k)
                ]
                logger.warning(f"ASPP {branch_name} BN weight key not found! Alternative keys: {alt_keys[:10]}")

        if bn_weight is not None and bn_mean is not None and bn_var is not None:
            # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
            bn_eps = 1e-5
            # Atrous conv has bias=False
            fused_weight, fused_bias = fuse_conv_bn_weights_unified(
                atrous_weight,
                None,  # No conv bias
                bn_weight,
                bn_bias,
                bn_mean,
                bn_var,
                eps=bn_eps,
            )
            setattr(params.aspp, f"{branch_name}_weight", fused_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", fused_bias.to(torch.bfloat16))
        else:
            setattr(params.aspp, f"{branch_name}_weight", atrous_weight.to(torch.bfloat16))
            setattr(params.aspp, f"{branch_name}_bias", None)

    # Fuse BN for global_avg_pool (conv -> bn -> relu)
    global_weight = state_dict[f"{prefix}depth_conv.3.global_avg_pool.1.weight"].float()
    global_bn_weight = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.weight", None)
    global_bn_bias = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.bias", None)
    global_bn_mean = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_mean", None)
    global_bn_var = state_dict.get(f"{prefix}depth_conv.3.global_avg_pool.2.running_var", None)

    if global_bn_weight is not None and global_bn_mean is not None and global_bn_var is not None:
        # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
        global_bn_eps = 1e-5
        # Global avg pool conv has bias=False
        fused_global_weight, fused_global_bias = fuse_conv_bn_weights_unified(
            global_weight,
            None,  # No conv bias
            global_bn_weight,
            global_bn_bias,
            global_bn_mean,
            global_bn_var,
            eps=global_bn_eps,
        )
        params.aspp.global_weight = fused_global_weight.to(torch.bfloat16)
        params.aspp.global_bias = fused_global_bias.to(torch.bfloat16)
    else:
        params.aspp.global_weight = global_weight.to(torch.bfloat16)
        params.aspp.global_bias = None

    # Fuse BN for final conv1 (conv1 -> bn1 -> relu)
    conv1_weight = state_dict[f"{prefix}depth_conv.3.conv1.weight"].float()
    conv1_bn_weight = state_dict.get(f"{prefix}depth_conv.3.bn1.weight", None)
    conv1_bn_bias = state_dict.get(f"{prefix}depth_conv.3.bn1.bias", None)
    conv1_bn_mean = state_dict.get(f"{prefix}depth_conv.3.bn1.running_mean", None)
    conv1_bn_var = state_dict.get(f"{prefix}depth_conv.3.bn1.running_var", None)

    if conv1_bn_weight is not None and conv1_bn_mean is not None and conv1_bn_var is not None:
        # BN eps is not in state dict, use default 1e-5 for PyTorch BatchNorm2d
        conv1_bn_eps = 1e-5
        # ASPP final conv1 has bias=False
        fused_conv1_weight, fused_conv1_bias = fuse_conv_bn_weights_unified(
            conv1_weight,
            None,  # No conv bias
            conv1_bn_weight,
            conv1_bn_bias,
            conv1_bn_mean,
            conv1_bn_var,
            eps=conv1_bn_eps,
        )
        params.aspp.conv1_weight = fused_conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = fused_conv1_bias.to(torch.bfloat16)
    else:
        params.aspp.conv1_weight = conv1_weight.to(torch.bfloat16)
        params.aspp.conv1_bias = None

    # DCN layer (depth_conv.4) - DeformConv2dPack has both weight and conv_offset
    params.dcn_weight = state_dict[f"{prefix}depth_conv.4.weight"].to(torch.bfloat16)
    params.dcn_bias = state_dict.get(f"{prefix}depth_conv.4.bias", None)
    if params.dcn_bias is not None:
        params.dcn_bias = params.dcn_bias.to(torch.bfloat16)

    # Load conv_offset layer (for DeformConv2dPack)
    # Offset shape: (deform_groups * 2 * kernel_size[0] * kernel_size[1], in_channels, kernel_size[0], kernel_size[1])
    # For DCN with groups=4, kernel=3: offset_channels = 1 * 2 * 3 * 3 = 18
    # But DeformConv2dPack uses deform_groups=1 by default, so offset_channels = 1 * 2 * 3 * 3 = 18
    try:
        conv_offset_weight = state_dict[f"{prefix}depth_conv.4.conv_offset.weight"]
        conv_offset_bias = state_dict.get(f"{prefix}depth_conv.4.conv_offset.bias", None)

        # Create a PyTorch Conv2d layer for offset generation
        # This will be used to generate offsets from input features
        offset_channels = conv_offset_weight.shape[0]  # Should be 18 for kernel=3, deform_groups=1
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_offset_bias is not None,
        )
        params.dcn_conv_offset.weight.data = conv_offset_weight
        if conv_offset_bias is not None:
            params.dcn_conv_offset.bias.data = conv_offset_bias
        params.dcn_conv_offset.eval()  # Set to eval mode
        logger.info(f"Loaded DCN conv_offset layer: {offset_channels} offset channels")
    except KeyError:
        logger.warning(f"conv_offset not found for depth_conv.4, DCN will use zero offsets (may reduce accuracy)")
        # Create a dummy conv_offset that outputs zeros
        offset_channels = 18  # 2 * 3 * 3 for kernel=3
        params.dcn_conv_offset = torch.nn.Conv2d(
            mid_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        params.dcn_conv_offset.weight.data.zero_()
        params.dcn_conv_offset.bias.data.zero_()
        params.dcn_conv_offset.eval()

    # Final conv (depth_conv.5)
    params.final_weight = state_dict[f"{prefix}depth_conv.5.weight"].to(torch.bfloat16)
    params.final_bias = state_dict.get(f"{prefix}depth_conv.5.bias", None)
    if params.final_bias is not None:
        params.final_bias = params.final_bias.to(torch.bfloat16)

    logger.info("Successfully prepared DepthNet parameters")
    return params
