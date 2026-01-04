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
from models.experimental.BevDepth.reference.bevdepth.layers.heads.deform_conv import (
    DeformConv2dPack,
)


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
            mode="nearest",
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

        # Initialize DCN (Deformable Conv) using reference implementation
        # TODO: Native TTNN implementation pending - https://github.com/tenstorrent/tt-metal/issues/25526
        if hasattr(parameters, "dcn_weight"):
            self.dcn = DeformConv2dPack(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                deform_groups=1,
                bias=False,
            )
            self.dcn.weight.data = parameters.dcn_weight.float()
            if hasattr(parameters, "dcn_conv_offset"):
                self.dcn.conv_offset.weight.data = parameters.dcn_conv_offset.weight.data.float()
                self.dcn.conv_offset.bias.data = parameters.dcn_conv_offset.bias.data.float()
            self.dcn_bias = parameters.dcn_bias.float() if parameters.dcn_bias is not None else None
            self.dcn.eval()
        else:
            self.dcn = None
            self.dcn_bias = None

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

        # DCN (Deformable Conv) - using reference DeformConv2dPack
        # TODO: Native TTNN implementation pending - https://github.com/tenstorrent/tt-metal/issues/25526
        if self.dcn is not None:
            x_torch = ttnn.to_torch(depth)

            if len(x_torch.shape) == 4:
                if x_torch.shape[1] == 1 and x_torch.shape[2] == height * width:
                    x_torch = x_torch.reshape(batch_size, height, width, self.mid_channels)
                elif x_torch.shape[0] == 1 and x_torch.shape[1] == 1:
                    x_torch = x_torch.reshape(batch_size, height, width, self.mid_channels)
                elif x_torch.shape[1] == height and x_torch.shape[2] == width:
                    pass
            elif len(x_torch.shape) == 3:
                x_torch = x_torch.reshape(batch_size, height, width, self.mid_channels)

            x_torch = x_torch.permute(0, 3, 1, 2).contiguous().float()

            with torch.no_grad():
                output = self.dcn(x_torch)

            if self.dcn_bias is not None:
                output = output + self.dcn_bias.view(1, -1, 1, 1)

            out_h, out_w = output.shape[2], output.shape[3]
            output = output.permute(0, 2, 3, 1).contiguous()

            depth = ttnn.from_torch(
                output,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
