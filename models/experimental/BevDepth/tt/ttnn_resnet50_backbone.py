# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass

from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from models.experimental.BevDepth.tt.utils import (
    create_conv2d_config,
    create_maxpool_config,
    post_process_conv_output,
    ensure_memory_config,
)


@dataclass
class ResNet50Optimizations:
    conv1_7x7: dict
    bottleneck_1x1_first: dict
    bottleneck_3x3: dict
    bottleneck_1x1_last: dict
    downsample_1x1: dict


resnet50_optimizations = ResNet50Optimizations(
    conv1_7x7={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "shard_layout": None,  # Auto sharding for memory efficiency
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_first={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "shard_layout": None,  # Auto sharding for memory efficiency
        "deallocate_activation": False,  # Keep input for downsample path
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_3x3={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "shard_layout": None,  # Auto sharding for memory efficiency
        "deallocate_activation": False,
        "reallocate_halo_output": True,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_last={
        "activation": None,
        "shard_layout": None,  # Auto sharding for memory efficiency
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    downsample_1x1={
        "activation": None,
        "shard_layout": None,  # Auto sharding for memory efficiency
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)


class Bottleneck:
    """
    ResNet50 Bottleneck block using TtConv2d builder API directly.
    """

    expansion = 4

    def __init__(
        self, parameters, in_channels, out_channels, stride=1, downsample=None, model_config=None, optimizations=None
    ):
        self.stride = stride
        self.has_downsample = downsample is not None
        self.model_config = model_config
        self.optimizations = optimizations or resnet50_optimizations

        # Store weights as PyTorch tensors for lazy conversion
        self.conv1_weight = parameters.conv1.weight
        self.conv1_bias = parameters.conv1.bias if hasattr(parameters.conv1, "bias") else None

        self.conv2_weight = parameters.conv2.weight
        self.conv2_bias = parameters.conv2.bias if hasattr(parameters.conv2, "bias") else None

        self.conv3_weight = parameters.conv3.weight
        self.conv3_bias = parameters.conv3.bias if hasattr(parameters.conv3, "bias") else None

        if self.has_downsample:
            self.downsample_weight = parameters.downsample[0].weight
            self.downsample_bias = parameters.downsample[0].bias if hasattr(parameters.downsample[0], "bias") else None
        else:
            self.downsample_weight = None
            self.downsample_bias = None

        # Cache for TtConv2d instances (keyed by input dimensions)
        self._conv1_cache = {}
        self._conv2_cache = {}
        self._conv3_cache = {}
        self._downsample_cache = {}

    def _get_conv1(self, device, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            conv_config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.conv1_weight.shape[1],
                out_channels=self.conv1_weight.shape[0],
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                model_config=self.model_config,
                conv_config=self.optimizations.bottleneck_1x1_first,
            )
            self._conv1_cache[cache_key] = TtConv2d(conv_config, device)
        return self._conv1_cache[cache_key]

    def _get_conv2(self, device, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv2_cache:
            conv_config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.conv2_weight.shape[1],
                out_channels=self.conv2_weight.shape[0],
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.conv2_weight,
                bias=self.conv2_bias,
                stride=(self.stride, self.stride),
                padding=(1, 1),
                model_config=self.model_config,
                conv_config=self.optimizations.bottleneck_3x3,
            )
            self._conv2_cache[cache_key] = TtConv2d(conv_config, device)
        return self._conv2_cache[cache_key]

    def _get_conv3(self, device, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv3_cache:
            conv_config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.conv3_weight.shape[1],
                out_channels=self.conv3_weight.shape[0],
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.conv3_weight,
                bias=self.conv3_bias,
                model_config=self.model_config,
                conv_config=self.optimizations.bottleneck_1x1_last,
            )
            self._conv3_cache[cache_key] = TtConv2d(conv_config, device)
        return self._conv3_cache[cache_key]

    def _get_downsample(self, device, batch_size, height, width):
        if not self.has_downsample:
            return None
        cache_key = (batch_size, height, width)
        if cache_key not in self._downsample_cache:
            conv_config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.downsample_weight.shape[1],
                out_channels=self.downsample_weight.shape[0],
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.downsample_weight,
                bias=self.downsample_bias,
                stride=(self.stride, self.stride),
                model_config=self.model_config,
                conv_config=self.optimizations.downsample_1x1,
            )
            self._downsample_cache[cache_key] = TtConv2d(conv_config, device)
        return self._downsample_cache[cache_key]

    def __call__(self, x, device, batch_size, height, width):
        identity = x

        # Conv1 - 1x1 with ReLU using TtConv2d builder API
        conv1 = self._get_conv1(device, batch_size, height, width)
        out, (out_h, out_w) = conv1(x, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, out_h, out_w, self.conv1_weight.shape[0])

        # Conv2 - 3x3 with ReLU using TtConv2d builder API
        conv2 = self._get_conv2(device, batch_size, height, width)
        out, (out_h, out_w) = conv2(out, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, out_h, out_w, self.conv2_weight.shape[0])

        # Conv3 - 1x1 without ReLU using TtConv2d builder API
        conv3 = self._get_conv3(device, batch_size, out_h, out_w)
        out, (final_h, final_w) = conv3(out, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, final_h, final_w, self.conv3_weight.shape[0])

        # Downsample path using TtConv2d builder API
        if self.has_downsample:
            downsample = self._get_downsample(device, batch_size, height, width)
            identity, _ = downsample(identity, return_output_dim=True)
            identity = post_process_conv_output(identity, batch_size, final_h, final_w, self.downsample_weight.shape[0])

        # Ensure memory configs match for add
        identity = ensure_memory_config(identity, reference_tensor=out)

        # Add and ReLU
        out = ttnn.add_(
            out,
            identity,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )

        return out, final_h, final_w


class ResNet50_BEVDepth:
    """
    ResNet50 backbone for BEVDepth using TtConv2d builder API directly.
    """

    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        return_intermediate=True,
        return_block_outputs=False,
        optimizations=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_config = model_config
        self.return_intermediate = return_intermediate
        self.return_block_outputs = return_block_outputs
        self.optimizations = optimizations or resnet50_optimizations

        # Store conv1 weights for lazy conversion
        self.conv1_weight = parameters.conv1.weight
        self.conv1_bias = parameters.conv1.bias if hasattr(parameters.conv1, "bias") else None

        # Cache for TtConv2d instance
        self._conv1_cache = {}
        self._maxpool_cache = {}

        # Build layers
        self.in_channels = 64
        self.layer1 = self._make_layer(parameters.layer1, 64, 3, stride=1)
        self.layer2 = self._make_layer(parameters.layer2, 128, 4, stride=2)
        self.layer3 = self._make_layer(parameters.layer3, 256, 6, stride=2)
        self.layer4 = self._make_layer(parameters.layer4, 512, 3, stride=2)

    def _get_conv1(self, device, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            conv_config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=3,
                out_channels=64,
                batch_size=batch_size,
                kernel_size=(7, 7),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                stride=(2, 2),
                padding=(3, 3),
                model_config=self.model_config,
                conv_config=self.optimizations.conv1_7x7,
            )
            self._conv1_cache[cache_key] = TtConv2d(conv_config, device)
        return self._conv1_cache[cache_key]

    def _get_maxpool(self, height, width, batch_size):
        cache_key = (height, width, batch_size)
        if cache_key not in self._maxpool_cache:
            config = create_maxpool_config(
                input_height=height,
                input_width=width,
                channels=64,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            )
            self._maxpool_cache[cache_key] = TtMaxPool2d(config, self.device)
        return self._maxpool_cache[cache_key]

    def _make_layer(self, layer_params, planes, blocks, stride=1):
        layers = []

        downsample = None
        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample = True

        layers.append(
            Bottleneck(
                parameters=layer_params[0],
                in_channels=self.in_channels,
                out_channels=planes,
                stride=stride,
                downsample=downsample,
                model_config=self.model_config,
                optimizations=self.optimizations,
            )
        )
        self.in_channels = planes * Bottleneck.expansion

        for i in range(1, blocks):
            layers.append(
                Bottleneck(
                    parameters=layer_params[i],
                    in_channels=self.in_channels,
                    out_channels=planes,
                    stride=1,
                    downsample=None,
                    model_config=self.model_config,
                    optimizations=self.optimizations,
                )
            )

        return layers

    def __call__(self, x, input_height=None, input_width=None):
        batch_size = self.batch_size

        if input_height is None or input_width is None:
            _, height, width, _ = x.shape
        else:
            height, width = input_height, input_width

        # Initialize features dict
        features = {}
        block_outputs = {}

        # Conv1: 7x7, stride 2 with ReLU using TtConv2d builder API
        conv1 = self._get_conv1(self.device, batch_size, height, width)
        x, (out_h, out_w) = conv1(x, return_output_dim=True)
        height, width = out_h, out_w
        x = post_process_conv_output(x, batch_size, height, width, 64)

        if self.return_block_outputs:
            features["conv1_output"] = x

        # MaxPool using TtMaxPool2d builder API
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        pool_input = ttnn.reshape(x, (batch_size, 1, height * width, 64))

        maxpool = self._get_maxpool(height, width, batch_size)
        x = maxpool(pool_input)

        height = height // 2
        width = width // 2

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (batch_size, height, width, 64))

        if self.return_block_outputs:
            features["layer1_input"] = x

        # Layer1
        for i, block in enumerate(self.layer1):
            x, height, width = block(x, self.device, batch_size, height, width)
            if self.return_block_outputs:
                block_outputs[f"layer1_block{i}"] = x

        if self.return_intermediate:
            features["layer1"] = x

        if self.return_block_outputs:
            features.update(block_outputs)

        # Layer2
        for i, block in enumerate(self.layer2):
            x, height, width = block(x, self.device, batch_size, height, width)
        if self.return_intermediate:
            features["layer2"] = x

        # Layer3
        for i, block in enumerate(self.layer3):
            x, height, width = block(x, self.device, batch_size, height, width)
        if self.return_intermediate:
            features["layer3"] = x

        # Layer4
        for i, block in enumerate(self.layer4):
            x, height, width = block(x, self.device, batch_size, height, width)
        if self.return_intermediate:
            features["layer4"] = x

        if self.return_intermediate:
            return features
        return x
