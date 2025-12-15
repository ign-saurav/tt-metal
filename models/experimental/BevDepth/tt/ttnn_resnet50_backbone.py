# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from dataclasses import dataclass
from models.experimental.BevDepth.tt.utils import ttnn_conv2d


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
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_first={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_3x3={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_last={
        "activation": None,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    downsample_1x1={
        "activation": None,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)


def get_conv_config(conv_type, optimizations=None):
    if optimizations is None:
        optimizations = resnet50_optimizations
    return getattr(optimizations, conv_type).copy()


class Bottleneck:
    expansion = 4

    def __init__(self, parameters, in_channels, out_channels, stride=1, downsample=None, model_config=None):
        self.stride = stride
        self.downsample = downsample
        self.model_config = model_config

        # Conv1: 1x1 - keep as PyTorch tensor, convert lazily
        self.conv1_weight_torch = parameters.conv1.weight
        self.conv1_bias_torch = parameters.conv1.bias if hasattr(parameters.conv1, "bias") else None
        self.conv1_weight_ttnn = None  # Converted TTNN tensor (cached)
        self.conv1_bias_ttnn = None

        # Conv2: 3x3 - keep as PyTorch tensor, convert lazily
        self.conv2_weight_torch = parameters.conv2.weight
        self.conv2_bias_torch = parameters.conv2.bias if hasattr(parameters.conv2, "bias") else None
        self.conv2_weight_ttnn = None
        self.conv2_bias_ttnn = None

        # Conv3: 1x1 - keep as PyTorch tensor, convert lazily
        self.conv3_weight_torch = parameters.conv3.weight
        self.conv3_bias_torch = parameters.conv3.bias if hasattr(parameters.conv3, "bias") else None
        self.conv3_weight_ttnn = None
        self.conv3_bias_ttnn = None

        # Downsample if exists - keep as PyTorch tensor, convert lazily
        if downsample:
            self.downsample_conv_weight_torch = parameters.downsample[0].weight
            self.downsample_conv_bias_torch = (
                parameters.downsample[0].bias if hasattr(parameters.downsample[0], "bias") else None
            )
            self.downsample_conv_weight_ttnn = None
            self.downsample_conv_bias_ttnn = None
        else:
            self.downsample_conv_weight_torch = None
            self.downsample_conv_bias_torch = None

    def _convert_weight_to_ttnn(self, torch_weight, device):
        """Convert PyTorch weight to TTNN format lazily (ROW_MAJOR_LAYOUT for host weights)"""
        return ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            # device=None keeps on host, TTNN will convert to TILE_LAYOUT when moving to device
        )

    def _get_conv1_weights(self, device):
        """Get conv1 weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)"""
        if self.conv1_weight_ttnn is None:
            self.conv1_weight_ttnn = self._convert_weight_to_ttnn(self.conv1_weight_torch, device)
        if self.conv1_bias_torch is not None and self.conv1_bias_ttnn is None:
            self.conv1_bias_ttnn = ttnn.from_torch(
                self.conv1_bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            )
        return self.conv1_weight_ttnn, self.conv1_bias_ttnn

    def _get_conv2_weights(self, device):
        """Get conv2 weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)"""
        if self.conv2_weight_ttnn is None:
            self.conv2_weight_ttnn = self._convert_weight_to_ttnn(self.conv2_weight_torch, device)
        if self.conv2_bias_torch is not None and self.conv2_bias_ttnn is None:
            self.conv2_bias_ttnn = ttnn.from_torch(
                self.conv2_bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            )
        return self.conv2_weight_ttnn, self.conv2_bias_ttnn

    def _get_conv3_weights(self, device):
        """Get conv3 weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)"""
        if self.conv3_weight_ttnn is None:
            self.conv3_weight_ttnn = self._convert_weight_to_ttnn(self.conv3_weight_torch, device)
        if self.conv3_bias_torch is not None and self.conv3_bias_ttnn is None:
            self.conv3_bias_ttnn = ttnn.from_torch(
                self.conv3_bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            )
        return self.conv3_weight_ttnn, self.conv3_bias_ttnn

    def _get_downsample_weights(self, device):
        """Get downsample weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)"""
        if self.downsample_conv_weight_torch is None:
            return None, None
        if self.downsample_conv_weight_ttnn is None:
            self.downsample_conv_weight_ttnn = self._convert_weight_to_ttnn(self.downsample_conv_weight_torch, device)
        if self.downsample_conv_bias_torch is not None and self.downsample_conv_bias_ttnn is None:
            self.downsample_conv_bias_ttnn = ttnn.from_torch(
                self.downsample_conv_bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            )
        return self.downsample_conv_weight_ttnn, self.downsample_conv_bias_ttnn

    def __call__(self, x, device, batch_size, height, width):
        identity = x
        # if self.downsample:
        #     identity = x  # Save reference before x is modified
        # else:
        #     identity = x

        # Conv1 - 1x1 with ReLU - pass PyTorch tensor directly
        config = get_conv_config("bottleneck_1x1_first")
        if self.downsample:
            config = config.copy()
            config["deallocate_activation"] = False
        config = config.copy()  # MUST copy before modifying
        config["deallocate_activation"] = False
        out = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_torch,  # Direct PyTorch tensor
            bias_tensor=self.conv1_bias_torch,
            device=device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.conv1_weight_torch.shape[1],
            out_channels=self.conv1_weight_torch.shape[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **config,
        )
        # Convert sharded to interleaved before reshape (required for reshape)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        # Convert sharded to interleaved before reshape (required for reshape)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, height, width, self.conv1_weight_torch.shape[0]))

        # Conv2 - 3x3 with ReLU
        out_height = height if self.stride == 1 else height // 2
        out_width = width if self.stride == 1 else width // 2

        config = get_conv_config("bottleneck_3x3")
        config = config.copy()  # MUST copy before modifying
        config["deallocate_activation"] = False
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_torch,  # Direct PyTorch tensor
            bias_tensor=self.conv2_bias_torch,
            device=device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=self.conv2_weight_torch.shape[1],
            out_channels=self.conv2_weight_torch.shape[0],
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **config,
        )
        # Convert sharded to interleaved before reshape (required for reshape)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, out_height, out_width, self.conv2_weight_torch.shape[0]))

        # Conv3 - 1x1 without ReLU
        config = get_conv_config("bottleneck_1x1_last")
        config = config.copy()  # MUST copy before modifying
        config["deallocate_activation"] = True  # Deallocate like the demo
        out = ttnn_conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_torch,  # Direct PyTorch tensor
            bias_tensor=self.conv3_bias_torch,
            device=device,
            batch_size=batch_size,
            input_height=out_height,
            input_width=out_width,
            in_channels=self.conv3_weight_torch.shape[1],
            out_channels=self.conv3_weight_torch.shape[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **config,
        )
        # Convert sharded to interleaved before reshape (required for reshape)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        if len(out.shape) == 3:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.reshape(out, (batch_size, out_height, out_width, self.conv3_weight_torch.shape[0]))

        if self.downsample:
            config = get_conv_config("downsample_1x1")
            config = config.copy()  # MUST copy before modifying

            config["deallocate_activation"] = False
            identity = ttnn_conv2d(
                input_tensor=identity,
                weight_tensor=self.downsample_conv_weight_torch,
                bias_tensor=self.downsample_conv_bias_torch,
                device=device,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                in_channels=self.downsample_conv_weight_torch.shape[1],
                out_channels=self.downsample_conv_weight_torch.shape[0],
                kernel_size=(1, 1),
                stride=(self.stride, self.stride),
                padding=(0, 0),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                **config,
            )
            # Convert sharded to interleaved before reshape (required for reshape)
            if identity.is_sharded():
                identity = ttnn.sharded_to_interleaved(identity, ttnn.DRAM_MEMORY_CONFIG)
            # Reshape identity if needed
            if len(identity.shape) == 3:
                identity = ttnn.reshape(
                    identity, (batch_size, out_height, out_width, self.downsample_conv_weight_torch.shape[0])
                )

            # Ensure both tensors have matching memory config and layout
            if identity.memory_config() != out.memory_config():
                identity = ttnn.to_memory_config(identity, out.memory_config())
            # if identity.layout != out.layout:
            #     identity = ttnn.to_layout(identity, out.layout)

        # Add and ReLU - use in-place add like the demo
        # Ensure both tensors have matching memory config
        if identity.memory_config() != out.memory_config():
            identity = ttnn.to_memory_config(identity, out.memory_config())

        # Use in-place add like the demo implementation
        out = ttnn.add_(
            out,
            identity,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )

        return out, out_height, out_width


class ResNet50_BEVDepth:
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

        # Conv1 - keep as PyTorch tensor, convert lazily
        self.conv1_weight_torch = parameters.conv1.weight
        self.conv1_bias_torch = parameters.conv1.bias if hasattr(parameters.conv1, "bias") else None
        self.conv1_weight_ttnn = None  # Converted TTNN tensor (cached)
        self.conv1_bias_ttnn = None

        # Build layers
        self.in_channels = 64
        self.layer1 = self._make_layer(parameters.layer1, 64, 3, stride=1)
        self.layer2 = self._make_layer(parameters.layer2, 128, 4, stride=2)
        self.layer3 = self._make_layer(parameters.layer3, 256, 6, stride=2)
        self.layer4 = self._make_layer(parameters.layer4, 512, 3, stride=2)

    def _get_conv1_weights(self):
        """Get conv1 weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)

        Note: ttnn.conv2d expects weights in PyTorch format or TTNN ROW_MAJOR_LAYOUT format.
        It will handle the conversion to TILE_LAYOUT internally.
        """
        if self.conv1_weight_ttnn is None:
            # Convert weight to bfloat16 first to match reference model precision
            weight_torch = self.conv1_weight_torch.to(torch.bfloat16)
            self.conv1_weight_ttnn = ttnn.from_torch(
                weight_torch,
                dtype=self.model_config["WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
                # device=None keeps on host, TTNN will convert to TILE_LAYOUT when moving to device
            )
        if self.conv1_bias_torch is not None and self.conv1_bias_ttnn is None:
            # Convert bias to bfloat16 first to match reference model precision
            bias_torch = self.conv1_bias_torch.to(torch.bfloat16)
            self.conv1_bias_ttnn = ttnn.from_torch(
                bias_torch.reshape(1, 1, 1, -1),
                dtype=self.model_config["WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
            )
        return self.conv1_weight_ttnn, self.conv1_bias_ttnn

        # Build layers
        self.in_channels = 64
        self.layer1 = self._make_layer(parameters.layer1, 64, 3, stride=1)
        self.layer2 = self._make_layer(parameters.layer2, 128, 4, stride=2)
        self.layer3 = self._make_layer(parameters.layer3, 256, 6, stride=2)
        self.layer4 = self._make_layer(parameters.layer4, 512, 3, stride=2)

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
                )
            )

        return layers

    def __call__(self, x, input_height=None, input_width=None):
        batch_size = self.batch_size

        if input_height is None or input_width is None:
            _, height, width, _ = x.shape
        else:
            height, width = input_height, input_width

        # Initialize features dict early for debugging outputs
        features = {}
        block_outputs = {}  # Store block-level outputs for debugging

        # Conv1: 7x7, stride 2 with ReLU
        # Use ttnn_conv2d wrapper to ensure proper weight/bias conversion
        config = get_conv_config("conv1_7x7")
        config = config.copy()  # MUST copy before modifying

        # Use ttnn_conv2d wrapper which handles PyTorch->TTNN conversion properly
        x = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_torch,  # Pass PyTorch tensor directly
            bias_tensor=self.conv1_bias_torch,  # Pass PyTorch tensor directly
            device=self.device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activations_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            activation=config.get("activation", ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)),
            deallocate_activation=config.get("deallocate_activation", True),
            reallocate_halo_output=config.get("reallocate_halo_output", False),
            shard_layout=config.get("shard_layout", ttnn.TensorMemoryLayout.BLOCK_SHARDED),
            packer_l1_acc=config.get("packer_l1_acc", True),
            enable_act_double_buffer=config.get("enable_act_double_buffer", True),
            enable_weights_double_buffer=config.get("enable_weights_double_buffer", False),
        )

        # Calculate output dimensions after conv1 (stride 2)
        height = (height + 2 * 3 - 7) // 2 + 1  # (input_h + 2*padding - kernel_h) // stride_h + 1
        width = (width + 2 * 3 - 7) // 2 + 1  # (input_w + 2*padding - kernel_w) // stride_w + 1

        # Convert sharded to interleaved before reshape (required for reshape)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # Reshape if needed (ttnn_conv2d might return different shape)
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (batch_size, height, width, 64))

        # Store conv1 output for debugging (before maxpool)
        if self.return_block_outputs:
            features["conv1_output"] = x

        # Reshape for maxpool - demo passes directly but we need to reshape for our format
        # Ensure x is not sharded before reshape
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        pool_input = ttnn.reshape(x, (batch_size, 1, height * width, 64))

        # MaxPool: 3x3, stride 2
        x = ttnn.max_pool2d(
            input_tensor=pool_input,
            batch_size=batch_size,
            input_h=height,
            input_w=width,
            channels=64,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        height = height // 2
        width = width // 2

        # Convert sharded to interleaved before reshape (required for reshape)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (batch_size, height, width, 64))

        # Store layer1 input if requested
        if self.return_block_outputs:
            features["layer1_input"] = x

        # Layer1
        for i, block in enumerate(self.layer1):
            x, height, width = block(x, self.device, batch_size, height, width)

            # Store block output if requested
            if self.return_block_outputs:
                block_outputs[f"layer1_block{i}"] = x

        if self.return_intermediate:
            features["layer1"] = x

        # Add block outputs to features if requested
        if self.return_block_outputs:
            features.update(block_outputs)

        # x1 = ttnn.clone(x)
        # ttnn.deallocate(x, force=True)

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
