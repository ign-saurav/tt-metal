# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
from loguru import logger
from models.experimental.BevDepth.tt.utils import ttnn_conv2d


def get_conv_config(conv_type, custom_overrides=None):
    """
    Get preset configuration for different convolution types in ResNet50.

    Args:
        conv_type: One of 'conv1_7x7', 'bottleneck_1x1_first', 'bottleneck_3x3',
                   'bottleneck_1x1_last', 'downsample_1x1'
        custom_overrides: Dict of parameters to override defaults

    Returns:
        Dict of parameters for ttnn_conv2d
    """

    configs = {
        "conv1_7x7": {
            "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": False,
            "packer_l1_acc": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": False,
        },
        "bottleneck_1x1_first": {
            "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": False,
            "packer_l1_acc": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": False,
        },
        "bottleneck_3x3": {
            "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "packer_l1_acc": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": False,
        },
        "bottleneck_1x1_last": {
            "activation": None,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": False,
            "packer_l1_acc": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": False,
        },
        "downsample_1x1": {
            "activation": None,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": False,
            "packer_l1_acc": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": False,
        },
    }

    if conv_type not in configs:
        raise ValueError(f"Unknown conv_type: {conv_type}. Available: {list(configs.keys())}")

    config = configs[conv_type].copy()

    if custom_overrides:
        config.update(custom_overrides)

    return config


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
        if len(out.shape) == 3:
            out = ttnn.reshape(out, (batch_size, out_height, out_width, self.conv2_weight_torch.shape[0]))

        # Conv3 - 1x1 without ReLU
        config = get_conv_config("bottleneck_1x1_last")
        config = config.copy()  # MUST copy before modifying
        config["deallocate_activation"] = False
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

        # Add and ReLU
        out = ttnn.add(out, identity, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])

        return out, out_height, out_width


class ResNet50_BEVDepth:
    def __init__(self, device, parameters, batch_size, model_config, return_intermediate=True):
        self.device = device
        self.batch_size = batch_size
        self.model_config = model_config
        self.return_intermediate = return_intermediate

        # Conv1 - keep as PyTorch tensor, convert lazily
        self.conv1_weight_torch = parameters.conv1.weight
        self.conv1_bias_torch = parameters.conv1.bias if hasattr(parameters.conv1, "bias") else None
        self.conv1_weight_ttnn = None  # Converted TTNN tensor (cached)
        self.conv1_bias_ttnn = None

        # Build layers
        self.in_channels = 64

        logger.info("Building layer1...")
        self.layer1 = self._make_layer(parameters.layer1, 64, 3, stride=1)
        logger.info(f"Layer1 created with {len(self.layer1)} blocks")

        logger.info("Building layer2...")
        self.layer2 = self._make_layer(parameters.layer2, 128, 4, stride=2)
        logger.info(f"Layer2 created with {len(self.layer2)} blocks")

        logger.info("Building layer3...")
        self.layer3 = self._make_layer(parameters.layer3, 256, 6, stride=2)
        logger.info(f"Layer3 created with {len(self.layer3)} blocks")

        logger.info("Building layer4...")
        self.layer4 = self._make_layer(parameters.layer4, 512, 3, stride=2)
        logger.info(f"Layer4 created with {len(self.layer4)} blocks")

        logger.info("ResNet50_BEVDepth initialization complete")

    def _get_conv1_weights(self):
        """Get conv1 weights in TTNN format (convert if needed, ROW_MAJOR_LAYOUT for host)"""
        if self.conv1_weight_ttnn is None:
            self.conv1_weight_ttnn = ttnn.from_torch(
                self.conv1_weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Host weights must be ROW_MAJOR_LAYOUT
                # device=None keeps on host, TTNN will convert to TILE_LAYOUT when moving to device
            )
        if self.conv1_bias_torch is not None and self.conv1_bias_ttnn is None:
            self.conv1_bias_ttnn = ttnn.from_torch(
                self.conv1_bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
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
        # Input is in (B, H, W, C) format from TTNN
        # Use provided dimensions or extract from tensor shape
        if input_height is None or input_width is None:
            _, height, width, _ = x.shape
        else:
            height, width = input_height, input_width

        # Conv1: 7x7, stride 2 with ReLU (convert weights lazily)
        conv1_weight, conv1_bias = self._get_conv1_weights()
        config = get_conv_config("conv1_7x7")
        x = ttnn_conv2d(
            input_tensor=x,
            weight_tensor=conv1_weight,
            bias_tensor=conv1_bias,
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
            **config,
        )

        height = height // 2
        width = width // 2
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

        x = ttnn.reshape(x, (batch_size, height, width, 64))

        logger.info(f"After maxpool, x.shape = {x.shape}, expected: ({batch_size}, {height}, {width}, 64)")

        features = {}

        # Layer1
        for i, block in enumerate(self.layer1):
            logger.info(f"Before layer1 block {i}, x.shape = {x.shape}, ndim = {len(x.shape)}")

            x, height, width = block(x, self.device, batch_size, height, width)
            logger.info(f"After layer1 block {i}, x.shape = {x.shape}")
        if self.return_intermediate:
            features["layer1"] = x

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
