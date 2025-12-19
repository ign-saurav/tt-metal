# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d, MaxPool2dConfiguration
from models.experimental.SSD512.tt.utils import _create_conv_config_from_params


@dataclass
class VggConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int


@dataclass
class VggPoolConfig:
    kernel_size: tuple
    stride: tuple
    padding: tuple
    dilation: tuple
    ceil_mode: bool


class ConvBlock:
    def __init__(
        self,
        input_height: int,
        input_width: int,
        config: VggConvConfig,
        batch_size: int,
        parameters: dict,
        device,
    ):
        from models.tt_cnn.tt.builder import (
            AutoShardedStrategyConfiguration,
            WidthSliceStrategyConfiguration,
        )

        self.config = config

        # Determine memory and slice strategy
        tensor_size_estimate = batch_size * input_height * input_width * config.in_channels
        use_l1 = input_height <= 64 and input_width <= 64 and tensor_size_estimate <= 1 * 1024 * 1024
        self.memory_config = ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG

        sharding_strategy = AutoShardedStrategyConfiguration()
        deallocate_activation = config.in_channels == 3 and config.out_channels == 64

        # For large DRAM operations, use WidthSliceStrategy to avoid auto-slice failures
        if not use_l1 and input_height >= 256:
            num_slices = max(2, min(8, (batch_size * input_height * input_width) // (128 * 128)))
            slice_strategy = WidthSliceStrategyConfiguration(num_slices=num_slices)
        else:
            slice_strategy = None

        conv_cfg = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            batch_size=batch_size,
            parameters=parameters,
            device=device,
            kernel_size=(config.kernel_size, config.kernel_size),
            stride=(config.stride, config.stride),
            padding=(config.padding, config.padding),
            dilation=(config.dilation, config.dilation),
            sharding_strategy=sharding_strategy,
            slice_strategy=slice_strategy,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            deallocate_activation=deallocate_activation,
            # config_tensors_in_dram=True,
        )
        self.conv = TtConv2d(conv_cfg, device)

        self.out_height = (
            input_height + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1
        ) // config.stride + 1
        self.out_width = (
            input_width + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1
        ) // config.stride + 1

    def __call__(self, x):
        if hasattr(x, "memory_config") and x.memory_config().buffer_type != self.memory_config.buffer_type:
            x = ttnn.to_memory_config(x, self.memory_config)

        x = self.conv(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, self.memory_config)
        return x, self.out_height, self.out_width


class PoolBlock:
    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: int,
        config: VggPoolConfig,
        batch_size: int,
        device,
    ):
        self.config = config
        self.device = device

        kernel_h, kernel_w = config.kernel_size
        stride_h, stride_w = config.stride
        padding_h, padding_w = config.padding
        dilation_h, dilation_w = config.dilation

        if config.ceil_mode:
            self.out_height = int((input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
            if (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) % stride_h != 0:
                self.out_height += 1
            self.out_width = int((input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
            if (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) % stride_w != 0:
                self.out_width += 1
        else:
            self.out_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
            self.out_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        tensor_size_estimate = batch_size * input_height * input_width * channels
        use_l1 = input_height <= 64 and input_width <= 64 and tensor_size_estimate <= 1 * 1024 * 1024
        self.memory_config = ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG

        pool_config = MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=batch_size,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            ceil_mode=config.ceil_mode,
            dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_input=False,
            reallocate_halo_output=True,
        )
        self.pool = TtMaxPool2d(pool_config, device)

    def __call__(self, x):
        if hasattr(x, "memory_config") and x.memory_config().buffer_type != self.memory_config.buffer_type:
            x = ttnn.to_memory_config(x, self.memory_config)

        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self.pool(x)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if hasattr(x, "memory_config") and x.memory_config().buffer_type != self.memory_config.buffer_type:
            x = ttnn.to_memory_config(x, self.memory_config)

        return x, self.out_height, self.out_width


class TtVggBackbone:
    def __init__(self, size: int, input_channels: int, batch_size: int, parameters: list, device):
        self.size = size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels

        vgg_cfg = {
            "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        }

        if size not in [512]:
            raise ValueError(f"Size must be 512, got {size}")

        self.cfg = vgg_cfg[str(size)]
        self.parameters = parameters

        # Build blocks immediately with fixed input size (512x512 for SSD512)
        self.input_height = 512
        self.input_width = 512
        self.blocks = self._build_blocks(self.cfg, input_channels, self.input_height, self.input_width, parameters)

    def _build_blocks(self, cfg, input_channels, input_height, input_width, parameters):
        blocks = []
        in_channels = input_channels
        current_h = input_height
        current_w = input_width
        param_idx = 0

        for v in cfg:
            if v == "M":
                pool_config = VggPoolConfig(
                    kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False
                )
                block = PoolBlock(current_h, current_w, in_channels, pool_config, self.batch_size, self.device)
                blocks.append(("pool", block))
                current_h = block.out_height
                current_w = block.out_width

            elif v == "C":
                pool_config = VggPoolConfig(
                    kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=True
                )
                block = PoolBlock(current_h, current_w, in_channels, pool_config, self.batch_size, self.device)
                blocks.append(("pool", block))
                current_h = block.out_height
                current_w = block.out_width

            else:
                conv_config = VggConvConfig(in_channels, v, 3, 1, 1, 1)
                block = ConvBlock(
                    current_h, current_w, conv_config, self.batch_size, parameters[param_idx], self.device
                )
                blocks.append(("conv", block))
                blocks.append(("relu", None))
                current_h = block.out_height
                current_w = block.out_width
                in_channels = v
                param_idx += 1

        pool5_config = VggPoolConfig(
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=False
        )
        block = PoolBlock(current_h, current_w, in_channels, pool5_config, self.batch_size, self.device)
        blocks.append(("pool", block))
        current_h = block.out_height
        current_w = block.out_width

        conv6_config = VggConvConfig(512, 1024, 3, 1, 6, 6)
        block = ConvBlock(current_h, current_w, conv6_config, self.batch_size, parameters[param_idx], self.device)
        blocks.append(("conv", block))
        blocks.append(("relu", None))
        current_h = block.out_height
        current_w = block.out_width
        param_idx += 1

        conv7_config = VggConvConfig(1024, 1024, 1, 1, 0, 1)
        block = ConvBlock(current_h, current_w, conv7_config, self.batch_size, parameters[param_idx], self.device)
        blocks.append(("conv", block))
        blocks.append(("relu", None))

        return blocks

    def __call__(self, x, return_sources=None):
        import torch

        # Blocks are already built in __init__, just convert input if needed
        if isinstance(x, torch.Tensor):
            x = x.permute(0, 2, 3, 1)
            tensor_size = x.numel() * 2
            memory_config = ttnn.L1_MEMORY_CONFIG if tensor_size <= 2 * 1024 * 1024 else ttnn.DRAM_MEMORY_CONFIG

            x = ttnn.from_torch(
                x,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
            )

        sources = []
        return_sources_set = set(return_sources) if return_sources is not None else None

        for idx, (block_type, block) in enumerate(self.blocks):
            if block_type == "conv":
                x, _, _ = block(x)
            elif block_type == "pool":
                x, _, _ = block(x)
            elif block_type == "relu":
                x = ttnn.relu(x)

            if return_sources_set is not None and idx in return_sources_set:
                sources.append(x)

        if return_sources is not None:
            return x, sources
        return x


def vgg_backbone(
    size: int = 512,
    input_channels: int = 3,
    batch_size: int = 1,
    parameters: list = None,
    device=None,
) -> TtVggBackbone:
    import torch

    if parameters is None:
        vgg_cfg = {
            "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
        }
        cfg = vgg_cfg[str(size)]

        parameters = []
        in_channels = input_channels

        for v in cfg:
            if v not in ["M", "C"]:
                dummy_weight = torch.zeros(v, in_channels, 3, 3)
                dummy_bias = torch.zeros(v)
                parameters.append({"weight": dummy_weight, "bias": dummy_bias})
                in_channels = v

        dummy_weight = torch.zeros(1024, 512, 3, 3)
        dummy_bias = torch.zeros(1024)
        parameters.append({"weight": dummy_weight, "bias": dummy_bias})

        dummy_weight = torch.zeros(1024, 1024, 1, 1)
        dummy_bias = torch.zeros(1024)
        parameters.append({"weight": dummy_weight, "bias": dummy_bias})

    return TtVggBackbone(size, input_channels, batch_size, parameters, device)


def build_vgg_backbone(
    size: int = 512,
    input_channels: int = 3,
    batch_size: int = 1,
    parameters: list = None,
    device=None,
) -> TtVggBackbone:
    return vgg_backbone(size, input_channels, batch_size, parameters, device)
