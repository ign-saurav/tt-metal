# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.SSD512.tt.utils import _create_conv_config_from_params


@dataclass
class ExtraBlockConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


class ConvBlock:
    def __init__(
        self,
        input_height: int,
        input_width: int,
        config: ExtraBlockConfig,
        batch_size: int,
        parameters: dict,
        device,
    ):
        self.config = config
        self.out_height = (input_height + 2 * config.padding - config.kernel_size) // config.stride + 1
        self.out_width = (input_width + 2 * config.padding - config.kernel_size) // config.stride + 1

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
        )
        self.conv = TtConv2d(conv_cfg, device)

    def __call__(self, x):
        x = self.conv(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x)
        return x, self.out_height, self.out_width


class ExtraBlock:
    def __init__(
        self,
        input_height: int,
        input_width: int,
        config: ExtraBlockConfig,
        batch_size: int,
        parameters: dict,
        device,
    ):
        self.block = ConvBlock(input_height, input_width, config, batch_size, parameters, device)

    def __call__(self, x):
        return self.block(x)


class TtExtrasBackbone:
    def __init__(self, size: int, input_channels: int, batch_size: int, parameters: list, device):
        self.size = size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels

        extras_cfg = {
            "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
            "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
        }

        if size not in [512]:
            raise ValueError(f"Size must be 512, got {size}")

        self.cfg = extras_cfg[str(size)]
        self.parameters = parameters
        self.blocks = None
        self.dimensions = None
        self.input_height = None
        self.input_width = None

    def _build_blocks(self, cfg, input_channels, input_height, input_width, parameters):
        blocks = []
        dimensions = []
        in_channels = input_channels
        flag = False
        current_h = input_height
        current_w = input_width
        param_idx = 0

        for k, v in enumerate(cfg):
            if in_channels != "S":
                if v == "S":
                    out_channels = cfg[k + 1]
                    kernel_size = 1 if not flag else 3
                    stride = 2
                    padding = 1

                    block_config = ExtraBlockConfig(in_channels, out_channels, kernel_size, stride, padding)
                    block = ExtraBlock(
                        current_h, current_w, block_config, self.batch_size, parameters[param_idx], self.device
                    )
                    blocks.append(block)

                    current_h = (current_h + 2 * padding - kernel_size) // stride + 1
                    current_w = (current_w + 2 * padding - kernel_size) // stride + 1
                    dimensions.append((current_h, current_w, out_channels))

                    param_idx += 1
                    flag = not flag
                else:
                    out_channels = v
                    kernel_size = 1 if not flag else 3
                    stride = 1
                    padding = 0

                    block_config = ExtraBlockConfig(in_channels, out_channels, kernel_size, stride, padding)
                    block = ExtraBlock(
                        current_h, current_w, block_config, self.batch_size, parameters[param_idx], self.device
                    )
                    blocks.append(block)

                    current_h = (current_h + 2 * padding - kernel_size) // stride + 1
                    current_w = (current_w + 2 * padding - kernel_size) // stride + 1
                    dimensions.append((current_h, current_w, out_channels))

                    param_idx += 1
                    flag = not flag

            in_channels = v

        if len(cfg) == 13:
            block_config = ExtraBlockConfig(in_channels, 256, 4, 1, 1)
            block = ExtraBlock(current_h, current_w, block_config, self.batch_size, parameters[param_idx], self.device)
            blocks.append(block)

            current_h = (current_h + 2 * 1 - 4) // 1 + 1
            current_w = (current_w + 2 * 1 - 4) // 1 + 1
            dimensions.append((current_h, current_w, 256))

        return blocks, dimensions

    def load_weights_from_torch(self, torch_model):
        pass

        for idx, torch_layer in enumerate(torch_model):
            weight = torch_layer.weight.data
            bias = torch_layer.bias.data if torch_layer.bias is not None else None

            self.parameters[idx]["weight"] = weight
            if bias is not None:
                self.parameters[idx]["bias"] = bias

    def __call__(self, x, return_sources=False):
        import torch

        if isinstance(x, torch.Tensor):
            _, _, input_h, input_w = x.shape
            if self.blocks is None or self.input_height != input_h or self.input_width != input_w:
                self.input_height = input_h
                self.input_width = input_w
                self.blocks, self.dimensions = self._build_blocks(
                    self.cfg, self.input_channels, input_h, input_w, self.parameters
                )

            x = x.permute(0, 2, 3, 1)
            x = ttnn.from_torch(
                x,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            if self.blocks is None:
                _, input_h, input_w, _ = x.shape
                self.input_height = input_h
                self.input_width = input_w
                self.blocks, self.dimensions = self._build_blocks(
                    self.cfg, self.input_channels, input_h, input_w, self.parameters
                )

        sources = []
        for idx, block in enumerate(self.blocks):
            x, h, w = block(x)
            if return_sources and idx % 2 == 1:
                sources.append(x)

        if return_sources:
            return x, sources
        return x


def extras_backbone(
    size: int = 512,
    input_channels: int = 1024,
    batch_size: int = 1,
    parameters: dict = None,
    device=None,
) -> TtExtrasBackbone:
    import torch

    if parameters is None:
        extras_cfg = {
            "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
            "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
        }
        cfg = extras_cfg[str(size)]

        parameters = []
        in_channels = input_channels
        flag = False

        for k, v in enumerate(cfg):
            if in_channels != "S":
                if v == "S":
                    out_channels = cfg[k + 1]
                    kernel_size = 1 if not flag else 3
                else:
                    out_channels = v
                    kernel_size = 1 if not flag else 3

                dummy_weight = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
                dummy_bias = torch.zeros(out_channels)
                parameters.append({"weight": dummy_weight, "bias": dummy_bias})
                flag = not flag

            in_channels = v

        if len(cfg) == 13:
            dummy_weight = torch.zeros(256, in_channels, 4, 4)
            dummy_bias = torch.zeros(256)
            parameters.append({"weight": dummy_weight, "bias": dummy_bias})

    return TtExtrasBackbone(size, input_channels, batch_size, parameters, device)


def build_extras_backbone(
    size: int = 512,
    input_channels: int = 1024,
    batch_size: int = 1,
    parameters: dict = None,
    device=None,
) -> TtExtrasBackbone:
    return extras_backbone(size, input_channels, batch_size, parameters, device)
