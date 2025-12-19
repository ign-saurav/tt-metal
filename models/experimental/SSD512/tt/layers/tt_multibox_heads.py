# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.SSD512.tt.utils import _create_conv_config_from_params


@dataclass
class HeadConfig:
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1


class ConvHead:
    def __init__(
        self,
        input_height: int,
        input_width: int,
        config: HeadConfig,
        batch_size: int,
        parameters: dict,
        device,
    ):
        from models.tt_cnn.tt.builder import AutoShardedStrategyConfiguration, L1FullSliceStrategyConfiguration

        self.config = config
        self.batch_size = batch_size

        # Multibox heads work with smaller feature maps, so we can use L1 for smaller layers
        tensor_size_estimate = batch_size * input_height * input_width * config.in_channels
        use_l1 = input_height <= 64 and input_width <= 64 and tensor_size_estimate <= 2 * 1024 * 1024

        self.memory_config = ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG

        sharding_strategy = AutoShardedStrategyConfiguration()
        slice_strategy = L1FullSliceStrategyConfiguration() if use_l1 else None
        enable_act_double_buffer = use_l1
        enable_weights_double_buffer = use_l1

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
            sharding_strategy=sharding_strategy,
            slice_strategy=slice_strategy,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
            # config_tensors_in_dram=(not use_l1),
        )
        self.conv = TtConv2d(conv_cfg, device)

        self.out_height = (input_height + 2 * config.padding - config.kernel_size) // config.stride + 1
        self.out_width = (input_width + 2 * config.padding - config.kernel_size) // config.stride + 1

    def __call__(self, x):
        if hasattr(x, "memory_config") and x.memory_config().buffer_type != self.memory_config.buffer_type:
            x = ttnn.to_memory_config(x, self.memory_config)

        x = self.conv(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, self.memory_config)

        # Explicitly reshape to ensure correct output format [batch, height, width, channels]
        x = x.reshape([self.batch_size, self.out_height, self.out_width, self.config.out_channels])

        return x


class TtMultiboxHeads:
    def __init__(
        self,
        size: int,
        num_classes: int,
        batch_size: int,
        loc_parameters: list,
        conf_parameters: list,
        vgg_channels: list,
        extra_channels: list,
        device,
    ):
        self.size = size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.vgg_channels = vgg_channels
        self.extra_channels = extra_channels
        self.loc_parameters = loc_parameters
        self.conf_parameters = conf_parameters

        self.mbox_cfg = {
            "300": [4, 6, 6, 6, 4, 4],
            "512": [4, 6, 6, 6, 4, 4, 4],
        }[str(size)]

        # Pre-define expected source dimensions for SSD512
        # Format: (height, width, channels)
        self.expected_source_dims = [
            (64, 64, 512),  # Conv4_3 normalized
            (32, 32, 1024),  # Conv7
            (16, 16, 512),  # Extra 1
            (8, 8, 256),  # Extra 2
            (4, 4, 256),  # Extra 3
            (2, 2, 256),  # Extra 4
            (1, 1, 256),  # Extra 5
        ]

        # Build heads immediately with expected dimensions
        self.loc_heads = []
        self.conf_heads = []

        all_channels = vgg_channels + extra_channels
        for idx in range(len(all_channels)):
            if idx < len(self.expected_source_dims):
                input_h, input_w, in_channels = self.expected_source_dims[idx]
            else:
                # Fallback for unexpected sources
                input_h, input_w, in_channels = 1, 1, all_channels[idx]

            num_boxes = self.mbox_cfg[idx] if idx < len(self.mbox_cfg) else self.mbox_cfg[-1]

            loc_out_channels = num_boxes * 4
            conf_out_channels = num_boxes * self.num_classes

            loc_config = HeadConfig(in_channels, loc_out_channels, 3, 1, 1)
            loc_head = ConvHead(input_h, input_w, loc_config, self.batch_size, self.loc_parameters[idx], self.device)
            self.loc_heads.append(loc_head)

            conf_config = HeadConfig(in_channels, conf_out_channels, 3, 1, 1)
            conf_head = ConvHead(input_h, input_w, conf_config, self.batch_size, self.conf_parameters[idx], self.device)
            self.conf_heads.append(conf_head)

    def __call__(self, sources):
        import torch

        # Heads are already built in __init__, no need to rebuild
        loc_preds = []
        conf_preds = []

        for idx, source in enumerate(sources):
            if isinstance(source, torch.Tensor):
                _, _, input_h, input_w = source.shape
                tensor_size = source.numel() * 2

                # Use L1 for smaller feature maps, DRAM for larger ones
                memory_config = (
                    ttnn.L1_MEMORY_CONFIG
                    if (input_h <= 64 and input_w <= 64 and tensor_size <= 2 * 1024 * 1024)
                    else ttnn.DRAM_MEMORY_CONFIG
                )

                source = source.permute(0, 2, 3, 1)
                source = ttnn.from_torch(
                    source,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                )

            # Use pre-built heads
            if idx < len(self.loc_heads):
                loc_pred = self.loc_heads[idx](source)
                conf_pred = self.conf_heads[idx](source)
            else:
                raise IndexError(f"Source index {idx} exceeds number of built heads ({len(self.loc_heads)})")

            loc_preds.append(loc_pred)
            conf_preds.append(conf_pred)

        return loc_preds, conf_preds


def multibox_heads(
    size: int = 512,
    num_classes: int = 21,
    batch_size: int = 1,
    loc_parameters: list = None,
    conf_parameters: list = None,
    vgg_channels: list = None,
    extra_channels: list = None,
    device=None,
) -> TtMultiboxHeads:
    import torch

    if vgg_channels is None:
        vgg_channels = [512, 1024]

    if extra_channels is None:
        extra_channels = [512, 256, 256, 256, 256, 256] if size == 512 else [512, 256, 256, 256]

    mbox_cfg = {
        "300": [4, 6, 6, 6, 4, 4],
        "512": [4, 6, 6, 6, 4, 4, 4],
    }[str(size)]

    total_heads = len(vgg_channels) + len(extra_channels)

    if loc_parameters is None:
        loc_parameters = []
        conf_parameters = []

        all_channels = vgg_channels + extra_channels

        for idx, in_channels in enumerate(all_channels):
            num_boxes = mbox_cfg[idx] if idx < len(mbox_cfg) else mbox_cfg[-1]

            loc_out_channels = num_boxes * 4
            dummy_loc_weight = torch.zeros(loc_out_channels, in_channels, 3, 3)
            dummy_loc_bias = torch.zeros(loc_out_channels)
            loc_parameters.append({"weight": dummy_loc_weight, "bias": dummy_loc_bias})

            conf_out_channels = num_boxes * num_classes
            dummy_conf_weight = torch.zeros(conf_out_channels, in_channels, 3, 3)
            dummy_conf_bias = torch.zeros(conf_out_channels)
            conf_parameters.append({"weight": dummy_conf_weight, "bias": dummy_conf_bias})

    return TtMultiboxHeads(
        size, num_classes, batch_size, loc_parameters, conf_parameters, vgg_channels, extra_channels, device
    )


def build_multibox_heads(
    size: int = 512,
    num_classes: int = 21,
    batch_size: int = 1,
    loc_parameters: list = None,
    conf_parameters: list = None,
    vgg_channels: list = None,
    extra_channels: list = None,
    device=None,
) -> TtMultiboxHeads:
    return multibox_heads(
        size, num_classes, batch_size, loc_parameters, conf_parameters, vgg_channels, extra_channels, device
    )
