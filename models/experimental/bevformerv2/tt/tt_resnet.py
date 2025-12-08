# SPDX-FileCopyrightText: © 2025
# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn import UnaryWithParam, UnaryOpType

from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from models.experimental.bevformerv2.tt.utils import create_conv2d_configuration, create_maxpool2d_configuration
from models.experimental.bevformerv2.tt.tt_bottleneck import TtBottleneck, get_bottleneck_optimisation
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig


class TtResNet50_MMD_C345:
    """
    TTNN implementation of MMDet-style ResNet50 returning C3, C4, C5:
      - C3: output after layer2 (1/8)
      - C4: output after layer3 (1/16)
      - C5: output after layer4 (1/32)
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        *,
        model_configs: BevFormerV2ModelConfig | None = None,
    ):
        self.device = device

        # ------------------------
        # Stem
        # ------------------------
        conv1_config = create_conv2d_configuration(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=UnaryWithParam(UnaryOpType.RELU),
            act_block_h=32,
            model_configs=model_configs,
            layer_path="stem.conv1",
        )
        self.conv1 = TtConv2d(conv1_config, device)

        # ------------------------
        # MaxPool (after stem conv1)
        # ------------------------
        # Get channels from conv1 output (out_channels)
        conv1_channels = conv1_config.out_channels
        maxpool_config = create_maxpool2d_configuration(
            conv_args.maxpool,
            channels=conv1_channels,
        )
        self.maxpool = TtMaxPool2d(maxpool_config, device)

        # ------------------------
        # Layer 1 (3 blocks)
        # ------------------------
        layer1_optimisations = get_bottleneck_optimisation("layer1")
        self.layer1_0 = TtBottleneck(
            conv_args.layer1[0],
            conv_pth.layer1_0,
            device,
            is_downsample=True,
            model_configs=model_configs,
            block_path="layer1.0",
            layer_optimisations=layer1_optimisations,
        )
        self.layer1_1 = TtBottleneck(
            conv_args.layer1[1],
            conv_pth.layer1_1,
            device,
            model_configs=model_configs,
            block_path="layer1.1",
            layer_optimisations=layer1_optimisations,
        )
        self.layer1_2 = TtBottleneck(
            conv_args.layer1[2],
            conv_pth.layer1_2,
            device,
            model_configs=model_configs,
            block_path="layer1.2",
            layer_optimisations=layer1_optimisations,
        )

        # ------------------------
        # Layer 2 (4 blocks)
        # ------------------------
        layer2_optimisations = get_bottleneck_optimisation("layer2")
        self.layer2_0 = TtBottleneck(
            conv_args.layer2[0],
            conv_pth.layer2_0,
            device,
            is_downsample=True,
            model_configs=model_configs,
            block_path="layer2.0",
            layer_optimisations=layer2_optimisations,
        )
        self.layer2_1 = TtBottleneck(
            conv_args.layer2[1],
            conv_pth.layer2_1,
            device,
            model_configs=model_configs,
            block_path="layer2.1",
            layer_optimisations=layer2_optimisations,
        )
        self.layer2_2 = TtBottleneck(
            conv_args.layer2[2],
            conv_pth.layer2_2,
            device,
            model_configs=model_configs,
            block_path="layer2.2",
            layer_optimisations=layer2_optimisations,
        )
        self.layer2_3 = TtBottleneck(
            conv_args.layer2[3],
            conv_pth.layer2_3,
            device,
            model_configs=model_configs,
            block_path="layer2.3",
            layer_optimisations=layer2_optimisations,
        )

        # ------------------------
        # Layer 3 (6 blocks)
        # ------------------------
        layer3_optimisations = get_bottleneck_optimisation("layer3")
        self.layer3_0 = TtBottleneck(
            conv_args.layer3[0],
            conv_pth.layer3_0,
            device,
            is_downsample=True,
            model_configs=model_configs,
            block_path="layer3.0",
            layer_optimisations=layer3_optimisations,
        )
        self.layer3_1 = TtBottleneck(
            conv_args.layer3[1],
            conv_pth.layer3_1,
            device,
            model_configs=model_configs,
            block_path="layer3.1",
            layer_optimisations=layer3_optimisations,
        )
        self.layer3_2 = TtBottleneck(
            conv_args.layer3[2],
            conv_pth.layer3_2,
            device,
            model_configs=model_configs,
            block_path="layer3.2",
            layer_optimisations=layer3_optimisations,
        )
        self.layer3_3 = TtBottleneck(
            conv_args.layer3[3],
            conv_pth.layer3_3,
            device,
            model_configs=model_configs,
            block_path="layer3.3",
            layer_optimisations=layer3_optimisations,
        )
        self.layer3_4 = TtBottleneck(
            conv_args.layer3[4],
            conv_pth.layer3_4,
            device,
            model_configs=model_configs,
            block_path="layer3.4",
            layer_optimisations=layer3_optimisations,
        )
        self.layer3_5 = TtBottleneck(
            conv_args.layer3[5],
            conv_pth.layer3_5,
            device,
            model_configs=model_configs,
            block_path="layer3.5",
            layer_optimisations=layer3_optimisations,
        )

        # ------------------------
        # Layer 4 (3 blocks)
        # ------------------------
        layer4_optimisations = get_bottleneck_optimisation("layer4")
        self.layer4_0 = TtBottleneck(
            conv_args.layer4[0],
            conv_pth.layer4_0,
            device,
            is_downsample=True,
            model_configs=model_configs,
            block_path="layer4.0",
            layer_optimisations=layer4_optimisations,
        )
        self.layer4_1 = TtBottleneck(
            conv_args.layer4[1],
            conv_pth.layer4_1,
            device,
            model_configs=model_configs,
            block_path="layer4.1",
            layer_optimisations=layer4_optimisations,
        )
        self.layer4_2 = TtBottleneck(
            conv_args.layer4[2],
            conv_pth.layer4_2,
            device,
            model_configs=model_configs,
            block_path="layer4.2",
            layer_optimisations=layer4_optimisations,
        )

    # ------------------------------
    # Forward Pass returning [C3, C4, C5]
    # ------------------------------
    def __call__(self, x, batch_size=1):
        outputs = []

        # Stem: conv1
        x = self.conv1(x)
        x = ttnn.sharded_to_interleaved(x)

        # MaxPool using TtMaxPool2d
        x = self.maxpool(x)

        # Layer1
        x = self.layer1_0(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.layer1_1(x)
        x = self.layer1_2(x)

        # Layer2 -> C3
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        c3 = ttnn.clone(x)
        outputs.append(c3)  # C3 appended (1/8 spatial)

        # Layer3 -> C4
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        c4 = ttnn.clone(x)
        outputs.append(c4)  # C4 appended (1/16 spatial)

        # Layer4 -> C5
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        c5 = ttnn.clone(x)
        outputs.append(c5)  # C5 appended (1/32 spatial)

        ttnn.deallocate(x)

        return outputs  # [C3, C4, C5]
