# SPDX-FileCopyrightText: © 2025
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from models.experimental.bevformerv2.tt.tt_bottleneck import TtBottleneck
from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig
from models.experimental.bevformerv2.tt.config import TtResNet50Configs


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
        configs: Optional[TtResNet50Configs] = None,
    ):
        self.device = device

        # Use provided configs or build them inline
        if configs is not None:
            self.configs = configs
        else:
            from models.experimental.bevformerv2.tt.config import create_resnet50_configs

            self.configs = create_resnet50_configs(conv_args, conv_pth, device, model_configs)

        # ------------------------
        # Stem
        # ------------------------
        self.conv1 = TtConv2d(self.configs.stem.conv1, device)
        self.maxpool = TtMaxPool2d(self.configs.stem.maxpool, device)

        # ------------------------
        # Layer 1 (3 blocks)
        # ------------------------
        self.layer1_0 = TtBottleneck(
            device=device,
            configs=self.configs.layer1.bottlenecks[0],
        )
        self.layer1_1 = TtBottleneck(
            device=device,
            configs=self.configs.layer1.bottlenecks[1],
        )
        self.layer1_2 = TtBottleneck(
            device=device,
            configs=self.configs.layer1.bottlenecks[2],
        )

        # ------------------------
        # Layer 2 (4 blocks)
        # ------------------------
        self.layer2_0 = TtBottleneck(
            device=device,
            configs=self.configs.layer2.bottlenecks[0],
        )
        self.layer2_1 = TtBottleneck(
            device=device,
            configs=self.configs.layer2.bottlenecks[1],
        )
        self.layer2_2 = TtBottleneck(
            device=device,
            configs=self.configs.layer2.bottlenecks[2],
        )
        self.layer2_3 = TtBottleneck(
            device=device,
            configs=self.configs.layer2.bottlenecks[3],
        )

        # ------------------------
        # Layer 3 (6 blocks)
        # ------------------------
        self.layer3_0 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[0],
        )
        self.layer3_1 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[1],
        )
        self.layer3_2 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[2],
        )
        self.layer3_3 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[3],
        )
        self.layer3_4 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[4],
        )
        self.layer3_5 = TtBottleneck(
            device=device,
            configs=self.configs.layer3.bottlenecks[5],
        )

        # ------------------------
        # Layer 4 (3 blocks)
        # ------------------------
        self.layer4_0 = TtBottleneck(
            device=device,
            configs=self.configs.layer4.bottlenecks[0],
        )
        self.layer4_1 = TtBottleneck(
            device=device,
            configs=self.configs.layer4.bottlenecks[1],
        )
        self.layer4_2 = TtBottleneck(
            device=device,
            configs=self.configs.layer4.bottlenecks[2],
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
