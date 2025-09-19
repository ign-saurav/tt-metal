# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Tuple

from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel


class TorchPanopticDeepLab(nn.Module):
    """
    Panoptic DeepLab model using modular decoder architecture.
    Combines semantic segmentation and instance segmentation with panoptic fusion.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        # Backbone
        self.backbone = ResNet52BackBone()

        # Semantic segmentation decoder
        self.semantic_decoder = DecoderModel(
            name="semantic_decoder",
        )

        # Instance segmentation decoders
        self.instance_decoder = DecoderModel(
            name="instance_decoder",
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Panoptic DeepLab.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            semantic_logits: Semantic segmentation logits
            instance_offset_head_logits: Instance segmentation logits - offset head
            instance_center_head_logits: Instance segmentation logits - center head
        """

        # Extract features from backbone
        features = self.backbone(x)

        # Extract specific feature maps
        backbone_features = features["res_5"]
        res3_features = features["res_3"]
        res2_features = features["res_2"]

        # Semantic segmentation branch
        semantic_logits, _ = self.semantic_decoder(backbone_features, res3_features, res2_features)

        # Instance segmentation branch
        instance_offset_head_logits, instance_center_head_logits = self.instance_decoder(
            backbone_features, res3_features, res2_features
        )

        return semantic_logits, instance_offset_head_logits, instance_center_head_logits
