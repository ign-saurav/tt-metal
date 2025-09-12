# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Dict

from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.decoder import TTDecoder, decoder_layer_optimisations


class TTPanopticDeepLab:
    """
    TTNN implementation of Panoptic DeepLab using backbone and decoder architecture.
    Combines backbone, semantic segmentation, and instance segmentation.
    """

    def __init__(
        self,
        parameters,
        model_config,
    ):
        self.model_config = model_config

        # Initialize backbone
        self.backbone = TTBackbone(parameters.backbone, model_config)

        # Initialize semantic segmentation decoder
        self.semantic_decoder = TTDecoder(
            parameters.semantic_decoder,
            model_config,
            layer_optimisations=decoder_layer_optimisations["Semantics_head"],
            name="Semantics_head",
        )

        # Initialize instance segmentation decoder
        self.instance_decoder = TTDecoder(
            parameters.instance_decoder,
            model_config,
            layer_optimisations=decoder_layer_optimisations["instance_head"],
            name="instance_head",
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        device,
    ) -> Dict[str, ttnn.Tensor]:
        """
        Forward pass of TTNN Panoptic DeepLab.

        Args:
            x: Input tensor of shape [B, H, W, C] in TTNN format
            device: TTNN device

        Returns:
            semantic_logits: Semantic segmentation logits
            instance_logit: Instance segmentation logits - offset head
            instance_logit_2: Instance segmentation logits - center head
        """

        # Extract features from backbone
        features = self.backbone(x, device)

        # Extract the specific feature maps the decoders expect
        backbone_features = features["res_5"]
        res3_features = features["res_3"]
        res2_features = features["res_2"]

        backbone_feature_instance_decoder = ttnn.clone(backbone_features)
        res3_feature_instance_decoder = ttnn.clone(res3_features)
        res2_feature_instance_decoder = ttnn.clone(res2_features)

        # Semantic segmentation branch
        semantic_logit, _ = self.semantic_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,
            device=device,
        )

        # Instance segmentation branch

        instance_offset_head_logit, instance_center_head_logit = self.instance_decoder(
            backbone_feature_instance_decoder,
            res3_feature_instance_decoder,
            res2_feature_instance_decoder,
            upsample_channels=256,
            device=device,
        )

        return semantic_logit, instance_offset_head_logit, instance_center_head_logit
