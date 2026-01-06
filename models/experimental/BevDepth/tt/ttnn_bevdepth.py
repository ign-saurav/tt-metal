# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations


class TtBEVDepth:
    def __init__(
        self,
        device,
        backbone_parameters,
        neck_parameters,
        depthnet_parameters,
        head_parameters,
        lss_conf=None,
        model_config=None,
    ):
        self.device = device

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.backbone = TtBaseLSSFPN(
            device=device,
            backbone_parameters=backbone_parameters,
            neck_parameters=neck_parameters,
            depthnet_parameters=depthnet_parameters,
            lss_conf=lss_conf,
            model_config=self.model_config,
        )

        head_model_config = {
            "MATH_FIDELITY": self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            "ACTIVATIONS_DTYPE": self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            "WEIGHTS_DTYPE": self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
        }
        self.head = TtBEVDepthHead(
            parameters=head_parameters,
            model_config=head_model_config,
            layer_optimisations=head_optimisations,
            device=self.device,
        )

    def __call__(self, x, mats_dict, timestamps=None):
        bev_feature = self.backbone(x, mats_dict, timestamps, is_return_depth=False)

        if isinstance(bev_feature, torch.Tensor):
            bev_feature_ttnn = ttnn.from_torch(
                bev_feature.permute(0, 2, 3, 1),
                dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
            )
            bev_feature_ttnn = ttnn.to_device(bev_feature_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            bev_feature_ttnn = bev_feature

        preds = self.head(bev_feature_ttnn, device=self.device)
        return preds
