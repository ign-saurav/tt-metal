# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead


class TtBEVDepth:
    """
    TTNN implementation of BEVDepth model.

    This class combines:
    - TtBaseLSSFPN: Backbone that processes images to BEV features
    - TtBEVDepthHead: BEV head for object detection predictions
    """

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
        """
        Initialize TTNN BEVDepth model for inference.

        Args:
            device: TTNN device
            backbone_parameters: Parameters for ResNet50 backbone
            neck_parameters: Parameters for SECONDFPN neck
            depthnet_parameters: Parameters for DepthNet
            head_parameters: Parameters for BEVDepthHead
            lss_conf: LSS configuration dict with:
                - x_bound: [min, max, step]
                - y_bound: [min, max, step]
                - z_bound: [min, max, step]
                - d_bound: [min, max, step]
                - final_dim: [H, W]
                - downsample_factor: int
                - output_channels: int
            model_config: Model configuration dict (dtype, math fidelity, etc.)
        """
        self.device = device

        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        # Initialize backbone
        self.backbone = TtBaseLSSFPN(
            device=device,
            backbone_parameters=backbone_parameters,
            neck_parameters=neck_parameters,
            depthnet_parameters=depthnet_parameters,
            lss_conf=lss_conf,
            model_config=self.model_config,
        )

        # Initialize head
        self.head = TtBEVDepthHead(
            parameters=head_parameters,
            model_config=self.model_config,
            checkpoint_path=None,
        )

    def __call__(self, x, mats_dict, timestamps=None):
        """
        Forward function for BEVDepth inference.

        Args:
            x (Tensor): Input images with shape (B, num_sweeps, num_cameras, 3, H, W).
            mats_dict (dict): Dictionary containing transformation matrices:
                - sensor2ego_mats: (B, num_sweeps, num_cameras, 4, 4)
                - intrin_mats: (B, num_sweeps, num_cameras, 4, 4)
                - ida_mats: (B, num_sweeps, num_cameras, 4, 4)
                - sensor2sensor_mats: (B, num_sweeps, num_cameras, 4, 4)
                - bda_mat: (B, 4, 4)
            timestamps (Tensor, optional): Timestamps with shape (B, num_sweeps, num_cameras).

        Returns:
            preds: Detection predictions
        """
        # Forward through backbone to get BEV features
        bev_feature = self.backbone(x, mats_dict, timestamps, is_return_depth=False)

        # Convert BEV feature to TTNN format for head
        # BEV feature is in NCHW format
        bev_feature_ttnn = ttnn.from_torch(
            bev_feature.permute(0, 2, 3, 1),  # NCHW -> NHWC
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        bev_feature_ttnn = ttnn.to_device(bev_feature_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Forward through head
        preds = self.head(bev_feature_ttnn, device=self.device)
        return preds
