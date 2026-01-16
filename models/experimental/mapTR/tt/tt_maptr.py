# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN MapTR model for map element detection (inference-only).

This module provides a complete TTNN implementation of the MapTR detector,
integrating the TTNN backbone (ResNet50), FPN neck, and MapTR head.
"""

import copy
import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple

from models.experimental.mapTR.tt.head import TtMapTRHead


class TtMapTR(nn.Module):
    """TTNN MapTR detector for map element detection (inference-only).

    This is a hybrid implementation that uses:
    - PyTorch backbone (ResNet50) for image feature extraction
    - PyTorch FPN for feature pyramid
    - TTNN head (TtMapTRHead) for efficient classification and regression

    Args:
        device: TTNN device for tensor operations.
        torch_backbone: PyTorch backbone module (ResNet50).
        torch_neck: PyTorch FPN module.
        head_params: Parameters for the TTNN detection head.
        transformer: PyTorch transformer module.
        positional_encoding: PyTorch positional encoding module.
        bbox_coder: Optional bbox coder for decoding predictions.
        use_grid_mask: Whether to use grid mask augmentation. Default: False.
        video_test_mode: Whether to use temporal information. Default: False.
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range.
        num_vec: Number of vectors. Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
    """

    def __init__(
        self,
        device: ttnn.Device,
        torch_backbone: nn.Module,
        torch_neck: nn.Module,
        head_params: dict,
        transformer: nn.Module,
        positional_encoding: nn.Module,
        bbox_coder: Optional[nn.Module] = None,
        use_grid_mask: bool = False,
        video_test_mode: bool = False,
        embed_dims: int = 256,
        num_classes: int = 3,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        num_decoder_layers: int = 6,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.device = device
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.pc_range = pc_range

        # PyTorch backbone and FPN (used for feature extraction)
        self.img_backbone = torch_backbone
        self.img_neck = torch_neck
        self.with_img_neck = torch_neck is not None

        # Initialize TTNN Head
        self.pts_bbox_head = TtMapTRHead(
            params=head_params,
            device=device,
            transformer=transformer,
            positional_encoding=positional_encoding,
            bbox_coder=bbox_coder,
            embed_dims=embed_dims,
            num_classes=num_classes,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
            num_vec=num_vec,
            num_pts_per_vec=num_pts_per_vec,
            num_decoder_layers=num_decoder_layers,
        )

        # Temporal state
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def extract_img_feat(
        self,
        img: torch.Tensor,
        img_metas: Optional[List[Dict]] = None,
        len_queue: Optional[int] = None,
    ) -> Optional[List[torch.Tensor]]:
        """Extract features from images using PyTorch backbone and FPN.

        Args:
            img: Input images as torch tensor with shape (B, N, C, H, W).
            img_metas: Image meta information.
            len_queue: Length of temporal queue.

        Returns:
            List of feature maps at different scales in (B, N, C, H, W) format.
        """
        if img is None:
            return None

        # Handle different input shapes
        if img.dim() == 5:
            B = img.size(0)
            if B == 1:
                img = img.squeeze(0)  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
        elif img.dim() == 4:
            B = 1
        else:
            raise ValueError(f"Unexpected image dimension: {img.dim()}")

        # Backbone forward (PyTorch)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        # FPN forward (PyTorch)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # Reshape features to (B, N, C, H, W) format
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def extract_feat(
        self,
        img: torch.Tensor,
        img_metas: Optional[List[Dict]] = None,
        len_queue: Optional[int] = None,
    ) -> Optional[List[torch.Tensor]]:
        """Extract features from images.

        Args:
            img: Input images.
            img_metas: Image meta information.
            len_queue: Length of temporal queue.

        Returns:
            List of feature maps in (B, N, C, H, W) format.
        """
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    def forward(
        self,
        img_metas: List[List[Dict]],
        img: Optional[ttnn.Tensor] = None,
        points: Optional[ttnn.Tensor] = None,
        **kwargs,
    ) -> List[Dict]:
        """Forward function for inference.

        Args:
            img_metas: List of image meta information.
            img: Input images.
            points: Input point cloud (optional).

        Returns:
            List of detection results.
        """
        return self.forward_test(img_metas=img_metas, img=img, points=points, **kwargs)

    def forward_test(
        self,
        img_metas: List[List[Dict]],
        img: Optional[ttnn.Tensor] = None,
        points: Optional[ttnn.Tensor] = None,
        **kwargs,
    ) -> List[Dict]:
        """Test function for inference.

        Args:
            img_metas: List of image meta information.
            img: Input images.
            points: Input point cloud.

        Returns:
            List of detection results.
        """
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        img = [img] if img is None else img
        points = [points] if points is None else points

        # Check scene change
        scene_token = img_metas[0][0].get("scene_token", None)
        if scene_token != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = scene_token

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Handle can_bus for temporal fusion
        can_bus = img_metas[0][0].get("can_bus", None)
        if can_bus is not None:
            tmp_pos = copy.deepcopy(can_bus[:3])
            tmp_angle = copy.deepcopy(can_bus[-1])
            if self.prev_frame_info["prev_bev"] is not None:
                img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
                img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
            else:
                img_metas[0][0]["can_bus"][-1] = 0
                img_metas[0][0]["can_bus"][:3] = 0
        else:
            tmp_pos = 0
            tmp_angle = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas: List[Dict],
        img: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        prev_bev: Optional[ttnn.Tensor] = None,
        rescale: bool = False,
        **kwargs,
    ) -> Tuple[ttnn.Tensor, List[Dict]]:
        """Test function without augmentation.

        Args:
            img_metas: Image meta information.
            img: Input images.
            points: Input point cloud.
            prev_bev: Previous BEV features.
            rescale: Whether to rescale predictions.

        Returns:
            Tuple of (new BEV features, list of detection results).
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x: List[torch.Tensor],
        img_metas: List[Dict],
        prev_bev: Optional[ttnn.Tensor] = None,
        rescale: bool = False,
    ) -> Tuple[ttnn.Tensor, List[Dict]]:
        """Test function for point cloud branch.

        Args:
            x: List of feature maps in (B, N, C, H, W) format.
            img_metas: Image meta information.
            prev_bev: Previous BEV features.
            rescale: Whether to rescale predictions.

        Returns:
            Tuple of (BEV features, list of detection results).
        """
        # Convert PyTorch features to TTNN format for the head
        # x is in (B, N, C, H, W) format, head expects (B*N, H, W, C) in NHWC
        tt_feats = []
        for feat in x:
            B, N, C, H, W = feat.shape
            # Reshape to (BN, C, H, W) then permute to (BN, H, W, C)
            feat_reshaped = feat.reshape(B * N, C, H, W).permute(0, 2, 3, 1).contiguous()
            feat_ttnn = ttnn.from_torch(
                feat_reshaped, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            tt_feats.append((feat_ttnn,))  # Wrap in tuple to match expected format

        outs = self.pts_bbox_head(tt_feats, None, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]
        return outs["bev_embed"], bbox_results

    def pred2result(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        pts: torch.Tensor,
        attrs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert detection results to dictionary."""
        result_dict = dict(
            boxes_3d=bboxes.cpu(),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.cpu(),
        )
        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()
        return result_dict

    def reset_temporal_state(self):
        """Reset temporal state for new sequence."""
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
