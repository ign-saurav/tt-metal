# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class GridMask(nn.Module):
    """Grid mask augmentation for images (inference-only version).

    Args:
        use_h (bool): Whether to apply mask along height. Default: True.
        use_w (bool): Whether to apply mask along width. Default: True.
        rotate (int): Rotation angle range. Default: 1.
        offset (bool): Whether to use offset. Default: False.
        ratio (float): Ratio of grid size. Default: 0.5.
        mode (int): Mask mode. Default: 1.
        prob (float): Probability of applying grid mask. Default: 0.7.
    """

    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        prob: float = 0.7,
    ):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function - in inference mode, just return input unchanged."""
        # GridMask is typically only used during training
        # For inference, we just return the input unchanged
        return x


class MapTR(nn.Module):
    """MapTR detector for map element detection (inference-only).

    Args:
        img_backbone (nn.Module): Image backbone network.
        img_neck (nn.Module, optional): Image neck (FPN). Default: None.
        pts_bbox_head (nn.Module): Detection head.
        use_grid_mask (bool): Whether to use grid mask augmentation. Default: False.
        video_test_mode (bool): Whether to use temporal information. Default: False.
        modality (str): Input modality ('vision', 'lidar', 'fusion'). Default: 'vision'.
        lidar_encoder (nn.Module, optional): LiDAR encoder for fusion mode. Default: None.
    """

    def __init__(
        self,
        img_backbone: nn.Module,
        pts_bbox_head: nn.Module,
        img_neck: Optional[nn.Module] = None,
        use_grid_mask: bool = False,
        video_test_mode: bool = False,
        modality: str = "vision",
        lidar_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.pts_bbox_head = pts_bbox_head
        self.with_img_neck = img_neck is not None

        self.grid_mask = GridMask(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # Temporal settings
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.modality = modality
        self.lidar_encoder = lidar_encoder

    def extract_img_feat(
        self,
        img: torch.Tensor,
        img_metas: Optional[List[Dict]] = None,
        len_queue: Optional[int] = None,
    ) -> Optional[List[torch.Tensor]]:
        """Extract features from images.

        Args:
            img: Input images with shape (B, N, C, H, W) or (B, C, H, W).
            img_metas: Image meta information.
            len_queue: Length of temporal queue.

        Returns:
            List of feature maps at different scales.
        """
        # Handle different input shapes
        # 5D: (B, N, C, H, W) - batched multi-camera
        # 4D: (N, C, H, W) - single batch multi-camera (already indexed)
        if img is None:
            return None

        if img.dim() == 5:
            B = img.size(0)
            if B == 1:
                img = img.squeeze(0)  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
        elif img.dim() == 4:
            # Already (N, C, H, W) format - single batch
            B = 1
        else:
            raise ValueError(f"Unexpected image dimension: {img.dim()}")

        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # Reshape features
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
            List of feature maps.
        """
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward(
        self,
        img_metas: List[List[Dict]],
        img: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[Dict]:
        """Forward function for inference.

        Args:
            img_metas: List of image meta information.
            img: Input images. Default: None.
            points: Input point cloud (for fusion mode). Default: None.

        Returns:
            List of detection results.
        """
        return self.forward_test(img_metas=img_metas, img=img, points=points, **kwargs)

    def obtain_history_bev(
        self,
        imgs_queue: torch.Tensor,
        img_metas_list: List[List[Dict]],
    ) -> torch.Tensor:
        """Obtain history BEV features iteratively.

        Args:
            imgs_queue: Queue of images with shape (B, T, N, C, H, W).
            img_metas_list: List of image meta information for each timestamp.

        Returns:
            BEV features from the last timestamp.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0].get("prev_bev_exists", True):
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, None, img_metas, prev_bev, only_bev=True)

            return prev_bev

    def forward_test(
        self,
        img_metas: List[List[Dict]],
        img: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[Dict]:
        """Test function for inference.

        Args:
            img_metas: List of image meta information.
            img: Input images. Default: None.
            points: Input point cloud. Default: None.

        Returns:
            List of detection results.
        """
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        img = [img] if img is None else img
        points = [points] if points is None else points

        # Check if scene changed
        scene_token = img_metas[0][0].get("scene_token", None)
        if scene_token != self.prev_frame_info["scene_token"]:
            # First sample of each scene - reset temporal info
            self.prev_frame_info["prev_bev"] = None

        # Update scene token
        self.prev_frame_info["scene_token"] = scene_token

        # Disable temporal information if not in video test mode
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps
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

        # Save BEV features and ego motion for next frame
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def pred2result(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        pts: torch.Tensor,
        attrs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert detection results to dictionary.

        Args:
            bboxes: Bounding boxes.
            scores: Prediction scores.
            labels: Class labels.
            pts: Point predictions.
            attrs: Attributes (optional).

        Returns:
            Dictionary containing detection results.
        """
        result_dict = dict(
            boxes_3d=bboxes.cpu(),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.cpu(),
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def simple_test_pts(
        self,
        x: List[torch.Tensor],
        img_metas: List[Dict],
        prev_bev: Optional[torch.Tensor] = None,
        rescale: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Test function for point cloud branch.

        Args:
            x: List of feature maps.
            img_metas: Image meta information.
            prev_bev: Previous BEV features.
            rescale: Whether to rescale predictions.

        Returns:
            Tuple of (BEV features, list of detection results).
        """
        outs = self.pts_bbox_head(x, None, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]
        return outs["bev_embed"], bbox_results

    def simple_test(
        self,
        img_metas: List[Dict],
        img: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
        rescale: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict]]:
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

    def reset_temporal_state(self):
        """Reset temporal state for new sequence."""
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
