# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid.

    Args:
        x: Tensor with values in range (0, 1).
        eps: Small value for numerical stability.

    Returns:
        Tensor after inverse sigmoid.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def bbox_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox: Bounding boxes with shape (..., 4) in (x1, y1, x2, y2) format.

    Returns:
        Bounding boxes with shape (..., 4) in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = bbox.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def bbox_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox: Bounding boxes with shape (..., 4) in (cx, cy, w, h) format.

    Returns:
        Bounding boxes with shape (..., 4) in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = bbox.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def normalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Normalize 2D bboxes to [0, 1] range.

    Args:
        bboxes: Bounding boxes in xyxy format.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Normalized bounding boxes in cxcywh format.
    """
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Normalize 2D points to [0, 1] range.

    Args:
        pts: Points with shape (..., 2).
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Normalized points.
    """
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize 2D bboxes from [0, 1] range to original range.

    Args:
        bboxes: Normalized bounding boxes in cxcywh format.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Denormalized bounding boxes in xyxy format.
    """
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return bboxes


def denormalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize 2D points from [0, 1] range to original range.

    Args:
        pts: Normalized points with shape (..., 2).
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Denormalized points.
    """
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


class MapTRHead(nn.Module):
    """MapTR Head for map element detection (inference-only).

    Args:
        transformer (nn.Module): The transformer module.
        positional_encoding (nn.Module): Positional encoding module.
        embed_dims (int): Embedding dimensions. Default: 256.
        num_classes (int): Number of classes. Default: 3.
        num_reg_fcs (int): Number of FC layers in regression branch. Default: 2.
        num_cls_fcs (int): Number of FC layers in classification branch. Default: 2.
        code_size (int): Size of the output code. Default: 10.
        bev_h (int): Height of BEV feature. Default: 30.
        bev_w (int): Width of BEV feature. Default: 30.
        pc_range (List[float]): Point cloud range. Default: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0].
        num_vec (int): Number of vectors (instances). Default: 20.
        num_pts_per_vec (int): Number of points per vector. Default: 2.
        query_embed_type (str): Type of query embedding. Default: 'all_pts'.
        transform_method (str): Method to transform points to bbox. Default: 'minmax'.
        with_box_refine (bool): Whether to use box refinement. Default: False.
        as_two_stage (bool): Whether to use two-stage detection. Default: False.
        bev_encoder_type (str): Type of BEV encoder. Default: 'BEVFormerEncoder'.
    """

    def __init__(
        self,
        transformer: nn.Module,
        positional_encoding: nn.Module,
        embed_dims: int = 256,
        num_classes: int = 3,
        num_reg_fcs: int = 2,
        num_cls_fcs: int = 2,
        code_size: int = 10,
        bev_h: int = 30,
        bev_w: int = 30,
        pc_range: List[float] = None,
        num_vec: int = 20,
        num_pts_per_vec: int = 2,
        query_embed_type: str = "all_pts",
        transform_method: str = "minmax",
        with_box_refine: bool = False,
        as_two_stage: bool = False,
        bev_encoder_type: str = "BEVFormerEncoder",
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.num_cls_fcs = num_cls_fcs - 1
        self.code_size = code_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]

        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_query = num_vec * num_pts_per_vec
        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = bev_encoder_type

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        # Classification branch
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        # Regression branch
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        # Number of prediction layers
        num_pred = (
            (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        # BEV and query embeddings
        if not self.as_two_stage:
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None

            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        img_metas: List[Dict],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            mlvl_feats: Multi-level features from backbone, each is a 5D-tensor
                with shape (B, N, C, H, W).
            lidar_feat: LiDAR features (optional).
            img_metas: Image meta information.
            prev_bev: Previous BEV features for temporal fusion. Default: None.
            only_bev: Only compute BEV features with encoder. Default: False.

        Returns:
            Dictionary containing:
                - bev_embed: BEV embeddings.
                - all_cls_scores: Classification scores from all decoder layers.
                - all_bbox_preds: Bounding box predictions from all decoder layers.
                - all_pts_preds: Point predictions from all decoder layers.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # Prepare query embeddings
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        # Prepare BEV queries and positional encoding
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        # Only return BEV features if requested
        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        # Full forward pass through transformer
        outputs = self.transformer(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # Classification
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))

            # Regression
            tmp = self.reg_branches[lvl](hs[lvl])

            # Add reference points
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()

            # Transform to bbox and pts
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

        return outs

    def transform_box(
        self,
        pts: torch.Tensor,
        y_first: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert points set to bounding box.

        Args:
            pts: Input points with shape (bs, num_query, code_size).
            y_first: If True, point format is [y1, x1, y2, x2 ... yn, xn],
                otherwise [x1, y1, x2, y2 ... xn, yn]. Default: False.

        Returns:
            Tuple of (bbox, pts_reshape) where:
                - bbox: Bounding boxes with shape (bs, num_vec, 4) in cxcywh format.
                - pts_reshape: Reshaped points with shape (bs, num_vec, num_pts_per_vec, 2).
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]

        if self.transform_method == "minmax":
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError(f"transform_method '{self.transform_method}' not implemented")

        return bbox, pts_reshape

    def get_bboxes(
        self,
        preds_dicts: Dict[str, torch.Tensor],
        img_metas: List[Dict],
        rescale: bool = False,
    ) -> List[List]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts: Prediction results dictionary.
            img_metas: Image meta information.
            rescale: Whether to rescale predictions. Default: False.

        Returns:
            List of [bboxes, scores, labels, pts] for each sample.
        """
        # Get final predictions (last decoder layer)
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_pts_preds = preds_dicts["all_pts_preds"]

        # Use predictions from last layer
        cls_scores = all_cls_scores[-1]  # (bs, num_vec, num_classes)
        bbox_preds = all_bbox_preds[-1]  # (bs, num_vec, 4)
        pts_preds = all_pts_preds[-1]  # (bs, num_vec, num_pts_per_vec, 2)

        num_samples = cls_scores.shape[0]
        ret_list = []

        for i in range(num_samples):
            # Get scores and labels
            cls_score = cls_scores[i]  # (num_vec, num_classes)
            scores, labels = cls_score.max(dim=-1)  # (num_vec,), (num_vec,)

            # Denormalize bboxes
            bbox_pred = bbox_preds[i]  # (num_vec, 4)
            bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)

            # Denormalize pts
            pts_pred = pts_preds[i]  # (num_vec, num_pts_per_vec, 2)
            pts = denormalize_2d_pts(pts_pred, self.pc_range)

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
