# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from models.experimental.mapTR.reference.utils import (
    inverse_sigmoid,
    bbox_xyxy_to_cxcywh,
    denormalize_2d_bbox,
    denormalize_2d_pts,
)


# Placeholder loss modules for weight loading compatibility
# These are not used during inference but are needed to match the original model structure
class FocalLoss(nn.Module):
    """Placeholder FocalLoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()
        self.use_sigmoid = kwargs.get("use_sigmoid", True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("FocalLoss is only for weight loading, not inference")


class L1Loss(nn.Module):
    """Placeholder L1Loss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("L1Loss is only for weight loading, not inference")


class GIoULoss(nn.Module):
    """Placeholder GIoULoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("GIoULoss is only for weight loading, not inference")


class PtsL1Loss(nn.Module):
    """Placeholder PtsL1Loss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PtsL1Loss is only for weight loading, not inference")


class PtsDirCosLoss(nn.Module):
    """Placeholder PtsDirCosLoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PtsDirCosLoss is only for weight loading, not inference")


class MapTRHead(nn.Module):
    """MapTR Head for map element detection (inference-only).

    This is a standalone PyTorch implementation derived from the original
    MapTRHead in projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py.

    Args:
        transformer (nn.Module): The transformer module (MapTRPerceptionTransformer).
        positional_encoding (nn.Module): Positional encoding module.
        bbox_coder (nn.Module, optional): Bbox coder for decoding predictions.
        embed_dims (int): Embedding dimensions. Default: 256.
        num_classes (int): Number of classes. Default: 3.
        num_reg_fcs (int): Number of FC layers in regression branch. Default: 2.
        num_cls_fcs (int): Number of FC layers in classification branch. Default: 2.
        code_size (int): Size of the output code (num_pts * 2). Default: 2.
        bev_h (int): Height of BEV feature. Default: 200.
        bev_w (int): Width of BEV feature. Default: 100.
        pc_range (List[float]): Point cloud range. Default: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0].
        num_vec (int): Number of vectors (instances). Default: 50.
        num_pts_per_vec (int): Number of points per vector. Default: 20.
        num_pts_per_gt_vec (int): Number of points per GT vector. Default: 20.
        query_embed_type (str): Type of query embedding ('all_pts' or 'instance_pts'). Default: 'instance_pts'.
        transform_method (str): Method to transform points to bbox. Default: 'minmax'.
        with_box_refine (bool): Whether to use box refinement. Default: True.
        as_two_stage (bool): Whether to use two-stage detection. Default: False.
        bev_encoder_type (str): Type of BEV encoder. Default: 'BEVFormerEncoder'.
        dir_interval (int): Interval for direction loss. Default: 1.
    """

    def __init__(
        self,
        transformer: nn.Module,
        positional_encoding: nn.Module,
        bbox_coder: Optional[nn.Module] = None,
        embed_dims: int = 256,
        num_classes: int = 3,
        num_reg_fcs: int = 2,
        num_cls_fcs: int = 2,
        code_size: int = 2,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        num_pts_per_gt_vec: int = 20,
        query_embed_type: str = "instance_pts",
        transform_method: str = "minmax",
        with_box_refine: bool = True,
        as_two_stage: bool = False,
        bev_encoder_type: str = "BEVFormerEncoder",
        dir_interval: int = 1,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.bbox_coder = bbox_coder
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
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_query = num_vec * num_pts_per_vec
        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = bev_encoder_type
        self.dir_interval = dir_interval

        # Loss functions (placeholders for weight loading compatibility)
        self.loss_cls = FocalLoss(use_sigmoid=True)
        self.loss_bbox = L1Loss()
        self.loss_iou = GIoULoss()
        self.loss_pts = PtsL1Loss()
        self.loss_dir = PtsDirCosLoss()

        # Activation (for weight loading compatibility)
        self.activate = nn.ReLU(inplace=True)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head.

        This follows the structure from the reference implementation.
        """
        # Classification branch (with LayerNorm as in reference)
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        # Regression branch (without LayerNorm as in reference)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        # Number of prediction layers
        # Last reg_branch is used to generate proposal from encode feature map
        # when as_two_stage is True.
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
                with shape (B, N, C, H, W) where N is the number of cameras.
            lidar_feat: LiDAR features (optional).
            img_metas: Image meta information containing 'can_bus' for temporal fusion.
            prev_bev: Previous BEV features for temporal fusion. Default: None.
            only_bev: Only compute BEV features with encoder. Default: False.

        Returns:
            Dictionary containing:
                - bev_embed: BEV embeddings with shape (bs, bev_h*bev_w, embed_dims).
                - all_cls_scores: Classification scores from all decoder layers
                    with shape (num_dec, bs, num_vec, num_classes).
                - all_bbox_preds: Bounding box predictions from all decoder layers
                    with shape (num_dec, bs, num_vec, 4) in cxcywh format.
                - all_pts_preds: Point predictions from all decoder layers
                    with shape (num_dec, bs, num_vec, num_pts_per_vec, 2).
                - enc_cls_scores: Encoder classification scores (None for single stage).
                - enc_bbox_preds: Encoder bbox predictions (None for single stage).
                - enc_pts_preds: Encoder pts predictions (None for single stage).
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
        hs = hs.permute(0, 2, 1, 3)  # (num_dec, bs, num_query, embed_dims)

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # Classification: aggregate over points per instance
            # hs[lvl] shape: (bs, num_query, embed_dims) = (bs, num_vec * num_pts_per_vec, embed_dims)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))

            # Regression: predict offset from reference points
            tmp = self.reg_branches[lvl](hs[lvl])

            # Add reference points
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()  # Normalize to [0, 1]

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
            pts: Input points with shape (bs, num_query, code_size) where
                code_size = 2 (x, y coordinates for each point).
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

        This method uses the bbox_coder if available, otherwise decodes manually.

        Args:
            preds_dicts: Prediction results dictionary from forward().
            img_metas: Image meta information.
            rescale: Whether to rescale predictions. Default: False.

        Returns:
            List of [bboxes, scores, labels, pts] for each sample in the batch.
        """
        # Use bbox_coder if available (follows reference implementation pattern)
        if self.bbox_coder is not None:
            preds_dicts_decoded = self.bbox_coder.decode(preds_dicts)

            num_samples = len(preds_dicts_decoded)
            ret_list = []
            for i in range(num_samples):
                preds = preds_dicts_decoded[i]
                bboxes = preds["bboxes"]
                scores = preds["scores"]
                labels = preds["labels"]
                pts = preds["pts"]
                ret_list.append([bboxes, scores, labels, pts])

            return ret_list

        # Manual decoding (fallback for when bbox_coder is not provided)
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
