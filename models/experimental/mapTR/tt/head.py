# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MapTRHead Implementation

This module provides a TTNN implementation of the MapTRHead for map element detection.
It is designed for inference-only use and supports loading pretrained weights from the
PyTorch reference implementation.

The implementation includes:
- 6 separate classification branches (one per decoder layer)
- 6 separate regression branches (one per decoder layer)
- BEV and query embeddings
- Box transformation utilities
"""

import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple

from models.experimental.mapTR.reference.utils import (
    denormalize_2d_bbox,
    denormalize_2d_pts,
)


# Placeholder loss modules for weight loading compatibility
class TtFocalLoss(nn.Module):
    """Placeholder FocalLoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()
        self.use_sigmoid = kwargs.get("use_sigmoid", True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("FocalLoss is only for weight loading, not inference")


class TtL1Loss(nn.Module):
    """Placeholder L1Loss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("L1Loss is only for weight loading, not inference")


class TtGIoULoss(nn.Module):
    """Placeholder GIoULoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("GIoULoss is only for weight loading, not inference")


class TtPtsL1Loss(nn.Module):
    """Placeholder PtsL1Loss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PtsL1Loss is only for weight loading, not inference")


class TtPtsDirCosLoss(nn.Module):
    """Placeholder PtsDirCosLoss for weight loading compatibility."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("PtsDirCosLoss is only for weight loading, not inference")


def ttnn_inverse_sigmoid(x: ttnn.Tensor, eps: float = 1e-5) -> ttnn.Tensor:
    """Compute inverse sigmoid using TTNN operations.

    inverse_sigmoid(x) = log(x / (1 - x))

    With clamping to avoid numerical instability:
    x_clamped = clamp(x, eps, 1 - eps)
    result = log(x_clamped / (1 - x_clamped))

    Args:
        x: Input tensor with values in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        Inverse sigmoid of input tensor.
    """
    # Clamp x to [eps, 1-eps] to avoid log(0) or log(inf)
    x_clamped = ttnn.clip(x, min=eps, max=1.0 - eps)

    # Compute 1 - x
    one_minus_x = ttnn.subtract(1.0, x_clamped)

    # Compute x / (1 - x)
    ratio = ttnn.div(x_clamped, one_minus_x)

    # Compute log(ratio)
    result = ttnn.log(ratio)

    return result


def ttnn_bbox_xyxy_to_cxcywh(bbox: ttnn.Tensor) -> ttnn.Tensor:
    """Convert bounding boxes from xyxy to cxcywh format using TTNN operations.

    Args:
        bbox: Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
              Shape: (..., 4)

    Returns:
        Bounding boxes in cxcywh format (cx, cy, w, h).
        Shape: (..., 4)
    """
    # Split bbox into components
    # bbox[..., 0] = xmin, bbox[..., 1] = ymin, bbox[..., 2] = xmax, bbox[..., 3] = ymax
    x0 = bbox[..., 0:1]
    y0 = bbox[..., 1:2]
    x1 = bbox[..., 2:3]
    y1 = bbox[..., 3:4]

    # Compute center and size
    cx = ttnn.multiply(ttnn.add(x0, x1), 0.5)
    cy = ttnn.multiply(ttnn.add(y0, y1), 0.5)
    w = ttnn.subtract(x1, x0)
    h = ttnn.subtract(y1, y0)

    # Concatenate to form output
    result = ttnn.concat([cx, cy, w, h], dim=-1)
    return result


class TtMapTRHead(nn.Module):
    """TTNN MapTR Head for map element detection (inference-only).

    This is a TTNN implementation derived from the original MapTRHead.
    It supports loading pretrained weights and running inference on Tenstorrent hardware.

    Key features:
    - 6 separate classification branches (one per decoder layer)
    - 6 separate regression branches (one per decoder layer)
    - Classification branches include LayerNorm
    - Regression branches do not include LayerNorm

    Args:
        params: Dictionary containing pretrained weights extracted from checkpoint.
        device: TTNN device for tensor operations.
        transformer: The transformer module (MapTRPerceptionTransformer).
        positional_encoding: Positional encoding module.
        bbox_coder: Optional bbox coder for decoding predictions.
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        num_reg_fcs: Number of FC layers in each branch. Default: 2.
        num_cls_fcs: Number of FC layers in classification branch. Default: 2.
        code_size: Size of output code (2 for x, y). Default: 2.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range. Default: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0].
        num_vec: Number of vectors (instances). Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
        query_embed_type: Type of query embedding. Default: 'instance_pts'.
        transform_method: Method to transform points to bbox. Default: 'minmax'.
        with_box_refine: Whether to use box refinement. Default: True.
        as_two_stage: Whether to use two-stage detection. Default: False.
        bev_encoder_type: Type of BEV encoder. Default: 'BEVFormerEncoder'.
        dir_interval: Interval for direction loss. Default: 1.
    """

    def __init__(
        self,
        params: dict,
        device: ttnn.Device,
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
        num_decoder_layers: int = 6,
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

        self.device = device
        self.params = params
        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.bbox_coder = bbox_coder
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.num_cls_fcs = num_cls_fcs - 1  # Follows reference implementation
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
        self.num_decoder_layers = num_decoder_layers
        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = bev_encoder_type
        self.dir_interval = dir_interval

        # Loss functions (placeholders for weight loading compatibility)
        self.loss_cls = TtFocalLoss(use_sigmoid=True)
        self.loss_bbox = TtL1Loss()
        self.loss_iou = TtGIoULoss()
        self.loss_pts = TtPtsL1Loss()
        self.loss_dir = TtPtsDirCosLoss()

        # Activation (for weight loading compatibility with reference)
        self.activate = nn.ReLU(inplace=True)

        # Initialize TTNN layers from parameters
        self._init_ttnn_layers()

    def _init_ttnn_layers(self):
        """Initialize TTNN classification and regression branches from params.

        The pretrained weights have the following structure:
        - cls_branches.{layer_idx}.{sublayer_idx}.weight/bias
          - sublayer 0: Linear(256, 256)
          - sublayer 1: LayerNorm(256)
          - sublayer 2: ReLU (no weights)
          - sublayer 3: Linear(256, 256)
          - sublayer 4: LayerNorm(256)
          - sublayer 5: ReLU (no weights)
          - sublayer 6: Linear(256, num_classes)

        - reg_branches.{layer_idx}.{sublayer_idx}.weight/bias
          - sublayer 0: Linear(256, 256)
          - sublayer 1: ReLU (no weights)
          - sublayer 2: Linear(256, 256)
          - sublayer 3: ReLU (no weights)
          - sublayer 4: Linear(256, code_size)
        """
        # Number of prediction layers (same as decoder layers for single-stage)
        num_pred = self.num_decoder_layers

        # Classification branch parameters for each decoder layer
        # Structure: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]
        # Index mapping: 0=Linear, 1=LayerNorm, 2=ReLU, 3=Linear, 4=LayerNorm, 5=ReLU, 6=Linear
        self.cls_branches_weights = []  # List of lists: [layer_idx][fc_idx]
        self.cls_branches_biases = []
        self.cls_norm_weights = []
        self.cls_norm_biases = []
        self.cls_final_weights = []
        self.cls_final_biases = []

        for layer_idx in range(num_pred):
            layer_weights = []
            layer_biases = []
            norm_weights = []
            norm_biases = []

            for fc_idx in range(self.num_reg_fcs):
                # Linear layer indices: 0, 3 (for num_reg_fcs=2)
                linear_idx = fc_idx * 3
                # LayerNorm indices: 1, 4
                norm_idx = fc_idx * 3 + 1

                w_key = f"cls_branches.{layer_idx}.{linear_idx}.weight"
                b_key = f"cls_branches.{layer_idx}.{linear_idx}.bias"
                nw_key = f"cls_branches.{layer_idx}.{norm_idx}.weight"
                nb_key = f"cls_branches.{layer_idx}.{norm_idx}.bias"

                layer_weights.append(self._to_ttnn(self.params[w_key]))
                layer_biases.append(self._to_ttnn(self.params[b_key]))
                norm_weights.append(self._to_ttnn(self.params[nw_key]))
                norm_biases.append(self._to_ttnn(self.params[nb_key]))

            self.cls_branches_weights.append(layer_weights)
            self.cls_branches_biases.append(layer_biases)
            self.cls_norm_weights.append(norm_weights)
            self.cls_norm_biases.append(norm_biases)

            # Final classification layer (index 6 for num_reg_fcs=2)
            final_idx = self.num_reg_fcs * 3
            self.cls_final_weights.append(self._to_ttnn(self.params[f"cls_branches.{layer_idx}.{final_idx}.weight"]))
            self.cls_final_biases.append(self._to_ttnn(self.params[f"cls_branches.{layer_idx}.{final_idx}.bias"]))

        # Regression branch parameters for each decoder layer
        # Structure: [Linear + ReLU] * num_reg_fcs + [Linear]
        # Index mapping: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear
        self.reg_branches_weights = []
        self.reg_branches_biases = []
        self.reg_final_weights = []
        self.reg_final_biases = []

        for layer_idx in range(num_pred):
            layer_weights = []
            layer_biases = []

            for fc_idx in range(self.num_reg_fcs):
                # Linear layer indices: 0, 2 (for num_reg_fcs=2)
                linear_idx = fc_idx * 2

                w_key = f"reg_branches.{layer_idx}.{linear_idx}.weight"
                b_key = f"reg_branches.{layer_idx}.{linear_idx}.bias"

                layer_weights.append(self._to_ttnn(self.params[w_key]))
                layer_biases.append(self._to_ttnn(self.params[b_key]))

            self.reg_branches_weights.append(layer_weights)
            self.reg_branches_biases.append(layer_biases)

            # Final regression layer (index 4 for num_reg_fcs=2)
            final_idx = self.num_reg_fcs * 2
            self.reg_final_weights.append(self._to_ttnn(self.params[f"reg_branches.{layer_idx}.{final_idx}.weight"]))
            self.reg_final_biases.append(self._to_ttnn(self.params[f"reg_branches.{layer_idx}.{final_idx}.bias"]))

        # Embedding parameters
        if self.bev_encoder_type == "BEVFormerEncoder":
            self.bev_embedding = self._to_ttnn(self.params["bev_embedding.weight"])
        else:
            self.bev_embedding = None

        if self.query_embed_type == "all_pts":
            self.query_embedding = self._to_ttnn(self.params["query_embedding.weight"])
        elif self.query_embed_type == "instance_pts":
            self.instance_embedding = self._to_ttnn(self.params["instance_embedding.weight"])
            self.pts_embedding = self._to_ttnn(self.params["pts_embedding.weight"])

    def _to_ttnn(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Convert PyTorch tensor to TTNN tensor."""
        return ttnn.from_torch(tensor, device=self.device, dtype=ttnn.bfloat16)

    def _linear(self, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor = None) -> ttnn.Tensor:
        """TTNN linear layer implementation."""
        return ttnn.linear(input_tensor, weight, bias=bias)

    def _layer_norm(
        self, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float = 1e-5
    ) -> ttnn.Tensor:
        """TTNN layer norm implementation."""
        return ttnn.layer_norm(input_tensor, epsilon=eps, weight=weight, bias=bias)

    def _relu(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """TTNN ReLU activation."""
        return ttnn.relu(input_tensor)

    def _cls_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN classification branch implementation for a specific decoder layer.

        Structure: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]

        Args:
            input_tensor: Input tensor with shape (bs, num_vec, embed_dims).
            layer_idx: Index of the decoder layer (0-5).

        Returns:
            Classification logits with shape (bs, num_vec, num_classes).
        """
        hidden = input_tensor

        # Hidden layers: [Linear + LayerNorm + ReLU] * num_reg_fcs
        for fc_idx in range(self.num_reg_fcs):
            hidden = self._linear(
                hidden, self.cls_branches_weights[layer_idx][fc_idx], self.cls_branches_biases[layer_idx][fc_idx]
            )
            hidden = self._layer_norm(
                hidden, self.cls_norm_weights[layer_idx][fc_idx], self.cls_norm_biases[layer_idx][fc_idx]
            )
            hidden = self._relu(hidden)

        # Final classification layer
        output = self._linear(hidden, self.cls_final_weights[layer_idx], self.cls_final_biases[layer_idx])
        return output

    def _reg_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch implementation for a specific decoder layer.

        Structure: [Linear + ReLU] * num_reg_fcs + [Linear]

        Args:
            input_tensor: Input tensor with shape (bs, num_query, embed_dims).
            layer_idx: Index of the decoder layer (0-5).

        Returns:
            Regression output with shape (bs, num_query, code_size).
        """
        hidden = input_tensor

        # Hidden layers: [Linear + ReLU] * num_reg_fcs
        for fc_idx in range(self.num_reg_fcs):
            hidden = self._linear(
                hidden, self.reg_branches_weights[layer_idx][fc_idx], self.reg_branches_biases[layer_idx][fc_idx]
            )
            hidden = self._relu(hidden)

        # Final regression layer
        output = self._linear(hidden, self.reg_final_weights[layer_idx], self.reg_final_biases[layer_idx])
        return output

    def forward(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        img_metas: List[Dict],
        prev_bev: Optional[ttnn.Tensor] = None,
        only_bev: bool = False,
    ) -> Dict[str, ttnn.Tensor]:
        """TTNN forward function.

        Args:
            mlvl_feats: Multi-level features from backbone, each is a 5D-tensor
                with shape (B, N, C, H, W) where N is the number of cameras.
            lidar_feat: LiDAR features (optional).
            img_metas: Image meta information containing 'can_bus' for temporal fusion.
            prev_bev: Previous BEV features for temporal fusion.
            only_bev: Only compute BEV features with encoder.

        Returns:
            Dictionary containing:
                - bev_embed: BEV embeddings.
                - all_cls_scores: Classification scores from all decoder layers.
                - all_bbox_preds: Bounding box predictions from all decoder layers.
                - all_pts_preds: Point predictions from all decoder layers.
                - enc_cls_scores: Encoder classification scores (None for single stage).
                - enc_bbox_preds: Encoder bbox predictions (None for single stage).
                - enc_pts_preds: Encoder pts predictions (None for single stage).
        """
        bs = ttnn.get_shape(mlvl_feats[0])[0]

        # Prepare query embeddings
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding
        elif self.query_embed_type == "instance_pts":
            # Add instance and point embeddings
            pts_embeds = ttnn.unsqueeze(self.pts_embedding, 0)
            instance_embeds = ttnn.unsqueeze(self.instance_embedding, 1)
            object_query_embeds = ttnn.add(pts_embeds, instance_embeds)
            object_query_embeds = ttnn.reshape(object_query_embeds, (self.num_query, -1))

        # Prepare BEV queries and positional encoding
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding
            bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), dtype=ttnn.bfloat16, device=self.device)
            bev_pos = self.positional_encoding(bev_mask)
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
            reg_branches=None,
            cls_branches=None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs

        # Permute hidden states: (num_dec, bs, num_query, embed_dims)
        hs = ttnn.permute(hs, (0, 2, 1, 3))

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(self.num_decoder_layers):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            # Apply inverse sigmoid to reference using correct formula
            reference = ttnn_inverse_sigmoid(reference)

            # Classification: aggregate over points per instance
            # hs[lvl] shape: (bs, num_query, embed_dims) = (bs, num_vec * num_pts_per_vec, embed_dims)
            hs_reshaped = ttnn.reshape(hs[lvl], (bs, self.num_vec, self.num_pts_per_vec, -1))
            hs_mean = ttnn.mean(hs_reshaped, dim=2)
            outputs_class = self._cls_branch(hs_mean, lvl)

            # Regression: predict offset from reference points
            tmp = self._reg_branch(hs[lvl], lvl)

            # Add reference points (offset prediction)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference[..., 0:2]
            tmp_updated = ttnn.add(tmp_xy, ref_xy)
            tmp_updated = ttnn.sigmoid(tmp_updated)  # Normalize to [0, 1]

            # Transform to bbox and pts
            outputs_coord, outputs_pts_coord = self.transform_box(tmp_updated)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        # Stack outputs
        outputs_classes = ttnn.stack(outputs_classes, dim=0)
        outputs_coords = ttnn.stack(outputs_coords, dim=0)
        outputs_pts_coords = ttnn.stack(outputs_pts_coords, dim=0)

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
        pts: ttnn.Tensor,
        y_first: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Convert points set to bounding box using TTNN operations.

        Args:
            pts: Input points with shape (bs, num_query, 2) where each point
                has (x, y) coordinates normalized to [0, 1].
            y_first: If True, point format is [y, x], otherwise [x, y].

        Returns:
            Tuple of (bbox, pts_reshape) where:
                - bbox: Bounding boxes with shape (bs, num_vec, 4) in cxcywh format.
                - pts_reshape: Reshaped points with shape (bs, num_vec, num_pts_per_vec, 2).
        """
        # Reshape points: (bs, num_query, 2) -> (bs, num_vec, num_pts_per_vec, 2)
        pts_shape = ttnn.get_shape(pts)
        bs = pts_shape[0]
        pts_reshape = ttnn.reshape(pts, (bs, self.num_vec, self.num_pts_per_vec, 2))

        # Extract x and y coordinates
        if y_first:
            pts_y = pts_reshape[:, :, :, 0]
            pts_x = pts_reshape[:, :, :, 1]
        else:
            pts_x = pts_reshape[:, :, :, 0]
            pts_y = pts_reshape[:, :, :, 1]

        if self.transform_method == "minmax":
            # Find min and max coordinates
            xmin = ttnn.min(pts_x, dim=2, keepdim=True)
            xmax = ttnn.max(pts_x, dim=2, keepdim=True)
            ymin = ttnn.min(pts_y, dim=2, keepdim=True)
            ymax = ttnn.max(pts_y, dim=2, keepdim=True)

            # Concatenate to form bbox in xyxy format
            bbox_xyxy = ttnn.concat([xmin, ymin, xmax, ymax], dim=-1)

            # Convert xyxy to cxcywh using utility function
            bbox = ttnn_bbox_xyxy_to_cxcywh(bbox_xyxy)
        else:
            raise NotImplementedError(f"transform_method '{self.transform_method}' not implemented")

        return bbox, pts_reshape

    def get_bboxes(
        self,
        preds_dicts: Dict[str, ttnn.Tensor],
        img_metas: List[Dict],
        rescale: bool = False,
    ) -> List[List]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts: Prediction results dictionary from forward().
            img_metas: Image meta information.
            rescale: Whether to rescale predictions.

        Returns:
            List of [bboxes, scores, labels, pts] for each sample in the batch.
        """
        # Convert TTNN tensors to torch for post-processing
        torch_preds = {}
        for key, value in preds_dicts.items():
            if value is not None:
                torch_preds[key] = ttnn.to_torch(value)
            else:
                torch_preds[key] = None

        # Use bbox_coder if available
        if self.bbox_coder is not None:
            preds_dicts_decoded = self.bbox_coder.decode(torch_preds)

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

        # Manual decoding (fallback)
        all_cls_scores = torch_preds["all_cls_scores"]
        all_bbox_preds = torch_preds["all_bbox_preds"]
        all_pts_preds = torch_preds["all_pts_preds"]

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
