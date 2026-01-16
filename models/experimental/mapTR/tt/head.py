# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN MapTR Head for map element detection (inference-only).

This module provides a TTNN implementation of the MapTR detection head,
following patterns from VADv2's tt_head.py for optimal TTNN compatibility.
"""

import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple


def ttnn_inverse_sigmoid(x: ttnn.Tensor, eps: float = 1e-5) -> ttnn.Tensor:
    """Compute inverse sigmoid using TTNN operations.

    inverse_sigmoid(x) = log(x / (1 - x))

    This implementation follows VADv2's approach using ttnn.ones() for
    proper tensor-tensor operations.

    Args:
        x: Input tensor with values in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        Inverse sigmoid of input tensor.
    """
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)

    # Create ones tensor of same shape for (1 - x) computation
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )

    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)

    return ttnn.log(ttnn.div(x1, x2))


def ttnn_bbox_xyxy_to_cxcywh(bbox: ttnn.Tensor) -> ttnn.Tensor:
    """Convert bounding boxes from xyxy to cxcywh format using TTNN operations.

    Args:
        bbox: Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
              Shape: (..., 4)

    Returns:
        Bounding boxes in cxcywh format (cx, cy, w, h).
        Shape: (..., 4)
    """
    # Split bbox into components using slicing
    x0 = bbox[..., 0:1]
    y0 = bbox[..., 1:2]
    x1 = bbox[..., 2:3]
    y1 = bbox[..., 3:4]

    # Compute center and size using proper tensor operations
    cx = ttnn.div(ttnn.add(x0, x1), 2.0)
    cy = ttnn.div(ttnn.add(y0, y1), 2.0)
    w = ttnn.subtract(x1, x0)
    h = ttnn.subtract(y1, y0)

    # Concatenate to form output
    result = ttnn.concat([cx, cy, w, h], dim=-1)
    return result


def denormalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize bounding boxes from [0, 1] to real-world coordinates."""
    new_bboxes = bboxes.clone()
    new_bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_bboxes


def denormalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize points from [0, 1] to real-world coordinates."""
    new_pts = pts.clone()
    new_pts[..., 0] = pts[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1] = pts[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


class TtMapTRHead(nn.Module):
    """TTNN MapTR Head for map element detection (inference-only).

    This is a TTNN implementation derived from the original MapTRHead.
    It uses PyTorch transformer internally and TTNN for efficient
    classification and regression branches.

    Args:
        params: Dictionary containing pretrained weights.
        device: TTNN device for tensor operations.
        transformer: The transformer module (PyTorch MapTRPerceptionTransformer).
        positional_encoding: Positional encoding module (PyTorch).
        bbox_coder: Optional bbox coder for decoding predictions.
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        num_reg_fcs: Number of FC layers in each branch. Default: 2.
        num_cls_fcs: Number of FC layers in classification branch. Default: 2.
        code_size: Size of output code (2 for x, y). Default: 2.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range.
        num_vec: Number of vectors (instances). Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
        query_embed_type: Type of query embedding. Default: 'instance_pts'.
        transform_method: Method to transform points to bbox. Default: 'minmax'.
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
        num_decoder_layers: int = 6,
        query_embed_type: str = "instance_pts",
        transform_method: str = "minmax",
        bev_encoder_type: str = "BEVFormerEncoder",
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.params = params
        self.device = device
        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.bbox_coder = bbox_coder

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
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
        self.num_decoder_layers = num_decoder_layers
        self.bev_encoder_type = bev_encoder_type

        self._init_weights()

    def _init_weights(self):
        """Initialize TTNN weights from params dictionary."""
        num_pred = self.num_decoder_layers

        # Classification branch weights (with LayerNorm)
        self.cls_branches_weights = []
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
                linear_idx = fc_idx * 3
                norm_idx = fc_idx * 3 + 1

                w_key = f"cls_branches.{layer_idx}.{linear_idx}.weight"
                b_key = f"cls_branches.{layer_idx}.{linear_idx}.bias"
                nw_key = f"cls_branches.{layer_idx}.{norm_idx}.weight"
                nb_key = f"cls_branches.{layer_idx}.{norm_idx}.bias"

                layer_weights.append(self._to_ttnn_linear_weight(self.params[w_key]))
                layer_biases.append(self._to_ttnn_linear_bias(self.params[b_key]))
                norm_weights.append(self._to_ttnn(self.params[nw_key]))
                norm_biases.append(self._to_ttnn(self.params[nb_key]))

            self.cls_branches_weights.append(layer_weights)
            self.cls_branches_biases.append(layer_biases)
            self.cls_norm_weights.append(norm_weights)
            self.cls_norm_biases.append(norm_biases)

            final_idx = self.num_reg_fcs * 3
            self.cls_final_weights.append(
                self._to_ttnn_linear_weight(self.params[f"cls_branches.{layer_idx}.{final_idx}.weight"])
            )
            self.cls_final_biases.append(
                self._to_ttnn_linear_bias(self.params[f"cls_branches.{layer_idx}.{final_idx}.bias"])
            )

        # Regression branch weights (without LayerNorm)
        self.reg_branches_weights = []
        self.reg_branches_biases = []
        self.reg_final_weights = []
        self.reg_final_biases = []

        for layer_idx in range(num_pred):
            layer_weights = []
            layer_biases = []

            for fc_idx in range(self.num_reg_fcs):
                linear_idx = fc_idx * 2

                w_key = f"reg_branches.{layer_idx}.{linear_idx}.weight"
                b_key = f"reg_branches.{layer_idx}.{linear_idx}.bias"

                layer_weights.append(self._to_ttnn_linear_weight(self.params[w_key]))
                layer_biases.append(self._to_ttnn_linear_bias(self.params[b_key]))

            self.reg_branches_weights.append(layer_weights)
            self.reg_branches_biases.append(layer_biases)

            final_idx = self.num_reg_fcs * 2
            self.reg_final_weights.append(
                self._to_ttnn_linear_weight(self.params[f"reg_branches.{layer_idx}.{final_idx}.weight"])
            )
            self.reg_final_biases.append(
                self._to_ttnn_linear_bias(self.params[f"reg_branches.{layer_idx}.{final_idx}.bias"])
            )

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
        """Convert PyTorch tensor to TTNN tensor with TILE layout."""
        return ttnn.from_torch(tensor, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _to_ttnn_linear_weight(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Convert PyTorch linear weight to TTNN with transposition."""
        weight_t = tensor.T.contiguous()
        return ttnn.from_torch(weight_t, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _to_ttnn_linear_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Convert PyTorch linear bias to TTNN with proper reshaping."""
        bias_reshaped = tensor.reshape(1, -1)
        return ttnn.from_torch(bias_reshaped, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

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
        """TTNN classification branch: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]"""
        hidden = input_tensor
        for fc_idx in range(self.num_reg_fcs):
            hidden = self._linear(
                hidden, self.cls_branches_weights[layer_idx][fc_idx], self.cls_branches_biases[layer_idx][fc_idx]
            )
            hidden = self._layer_norm(
                hidden, self.cls_norm_weights[layer_idx][fc_idx], self.cls_norm_biases[layer_idx][fc_idx]
            )
            hidden = self._relu(hidden)
        output = self._linear(hidden, self.cls_final_weights[layer_idx], self.cls_final_biases[layer_idx])
        return output

    def _reg_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch: [Linear + ReLU] * num_reg_fcs + [Linear]"""
        hidden = input_tensor
        for fc_idx in range(self.num_reg_fcs):
            hidden = self._linear(
                hidden, self.reg_branches_weights[layer_idx][fc_idx], self.reg_branches_biases[layer_idx][fc_idx]
            )
            hidden = self._relu(hidden)
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
        """TTNN forward function."""
        # Convert TTNN features to torch for transformer
        num_cams = 6
        mlvl_feats_torch = []
        for feat in mlvl_feats:
            if isinstance(feat, tuple):
                feat_tensor = feat[0]
            else:
                feat_tensor = feat
            feat_torch = ttnn.to_torch(feat_tensor).float()
            if len(feat_torch.shape) == 4:
                BN, H, W, C = feat_torch.shape
                feat_torch = feat_torch.permute(0, 3, 1, 2)
                B = BN // num_cams
                if B * num_cams == BN and B > 0:
                    feat_torch = feat_torch.reshape(B, num_cams, C, H, W)
                else:
                    feat_torch = feat_torch.reshape(1, BN, C, H, W)
            mlvl_feats_torch.append(feat_torch)

        bs = mlvl_feats_torch[0].shape[0]

        # Prepare query embeddings
        if self.query_embed_type == "all_pts":
            object_query_embeds = ttnn.to_torch(self.query_embedding).float()
        elif self.query_embed_type == "instance_pts":
            pts_embeds = ttnn.to_torch(self.pts_embedding).float().unsqueeze(0)
            instance_embeds = ttnn.to_torch(self.instance_embedding).float().unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)
        else:
            object_query_embeds = None

        # Prepare BEV queries and positional encoding
        if self.bev_embedding is not None:
            bev_queries = ttnn.to_torch(self.bev_embedding).float()
            bev_mask_torch = torch.zeros((bs, self.bev_h, self.bev_w), dtype=torch.float32)
            bev_pos = self.positional_encoding(bev_mask_torch)
        else:
            bev_queries = None
            bev_pos = None

        # Transformer forward (PyTorch)
        outputs = self.transformer(
            mlvl_feats_torch,
            None,  # lidar_feat
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=None,
            cls_branches=None,
            img_metas=img_metas,
            prev_bev=None,
        )

        bev_embed_torch, hs_torch, init_reference_torch, inter_references_torch = outputs

        # Convert outputs back to TTNN
        bev_embed = ttnn.from_torch(bev_embed_torch, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)

        hs_ttnn = []
        for hs_layer in hs_torch:
            hs_layer_ttnn = ttnn.from_torch(hs_layer, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
            hs_ttnn.append(hs_layer_ttnn)
        hs = ttnn.stack(hs_ttnn, dim=0)
        hs = ttnn.permute(hs, (0, 2, 1, 3))

        init_reference = ttnn.from_torch(
            init_reference_torch, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
        )
        inter_references = []
        for ref in inter_references_torch:
            ref_ttnn = ttnn.from_torch(ref, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
            inter_references.append(ref_ttnn)

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(self.num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = ttnn_inverse_sigmoid(reference)

            # Classification
            hs_reshaped = ttnn.reshape(hs[lvl], (bs, self.num_vec, self.num_pts_per_vec, -1))
            hs_mean = ttnn.mean(hs_reshaped, dim=2)
            outputs_class = self._cls_branch(hs_mean, lvl)

            # Regression
            tmp = self._reg_branch(hs[lvl], lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference[..., 0:2]
            tmp_updated = ttnn.add(tmp_xy, ref_xy)
            tmp_updated = ttnn.sigmoid(tmp_updated)

            # Transform to bbox and pts
            outputs_coord, outputs_pts_coord = self.transform_box(tmp_updated, bs)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

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

    def transform_box(self, pts: ttnn.Tensor, bs: int, y_first: bool = False) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Convert points to bounding box using TTNN operations."""
        pts_reshape = ttnn.reshape(pts, (bs, self.num_vec, self.num_pts_per_vec, 2))

        if y_first:
            pts_y = pts_reshape[..., 0]
            pts_x = pts_reshape[..., 1]
        else:
            pts_x = pts_reshape[..., 0]
            pts_y = pts_reshape[..., 1]

        if self.transform_method == "minmax":
            xmin = ttnn.min(pts_x, dim=2, keepdim=True)[0]
            xmax = ttnn.max(pts_x, dim=2, keepdim=True)[0]
            ymin = ttnn.min(pts_y, dim=2, keepdim=True)[0]
            ymax = ttnn.max(pts_y, dim=2, keepdim=True)[0]

            bbox_xyxy = ttnn.concat([xmin, ymin, xmax, ymax], dim=-1)
            bbox = ttnn_bbox_xyxy_to_cxcywh(bbox_xyxy)

            # Ensure batch dimension is preserved
            if len(bbox.shape) == 2:
                bbox = ttnn.reshape(bbox, (bs, self.num_vec, 4))
        else:
            raise NotImplementedError(f"transform_method '{self.transform_method}' not implemented")

        return bbox, pts_reshape

    def get_bboxes(
        self, preds_dicts: Dict[str, ttnn.Tensor], img_metas: List[Dict], rescale: bool = False
    ) -> List[List]:
        """Generate bboxes from predictions."""
        torch_preds = {}
        for key, value in preds_dicts.items():
            if value is not None:
                torch_preds[key] = ttnn.to_torch(value)
            else:
                torch_preds[key] = None

        if self.bbox_coder is not None:
            preds_dicts_decoded = self.bbox_coder.decode(torch_preds)
            ret_list = []
            for i in range(len(preds_dicts_decoded)):
                preds = preds_dicts_decoded[i]
                ret_list.append([preds["bboxes"], preds["scores"], preds["labels"], preds["pts"]])
            return ret_list

        # Manual decoding
        all_cls_scores = torch_preds["all_cls_scores"]
        all_bbox_preds = torch_preds["all_bbox_preds"]
        all_pts_preds = torch_preds["all_pts_preds"]

        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        pts_preds = all_pts_preds[-1]

        num_samples = cls_scores.shape[0]
        ret_list = []

        for i in range(num_samples):
            cls_score = cls_scores[i].sigmoid()
            scores, labels = cls_score.max(dim=-1)
            bboxes = denormalize_2d_bbox(bbox_preds[i], self.pc_range)
            pts = denormalize_2d_pts(pts_preds[i], self.pc_range)
            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
