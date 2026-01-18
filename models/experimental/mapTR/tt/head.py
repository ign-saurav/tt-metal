# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN MapTR Head for map element detection (inference-only).

This module provides a TTNN implementation of the MapTR detection head,
following patterns from VADv2's tt_head.py for optimal TTNN compatibility.

Updated to use VADv2's weight loading approach for better PCC.
"""

import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple, Any


def inverse_sigmoid(x: ttnn.Tensor, eps: float = 1e-5) -> ttnn.Tensor:
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


# Keep the old name for backwards compatibility
ttnn_inverse_sigmoid = inverse_sigmoid


def bbox_xyxy_to_cxcywh(bbox: ttnn.Tensor) -> ttnn.Tensor:
    """Convert bounding boxes from xyxy to cxcywh format using TTNN operations.

    This follows VADv2's implementation for consistency and better PCC.

    Args:
        bbox: Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
              Shape: (..., 4)

    Returns:
        Bounding boxes in cxcywh format (cx, cy, w, h).
        Shape: (..., 4)
    """
    # Split bbox into components using slicing (following VADv2's approach)
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
    result = ttnn.concat([cx, cy, w, h], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    return result


# Keep the old name for backwards compatibility
ttnn_bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh


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


class TtLearnedPositionalEncoding:
    """TTNN Learned Positional Encoding following VADv2's implementation."""

    def __init__(
        self,
        params: Any,
        device: ttnn.Device,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
    ):
        super(TtLearnedPositionalEncoding, self).__init__()
        self.row_embed = ttnn.embedding
        self.col_embed = ttnn.embedding
        self.params = params
        self.device = device
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def __call__(self, mask: ttnn.Tensor) -> ttnn.Tensor:
        _, h, w = mask.shape
        x = ttnn.arange(w, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.arange(h, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_embed = self.col_embed(
            x,
            weight=self.params.col_embed.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        y_embed = self.row_embed(y, weight=self.params.row_embed.weight, layout=ttnn.TILE_LAYOUT)
        x_embed = ttnn.unsqueeze(x_embed, 0)
        x_embed = ttnn.repeat(x_embed, (h, 1, 1))
        y_embed = ttnn.unsqueeze(y_embed, 1)
        y_embed = ttnn.repeat(y_embed, (1, w, 1))

        out = ttnn.concat((x_embed, y_embed), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(y_embed)
        ttnn.deallocate(x_embed)
        out = ttnn.permute(out, (2, 0, 1))
        out = ttnn.unsqueeze(out, 0)
        out = ttnn.repeat(out, (mask.shape[0], 1, 1, 1))
        pos = out
        return pos


class TtMapTRHead:
    """TTNN MapTR Head for map element detection (inference-only).

    This is a TTNN implementation derived from the original MapTRHead.
    Updated to use VADv2's weight loading approach for better PCC.

    Now supports two modes:
    1. Legacy mode: Uses flat dictionary params (backwards compatible)
    2. VADv2 mode: Uses preprocessed hierarchical params from VADv2-style preprocessor

    Args:
        params: Dictionary or preprocessed parameters containing pretrained weights.
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
        use_vadv2_params: Whether to use VADv2-style preprocessed params. Default: False.
    """

    def __init__(
        self,
        params: Any,
        device: ttnn.Device,
        transformer: nn.Module = None,
        positional_encoding: nn.Module = None,
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
        use_vadv2_params: bool = False,
        torch_reg_branches: nn.Module = None,
        with_box_refine: bool = True,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.params = params
        self.device = device
        self.transformer = transformer
        self.positional_encoding = positional_encoding
        self.bbox_coder = bbox_coder
        self.torch_reg_branches = torch_reg_branches
        self.with_box_refine = with_box_refine

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
        self.use_vadv2_params = use_vadv2_params

        if use_vadv2_params:
            self._init_weights_vadv2()
        else:
            self._init_weights_legacy()

    def _init_weights_vadv2(self):
        """Initialize TTNN weights from VADv2-style preprocessed params.

        This uses the hierarchical parameter structure from preprocess_model_parameters
        which provides better PCC due to proper weight preprocessing.
        """
        # Store reference to head params for branch access
        # Handle case where params is already head params or has .head attribute
        try:
            # Check if params has a "head" key/attribute
            if hasattr(self.params, "head"):
                self.head_params = self.params.head
            else:
                # params is already the head params
                self.head_params = self.params
        except (KeyError, AttributeError):
            # params is already the head params
            self.head_params = self.params

        # BEV embedding (if using BEVFormer encoder)
        if self.bev_encoder_type == "BEVFormerEncoder":
            if hasattr(self.head_params, "bev_embedding"):
                self.bev_embedding = self.head_params.bev_embedding.weight
            else:
                self.bev_embedding = None
        else:
            self.bev_embedding = None

        # Query embeddings
        if self.query_embed_type == "all_pts":
            if hasattr(self.head_params, "query_embedding"):
                self.query_embedding = self.head_params.query_embedding.weight
            else:
                self.query_embedding = None
        elif self.query_embed_type == "instance_pts":
            if hasattr(self.head_params, "instance_embedding"):
                self.instance_embedding = self.head_params.instance_embedding.weight
            else:
                self.instance_embedding = None
            if hasattr(self.head_params, "pts_embedding"):
                self.pts_embedding = self.head_params.pts_embedding.weight
            else:
                self.pts_embedding = None

    def _init_weights_legacy(self):
        """Initialize TTNN weights from legacy flat dictionary params."""
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
            if "bev_embedding.weight" in self.params:
                self.bev_embedding = self._to_ttnn(self.params["bev_embedding.weight"])
            else:
                self.bev_embedding = None
        else:
            self.bev_embedding = None

        if self.query_embed_type == "all_pts":
            if "query_embedding.weight" in self.params:
                self.query_embedding = self._to_ttnn(self.params["query_embedding.weight"])
        elif self.query_embed_type == "instance_pts":
            if "instance_embedding.weight" in self.params:
                self.instance_embedding = self._to_ttnn(self.params["instance_embedding.weight"])
            if "pts_embedding.weight" in self.params:
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
        return ttnn.linear(input_tensor, weight, bias=bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _layer_norm(
        self, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float = 1e-5
    ) -> ttnn.Tensor:
        """TTNN layer norm implementation."""
        return ttnn.layer_norm(input_tensor, epsilon=eps, weight=weight, bias=bias)

    def _relu(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """TTNN ReLU activation."""
        return ttnn.relu(input_tensor)

    def _cls_branch_vadv2(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN classification branch using VADv2-style params.

        Structure: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]
        """
        cls_params = self.head_params.branches.cls_branches[str(layer_idx)]
        cls_tmp = input_tensor

        # Apply layers following VADv2's pattern
        for i in range(0, 5, 2):  # 0, 2, 4 for Linear layers
            cls_tmp = ttnn.linear(
                cls_tmp,
                cls_params[str(i)].weight,
                bias=cls_params[str(i)].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            norm_key = f"{i+1}_norm"
            if norm_key in cls_params:
                cls_tmp = ttnn.layer_norm(cls_tmp, weight=cls_params[norm_key].weight, bias=cls_params[norm_key].bias)
            if i < 4:
                cls_tmp = ttnn.relu(cls_tmp)

        return cls_tmp

    def _reg_branch_vadv2(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch using VADv2-style params.

        Structure: [Linear + ReLU] * num_reg_fcs + [Linear]
        """
        reg_params = self.head_params.branches.reg_branches[str(layer_idx)]
        reg_tmp = input_tensor

        # Apply layers following VADv2's pattern
        for i in range(3):  # 0, 1, 2 for Linear layers
            reg_tmp = ttnn.linear(
                reg_tmp,
                reg_params[str(i)].weight,
                bias=reg_params[str(i)].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if i < 2:
                reg_tmp = ttnn.relu(reg_tmp)

        return reg_tmp

    def _cls_branch_legacy(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN classification branch using legacy params.

        Structure: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]
        """
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

    def _reg_branch_legacy(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch using legacy params.

        Structure: [Linear + ReLU] * num_reg_fcs + [Linear]
        """
        hidden = input_tensor
        for fc_idx in range(self.num_reg_fcs):
            hidden = self._linear(
                hidden, self.reg_branches_weights[layer_idx][fc_idx], self.reg_branches_biases[layer_idx][fc_idx]
            )
            hidden = self._relu(hidden)
        output = self._linear(hidden, self.reg_final_weights[layer_idx], self.reg_final_biases[layer_idx])
        return output

    def _cls_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN classification branch: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]"""
        if self.use_vadv2_params:
            return self._cls_branch_vadv2(input_tensor, layer_idx)
        else:
            return self._cls_branch_legacy(input_tensor, layer_idx)

    def _reg_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch: [Linear + ReLU] * num_reg_fcs + [Linear]"""
        if self.use_vadv2_params:
            return self._reg_branch_vadv2(input_tensor, layer_idx)
        else:
            return self._reg_branch_legacy(input_tensor, layer_idx)

    def __call__(
        self,
        mlvl_feats: List[ttnn.Tensor] = None,
        lidar_feat: Optional[ttnn.Tensor] = None,
        img_metas: List[Dict] = None,
        prev_bev: Optional[ttnn.Tensor] = None,
        only_bev: bool = False,
        hs: Optional[ttnn.Tensor] = None,
        init_reference: Optional[ttnn.Tensor] = None,
        inter_references: Optional[List[ttnn.Tensor]] = None,
        bev_embed: Optional[ttnn.Tensor] = None,
    ) -> Dict[str, ttnn.Tensor]:
        """TTNN forward function.

        Supports two modes:
        1. Full forward: Process mlvl_feats through transformer (requires transformer)
        2. Head-only: Process precomputed hs, init_reference, inter_references

        Args:
            mlvl_feats: Multi-level features from backbone/FPN.
            lidar_feat: LiDAR features (optional).
            img_metas: Image metadata.
            prev_bev: Previous BEV features.
            only_bev: Return only BEV features.
            hs: Precomputed decoder hidden states (for head-only mode).
            init_reference: Precomputed initial reference points.
            inter_references: Precomputed intermediate reference points.
            bev_embed: Precomputed BEV embedding.

        Returns:
            Dictionary of output predictions.
        """
        # Check if we have precomputed decoder outputs (head-only mode)
        if hs is not None and init_reference is not None:
            return self._forward_head_only(hs, init_reference, inter_references, bev_embed)

        # Full forward mode - requires transformer
        if self.transformer is None:
            raise ValueError(
                "Transformer is required for full forward mode. "
                "Either provide transformer or use precomputed hs/init_reference."
            )

        return self._forward_full(mlvl_feats, lidar_feat, img_metas, prev_bev, only_bev)

    def forward(self, *args, **kwargs) -> Dict[str, ttnn.Tensor]:
        """Alias for __call__ for nn.Module compatibility."""
        return self.__call__(*args, **kwargs)

    def _forward_head_only(
        self,
        hs: ttnn.Tensor,
        init_reference: ttnn.Tensor,
        inter_references: Optional[List[ttnn.Tensor]],
        bev_embed: Optional[ttnn.Tensor],
    ) -> Dict[str, ttnn.Tensor]:
        """Forward pass using precomputed decoder outputs.

        This mode is optimized for testing classification/regression branches
        with VADv2-style weight loading.
        """
        bs = hs.shape[1] if len(hs.shape) == 4 else 1

        # Permute hs to match expected shape: (num_layers, bs, num_query, embed_dims)
        if len(hs.shape) == 4:
            hs = ttnn.permute(hs, (0, 2, 1, 3))

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(self.num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # Classification: average over points per vector
            hs_lvl = hs[lvl]
            hs_reshaped = ttnn.reshape(hs_lvl, (bs, self.num_vec, self.num_pts_per_vec, -1))
            hs_mean = ttnn.mean(hs_reshaped, dim=2)
            outputs_class = self._cls_branch(hs_mean, lvl)

            # Regression
            tmp = self._reg_branch(hs_lvl, lvl)

            # Update reference points
            assert reference.shape[-1] == 2
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference[..., 0:2]
            tmp_updated = ttnn.add(tmp_xy, ref_xy)
            tmp_updated = ttnn.sigmoid(tmp_updated)

            # Transform to bbox and pts
            outputs_coord, outputs_pts_coord = self.transform_box(tmp_updated, bs)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

            # Memory cleanup
            ttnn.deallocate(reference)

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

    def _forward_full(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        img_metas: List[Dict],
        prev_bev: Optional[ttnn.Tensor] = None,
        only_bev: bool = False,
    ) -> Dict[str, ttnn.Tensor]:
        """Full forward pass including transformer."""
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
        # Pass reg_branches for iterative box refinement if available
        reg_branches_for_transformer = self.torch_reg_branches if self.with_box_refine else None
        outputs = self.transformer(
            mlvl_feats_torch,
            None,  # lidar_feat
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=reg_branches_for_transformer,
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
            reference = inverse_sigmoid(reference)

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

            # Memory cleanup
            ttnn.deallocate(reference)

        # Memory cleanup
        ttnn.deallocate(init_reference)
        for ref in inter_references:
            ttnn.deallocate(ref)
        ttnn.deallocate(hs)

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
        """Convert points to bounding box using TTNN operations.

        This follows VADv2's map_transform_box implementation for consistency.

        Args:
            pts: Points tensor with shape (bs * num_vec * num_pts_per_vec, 2) or (bs, num_query, 2)
            bs: Batch size
            y_first: Whether y coordinate comes first

        Returns:
            Tuple of (bbox in cxcywh format, pts_reshape)
        """
        pts_reshape = ttnn.reshape(pts, (bs, self.num_vec, self.num_pts_per_vec, 2))

        # Following VADv2's approach: pts_y is dim 0 if y_first else dim 1
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]

        if self.transform_method == "minmax":
            # Note: ttnn.min/max return a tensor directly, not a tuple like PyTorch
            # Do NOT use [0] indexing as that would slice the tensor and lose batch dimension
            xmin = ttnn.min(pts_x, dim=2, keepdim=True)
            xmax = ttnn.max(pts_x, dim=2, keepdim=True)
            ymin = ttnn.min(pts_y, dim=2, keepdim=True)
            ymax = ttnn.max(pts_y, dim=2, keepdim=True)

            bbox = ttnn.concat([xmin, ymin, xmax, ymax], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bbox = bbox_xyxy_to_cxcywh(bbox)
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
