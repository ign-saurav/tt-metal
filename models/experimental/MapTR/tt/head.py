# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from typing import Dict, List, Optional, Tuple, Any
from models.experimental.MapTR.tt.utils import (
    inverse_sigmoid,
    bbox_xyxy_to_cxcywh,
    denormalize_2d_bbox,
    denormalize_2d_pts,
)


class TtLearnedPositionalEncoding:
    """TTNN Learned Positional Encoding"""

    def __init__(
        self,
        params: Any,
        device: ttnn.Device,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
    ):
        """Initialize TtLearnedPositionalEncoding.

        Args:
            params: Parameters containing row_embed and col_embed weights.
            device: TTNN device.
            num_feats: Number of features (embed_dims // 2).
            row_num_embed: Number of row embeddings.
            col_num_embed: Number of column embeddings.
        """
        self.row_embed = ttnn.embedding
        self.col_embed = ttnn.embedding
        self.params = params
        self.device = device
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def __call__(self, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Generate positional encoding from mask.

        Args:
            mask: Input mask tensor of shape (bs, h, w).

        Returns:
            Positional encoding tensor of shape (bs, embed_dims, h, w).
        """
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
        return out


class TtMapTRHead:
    """TTNN MapTR Head for map element detection (inference-only).


    Args:
        params: Parameters containing pretrained weights .
        device: TTNN device for tensor operations.
        transformer: Optional TtMapTRPerceptionTransformer for full forward pass.
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        num_reg_fcs: Number of FC layers in each branch. Default: 2.
        code_size: Size of output code (2 for x, y). Default: 2.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range.
        num_vec: Number of vectors (instances). Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
        query_embed_type: Type of query embedding. Default: 'instance_pts'.
        transform_method: Method to transform points to bbox. Default: 'minmax'.
        bev_encoder_type: BEV encoder type. Default: 'BEVFormerEncoder'.
        with_box_refine: Whether to use iterative box refinement. Default: True.
    """

    def __init__(
        self,
        params: Any,
        device: ttnn.Device,
        transformer: Any = None,
        positional_encoding: Any = None,
        embed_dims: int = 256,
        num_classes: int = 3,
        num_reg_fcs: int = 2,
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
        with_box_refine: bool = True,
        as_two_stage: bool = False,
    ):
        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.params = params
        self.device = device
        self.transformer = transformer
        self.positional_encoding = positional_encoding

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
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # Initialize positional encoding if provided in params
        if positional_encoding is None and hasattr(self.params, "positional_encoding"):
            self.positional_encoding = TtLearnedPositionalEncoding(
                params=self.params.positional_encoding,
                device=device,
                num_feats=embed_dims // 2,
                row_num_embed=bev_h,
                col_num_embed=bev_w,
            )

        # Store references to embeddings from params
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding references from params."""
        # BEV embedding (for BEVFormerEncoder)
        if self.bev_encoder_type == "BEVFormerEncoder":
            if hasattr(self.params, "bev_embedding"):
                self.bev_embedding = self.params.bev_embedding
            else:
                self.bev_embedding = None
        else:
            self.bev_embedding = None

        # Query embeddings
        if self.query_embed_type == "all_pts":
            if hasattr(self.params, "query_embedding"):
                self.query_embedding = self.params.query_embedding
            else:
                self.query_embedding = None
        elif self.query_embed_type == "instance_pts":
            if hasattr(self.params, "instance_embedding"):
                self.instance_embedding = self.params.instance_embedding
            else:
                self.instance_embedding = None
            if hasattr(self.params, "pts_embedding"):
                self.pts_embedding = self.params.pts_embedding
            else:
                self.pts_embedding = None

    def _cls_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN classification branch

        Structure: [Linear + LayerNorm + ReLU] * num_reg_fcs + [Linear]

        Args:
            input_tensor: Input tensor of shape (bs, num_vec, embed_dims).
            layer_idx: Decoder layer index.

        Returns:
            Classification scores of shape (bs, num_vec, num_classes).
        """
        cls_params = self.params.branches.cls_branches[str(layer_idx)]
        cls_tmp = input_tensor

        # Apply layers: Linear(0) + LayerNorm(1) + ReLU + Linear(2) + LayerNorm(3) + ReLU + Linear(4)
        for i in range(0, 5, 2):  # 0, 2, 4 for Linear layers
            cls_tmp = ttnn.linear(
                cls_tmp,
                cls_params[str(i)].weight,
                bias=cls_params[str(i)].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            norm_key = f"{i+1}_norm"
            try:
                norm_layer = cls_params[norm_key]
                cls_tmp = ttnn.layer_norm(cls_tmp, weight=norm_layer.weight, bias=norm_layer.bias)
            except KeyError:
                pass  # No norm layer for this position
            if i < 4:
                cls_tmp = ttnn.relu(cls_tmp)

        return cls_tmp

    def _reg_branch(self, input_tensor: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """TTNN regression branch

        Structure: [Linear + ReLU] * num_reg_fcs + [Linear]

        Args:
            input_tensor: Input tensor of shape (bs, num_query, embed_dims).
            layer_idx: Decoder layer index.

        Returns:
            Regression output of shape (bs, num_query, code_size).
        """
        reg_params = self.params.branches.reg_branches[str(layer_idx)]
        reg_tmp = input_tensor

        # Apply layers: Linear(0) + ReLU + Linear(1) + ReLU + Linear(2)
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

    def transform_box(self, pts: ttnn.Tensor, y_first: bool = False) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Convert points to bounding box using TTNN operations.


        Args:
            pts: Points tensor with shape (bs, num_query, 2) or (bs * num_vec * num_pts_per_vec, 2).
            y_first: Whether y coordinate comes first.

        Returns:
            Tuple of (bbox in cxcywh format, pts_reshape).
        """
        # Get batch size from pts shape
        if len(pts.shape) == 3:
            bs = pts.shape[0]
        else:
            bs = 1

        pts_reshape = ttnn.reshape(pts, (bs, self.num_vec, self.num_pts_per_vec, 2))

        # pts_y is dim 0 if y_first else dim 1
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]

        if self.transform_method == "minmax":
            xmin = ttnn.min(pts_x, dim=2, keepdim=True)
            xmax = ttnn.max(pts_x, dim=2, keepdim=True)
            ymin = ttnn.min(pts_y, dim=2, keepdim=True)
            ymax = ttnn.max(pts_y, dim=2, keepdim=True)

            bbox = ttnn.concat([xmin, ymin, xmax, ymax], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError(f"transform_method '{self.transform_method}' not implemented")

        return bbox, pts_reshape

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
        """Forward function.

        Supports two modes:
        1. Full forward: Process mlvl_feats through transformer (requires transformer)
        2. Head-only: Process precomputed hs, init_reference, inter_references

        Args:
            mlvl_feats: Multi-level features from backbone/FPN.
            lidar_feat: LiDAR features (optional).
            img_metas: Image metadata.
            prev_bev: Previous BEV features.
            only_bev: Return only BEV features.
            hs: Precomputed decoder hidden states.
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

    def _forward_head_only(
        self,
        hs: ttnn.Tensor,
        init_reference: ttnn.Tensor,
        inter_references: Optional[List[ttnn.Tensor]],
        bev_embed: Optional[ttnn.Tensor],
    ) -> Dict[str, ttnn.Tensor]:
        """Forward pass using precomputed decoder outputs.

        Args:
            hs: Hidden states from decoder, shape (num_layers, num_query, bs, embed_dims).
            init_reference: Initial reference points, shape (bs, num_query, 2).
            inter_references: Intermediate reference points.
            bev_embed: BEV embedding tensor.

        Returns:
            Dictionary of output predictions.
        """
        # Permute hs to match expected shape: (num_layers, bs, num_query, embed_dims)
        if len(hs.shape) == 4:
            hs = ttnn.permute(hs, (0, 2, 1, 3))
            bs = hs.shape[1]
        else:
            bs = 1

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
            outputs_coord, outputs_pts_coord = self.transform_box(tmp_updated)

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
        """Full forward pass including transformer.

        Args:
            mlvl_feats: Multi-level features from backbone/FPN.
            lidar_feat: LiDAR features (optional).
            img_metas: Image metadata.
            prev_bev: Previous BEV features.
            only_bev: Return only BEV features.

        Returns:
            Dictionary of output predictions.
        """
        bs = mlvl_feats[0].shape[0]

        # Prepare query embeddings
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight
        elif self.query_embed_type == "instance_pts":
            pts_embeds = ttnn.unsqueeze(self.pts_embedding.weight, 0)
            instance_embeds = ttnn.unsqueeze(self.instance_embedding.weight, 1)
            object_query_embeds = ttnn.add(pts_embeds, instance_embeds)
            object_query_embeds = ttnn.reshape(
                object_query_embeds,
                (object_query_embeds.shape[0] * object_query_embeds.shape[1], object_query_embeds.shape[2]),
            )
        else:
            object_query_embeds = None

        # Prepare BEV queries and positional encoding
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight
            bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), device=self.device, dtype=ttnn.bfloat16)
            if self.positional_encoding is not None:
                bev_pos = self.positional_encoding(bev_mask)
                bev_pos = ttnn.to_layout(bev_pos, layout=ttnn.ROW_MAJOR_LAYOUT)
            else:
                bev_pos = None
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        # Full transformer forward
        outputs = self.transformer(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=True if self.with_box_refine else None,
            cls_branches=True if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        # Cleanup
        if bev_queries is not None:
            ttnn.deallocate(bev_queries)
        if bev_mask is not None:
            ttnn.deallocate(bev_mask)
        if bev_pos is not None:
            ttnn.deallocate(bev_pos)

        bev_embed, hs, init_reference, inter_references = outputs

        # Permute hs: (num_layers, num_query, bs, embed_dims) -> (num_layers, bs, num_query, embed_dims)
        hs = ttnn.permute(hs, (0, 2, 1, 3))
        bs = hs.shape[1]
        num_layers = hs.shape[0]
        print(f"[TT Head] hs after permute: shape={hs.shape}, num_layers={num_layers}")

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(num_layers):
            if lvl == 0:
                reference = init_reference
            else:
                # inter_references is stacked tensor (num_layers, bs, num_query, 2)
                reference = inter_references[lvl - 1]
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
            outputs_coord, outputs_pts_coord = self.transform_box(tmp_updated)

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

    def get_bboxes(
        self, preds_dicts: Dict[str, ttnn.Tensor], img_metas: List[Dict], rescale: bool = False
    ) -> List[List]:
        """Generate bboxes from predictions.

        Args:
            preds_dicts: Dictionary of predictions from forward pass.
            img_metas: Image metadata.
            rescale: Whether to rescale boxes.

        Returns:
            List of [bboxes, scores, labels, pts] for each sample.
        """
        # Convert to torch for decoding
        torch_preds = {}
        for key, value in preds_dicts.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    torch_preds[key] = value.float()
                else:
                    torch_preds[key] = ttnn.to_torch(value).float()
            else:
                torch_preds[key] = None

        # Manual decoding
        all_cls_scores = torch_preds["all_cls_scores"]
        all_bbox_preds = torch_preds["all_bbox_preds"]
        all_pts_preds = torch_preds["all_pts_preds"]

        # Use last decoder layer output
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        pts_preds = all_pts_preds[-1]

        num_samples = cls_scores.shape[0]
        ret_list = []

        for i in range(num_samples):
            # Match PyTorch MapTRNMSFreeCoder.decode behavior:
            # 1. Sigmoid on cls_scores
            # 2. Flatten and take top-k scores
            # 3. Compute labels from flattened indices
            # 4. Reorder bbox/pts by top-k query indices

            cls_score = cls_scores[i].sigmoid()  # (num_query, num_classes)
            num_query = cls_score.shape[0]
            num_classes = self.num_classes

            # Flatten and get top-k (self.num_vec = max_num = 50)
            scores_flat, indexs = cls_score.view(-1).topk(self.num_vec)
            labels = indexs % num_classes
            bbox_index = indexs // num_classes
            bbox_index = torch.clamp(bbox_index, 0, num_query - 1)

            # Reorder predictions by top-k query indices
            bbox_pred_reordered = bbox_preds[i][bbox_index]
            pts_pred_reordered = pts_preds[i][bbox_index]

            # Denormalize
            bboxes = denormalize_2d_bbox(bbox_pred_reordered, self.pc_range)
            pts = denormalize_2d_pts(pts_pred_reordered, self.pc_range)

            ret_list.append([bboxes, scores_flat, labels, pts])

        return ret_list
