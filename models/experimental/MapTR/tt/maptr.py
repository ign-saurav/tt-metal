# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Complete TTNN MapTR model for map element detection (inference-only).

This module provides a complete end-to-end TTNN implementation of the MapTR detector,
integrating the TTNN backbone (ResNet50), FPN neck, transformer, and MapTR head.

This is a pure TTNN implementation without any PyTorch dependencies during inference.
"""

import copy
import ttnn
from typing import Dict, List, Optional, Any

from models.experimental.MapTR.tt.backbone import TtResNet50
from models.experimental.MapTR.tt.fpn import TtFPN
from models.experimental.MapTR.tt.head import TtMapTRHead
from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.decoder import TtMapTRDecoder


class TtMapTR:
    """Complete TTNN MapTR detector for map element detection (inference-only).

    This is a full TTNN implementation:
    - TTNN backbone (TtResNet50) for image feature extraction
    - TTNN FPN (TtFPN) for feature pyramid
    - TTNN transformer (TtMapTRPerceptionTransformer) for BEV and query processing
    - TTNN head (TtMapTRHead) for classification and regression

    Args:
        device: TTNN device for tensor operations.
        params: Complete preprocessed parameters object containing:
            - backbone: ResNet50 backbone parameters
            - backbone_conv_args: Convolution arguments for backbone
            - neck: FPN parameters
            - neck_lateral_config: FPN lateral conv config
            - neck_fpn_config: FPN fpn conv config
            - transformer: Transformer parameters
            - head: Head parameters
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range.
        num_vec: Number of vectors. Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
        num_encoder_layers: Number of encoder layers. Default: 6.
        num_cams: Number of cameras. Default: 6.
        rotate_prev_bev: Whether to rotate previous BEV. Default: True.
        use_shift: Whether to use shift. Default: True.
        use_can_bus: Whether to use CAN bus. Default: True.
        can_bus_norm: Whether to normalize CAN bus. Default: True.
        use_cams_embeds: Whether to use camera embeddings. Default: True.
        with_box_refine: Whether to use box refinement. Default: True.
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: Any,
        embed_dims: int = 256,
        num_classes: int = 3,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        num_cams: int = 6,
        num_feature_levels: int = 1,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        with_box_refine: bool = True,
        feedforward_channels: int = 512,
        num_heads: int = 8,
        num_points_in_pillar: int = 4,
        num_points: int = 8,
    ):
        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_query = num_vec * num_pts_per_vec
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_cams = num_cams
        self.num_feature_levels = num_feature_levels
        self.with_box_refine = with_box_refine

        # Initialize backbone
        self.img_backbone = TtResNet50(
            conv_args=params.backbone_conv_args,
            conv_pth=params.backbone,
            device=device,
        )

        # Initialize FPN neck
        self.img_neck = TtFPN(
            lateral_conv_config=params.neck_lateral_config,
            fpn_conv_config=params.neck_fpn_config,
            device=device,
        )

        # Initialize encoder
        encoder = TtBEVFormerEncoder(
            params=params.encoder,
            device=device,
            num_layers=num_encoder_layers,
            pc_range=pc_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=False,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            feedforward_channels=feedforward_channels,
        )

        # Initialize decoder
        decoder = TtMapTRDecoder(
            num_layers=num_decoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            params=params.decoder,
            params_branches=params.head,
            device=device,
            feedforward_channels=feedforward_channels,
        )

        # Initialize transformer
        self.transformer = TtMapTRPerceptionTransformer(
            params=params.transformer,
            device=device,
            encoder=encoder,
            decoder=decoder,
            embed_dims=embed_dims,
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            rotate_prev_bev=rotate_prev_bev,
            use_shift=use_shift,
            use_can_bus=use_can_bus,
            can_bus_norm=can_bus_norm,
            use_cams_embeds=use_cams_embeds,
            rotate_center=[bev_h // 2, bev_w // 2],
        )

        # Initialize head
        self.pts_bbox_head = TtMapTRHead(
            params=params.head,
            device=device,
            transformer=self.transformer,
            positional_encoding=None,  # Will use internal encoding
            embed_dims=embed_dims,
            num_classes=num_classes,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
            num_vec=num_vec,
            num_pts_per_vec=num_pts_per_vec,
            num_decoder_layers=num_decoder_layers,
            with_box_refine=with_box_refine,
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
        img: ttnn.Tensor,
        img_metas: Optional[List[Dict]] = None,
        batch_size: int = 6,
    ) -> Optional[List[ttnn.Tensor]]:
        """Extract features from images using TTNN backbone and FPN.

        Args:
            img: Input images as TTNN tensor in flattened NHWC format.
                 Shape: (1, 1, N*H*W, C) where N is batch*num_cams
            img_metas: Image meta information.
            batch_size: Batch size (typically num_cams for single frame).

        Returns:
            List of TTNN feature tensors.
        """
        if img is None:
            return None

        # Backbone forward (TTNN)
        img_feats = self.img_backbone(img, batch_size=batch_size)

        # FPN forward (TTNN)
        img_feats = self.img_neck(img_feats)

        return img_feats

    def __call__(
        self,
        img: ttnn.Tensor,
        img_metas: List[Dict],
        prev_bev: Optional[ttnn.Tensor] = None,
        batch_size: int = 6,
    ) -> Dict[str, ttnn.Tensor]:
        """Forward function for inference.

        Args:
            img: Input images as TTNN tensor.
            img_metas: Image meta information.
            prev_bev: Previous BEV features.
            batch_size: Batch size.

        Returns:
            Dictionary of detection results.
        """
        # Check scene change for temporal fusion
        scene_token = img_metas[0].get("scene_token", None)
        if scene_token != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = scene_token

        # Use provided prev_bev or from cache
        if prev_bev is None:
            prev_bev = self.prev_frame_info["prev_bev"]

        # Handle can_bus for temporal fusion
        can_bus = img_metas[0].get("can_bus", None)
        if can_bus is not None:
            tmp_pos = copy.deepcopy(can_bus[:3])
            tmp_angle = copy.deepcopy(can_bus[-1])
            if self.prev_frame_info["prev_bev"] is not None:
                img_metas[0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
                img_metas[0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
            else:
                img_metas[0]["can_bus"][-1] = 0
                img_metas[0]["can_bus"][:3] = 0
        else:
            tmp_pos = 0
            tmp_angle = 0

        # Extract features
        img_feats = self.extract_img_feat(img, img_metas, batch_size=batch_size)

        # Run head forward (includes transformer)
        outs = self.pts_bbox_head(
            mlvl_feats=img_feats,
            lidar_feat=None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        # Update temporal state
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        if "bev_embed" in outs and outs["bev_embed"] is not None:
            self.prev_frame_info["prev_bev"] = outs["bev_embed"]

        return outs

    def forward_head_only(
        self,
        hs: ttnn.Tensor,
        init_reference: ttnn.Tensor,
        inter_references: List[ttnn.Tensor],
        bev_embed: Optional[ttnn.Tensor] = None,
    ) -> Dict[str, ttnn.Tensor]:
        """Forward pass using precomputed decoder outputs (head-only mode).

        Args:
            hs: Hidden states from decoder.
            init_reference: Initial reference points.
            inter_references: Intermediate reference points.
            bev_embed: BEV embedding tensor.

        Returns:
            Dictionary of output predictions.
        """
        return self.pts_bbox_head(
            hs=hs,
            init_reference=init_reference,
            inter_references=inter_references,
            bev_embed=bev_embed,
        )

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
        return self.pts_bbox_head.get_bboxes(preds_dicts, img_metas, rescale=rescale)

    def reset_temporal_state(self):
        """Reset temporal state for new sequence."""
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }


def create_ttnn_maptr_model(
    params: Any,
    device: ttnn.Device,
    config: Optional[Dict] = None,
) -> TtMapTR:
    """Factory function to create a complete TTNN MapTR model.

    Args:
        params: Preprocessed parameters from weight loading.
        device: TTNN device.
        config: Optional configuration dictionary.

    Returns:
        Initialized TtMapTR model.
    """
    if config is None:
        # Default MapTR tiny config
        config = {
            "embed_dims": 256,
            "num_classes": 3,
            "bev_h": 200,
            "bev_w": 100,
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "num_vec": 50,
            "num_pts_per_vec": 20,
            "num_decoder_layers": 6,
            "num_encoder_layers": 6,
            "num_cams": 6,
            "num_feature_levels": 1,
            "feedforward_channels": 512,
            "num_heads": 8,
            "num_points_in_pillar": 4,
            "num_points": 8,
            "with_box_refine": True,
        }

    return TtMapTR(
        device=device,
        params=params,
        embed_dims=config["embed_dims"],
        num_classes=config["num_classes"],
        bev_h=config["bev_h"],
        bev_w=config["bev_w"],
        pc_range=config["pc_range"],
        num_vec=config["num_vec"],
        num_pts_per_vec=config["num_pts_per_vec"],
        num_decoder_layers=config["num_decoder_layers"],
        num_encoder_layers=config["num_encoder_layers"],
        num_cams=config["num_cams"],
        num_feature_levels=config["num_feature_levels"],
        feedforward_channels=config["feedforward_channels"],
        num_heads=config["num_heads"],
        num_points_in_pillar=config["num_points_in_pillar"],
        num_points=config["num_points"],
        with_box_refine=config["with_box_refine"],
    )
