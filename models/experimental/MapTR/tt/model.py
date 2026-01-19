# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Complete TTNN MapTR model for map element detection (inference-only).

This module provides a complete TTNN implementation of the MapTR detector,
using TTNN for all components: backbone, neck, transformer (encoder + decoder), and head.
No PyTorch components are used in the forward pass.
"""

import copy
import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple
from loguru import logger

from models.experimental.MapTR.tt.backbone import TtResNet50
from models.experimental.MapTR.tt.fpn import TtFPN
from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.decoder import TtMapTRDecoder
from models.experimental.MapTR.tt.head import TtMapTRHead, TtLearnedPositionalEncoding
from models.tt_cnn.tt.builder import Conv2dConfiguration, AutoShardedStrategyConfiguration


class TtMapTR(nn.Module):
    """Complete TTNN MapTR detector for map element detection (inference-only).

    This implementation uses TTNN for all components:
    - TTNN backbone (TtResNet50) for image feature extraction
    - TTNN FPN (TtFPN) for feature pyramid
    - TTNN transformer (TtMapTRPerceptionTransformer) with:
      - TTNN encoder (TtBEVFormerEncoder)
      - TTNN decoder (TtMapTRDecoder)
    - TTNN head (TtMapTRHead) for classification and regression

    Args:
        device: TTNN device for tensor operations.
        params: Preprocessed parameters object containing all model weights.
        embed_dims: Embedding dimensions. Default: 256.
        num_classes: Number of classes. Default: 3.
        bev_h: Height of BEV feature. Default: 200.
        bev_w: Width of BEV feature. Default: 100.
        pc_range: Point cloud range.
        num_vec: Number of vectors. Default: 50.
        num_pts_per_vec: Number of points per vector. Default: 20.
        num_decoder_layers: Number of decoder layers. Default: 6.
        num_encoder_layers: Number of encoder layers. Default: 6.
        use_grid_mask: Whether to use grid mask augmentation. Default: False.
        video_test_mode: Whether to use temporal information. Default: False.
    """

    def __init__(
        self,
        device: ttnn.Device,
        params,
        embed_dims: int = 256,
        num_classes: int = 3,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        use_grid_mask: bool = False,
        video_test_mode: bool = False,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.device = device
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.pc_range = pc_range
        self.params = params

        # Initialize TTNN Backbone
        logger.info("Initializing TTNN ResNet50 backbone...")
        self.img_backbone = TtResNet50(params.conv_args.img_backbone, params.img_backbone, device)

        # Initialize TTNN FPN
        logger.info("Initializing TTNN FPN...")
        lateral_config = Conv2dConfiguration.from_model_args(
            conv2d_args=params.conv_args.img_neck.lateral_convs,
            weights=params.img_neck.lateral_convs.conv.weight,
            bias=params.img_neck.lateral_convs.conv.bias,
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        fpn_config = Conv2dConfiguration.from_model_args(
            conv2d_args=params.conv_args.img_neck.fpn_convs,
            weights=params.img_neck.fpn_convs.conv.weight,
            bias=params.img_neck.fpn_convs.conv.bias,
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.img_neck = TtFPN(lateral_config, fpn_config, device)
        self.with_img_neck = True

        # Initialize TTNN Encoder
        logger.info("Initializing TTNN BEVFormerEncoder...")
        tt_encoder = TtBEVFormerEncoder(
            params=params.transformer.encoder,
            device=device,
            num_layers=num_encoder_layers,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            embed_dims=embed_dims,
            num_heads=8,
            num_levels=1,
            num_points=8,
            im2col_step=192,
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )

        # Initialize TTNN Decoder
        logger.info("Initializing TTNN MapTRDecoder...")
        tt_decoder = TtMapTRDecoder(
            num_layers=num_decoder_layers,
            embed_dims=embed_dims,
            num_heads=8,
            params=params.transformer.decoder,
            params_branches=params.head.branches,
            device=device,
            feedforward_channels=512,
        )

        # Initialize TTNN Transformer
        logger.info("Initializing TTNN MapTRPerceptionTransformer...")
        self.transformer = TtMapTRPerceptionTransformer(
            params=params.transformer,
            device=device,
            encoder=tt_encoder,
            decoder=tt_decoder,
            embed_dims=embed_dims,
            num_feature_levels=1,
            num_cams=6,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            can_bus_norm=True,
            use_cams_embeds=True,
            rotate_center=[100, 100],
            fuser=None,
        )

        # Initialize TTNN Positional Encoding
        logger.info("Initializing TTNN Positional Encoding...")
        positional_encoding = TtLearnedPositionalEncoding(
            params=params.head.positional_encoding,
            device=device,
            num_feats=embed_dims // 2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
        )

        # Initialize TTNN Head
        logger.info("Initializing TTNN MapTRHead...")
        self.pts_bbox_head = TtMapTRHead(
            params=params.head,
            device=device,
            transformer=self.transformer,
            positional_encoding=positional_encoding,
            bbox_coder=None,  # Not used in inference
            embed_dims=embed_dims,
            num_classes=num_classes,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
            num_vec=num_vec,
            num_pts_per_vec=num_pts_per_vec,
            num_decoder_layers=num_decoder_layers,
            use_vadv2_params=False,  # Using our own params structure
            with_box_refine=True,
        )

        # Temporal state
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def extract_img_feat_ttnn(
        self,
        img: ttnn.Tensor,
        img_metas: Optional[List[Dict]] = None,
        len_queue: Optional[int] = None,
    ) -> Optional[List[ttnn.Tensor]]:
        """Extract features from images using TTNN backbone and FPN.

        Args:
            img: Input images as TTNN tensor.
            img_metas: Image meta information.
            len_queue: Length of temporal queue.

        Returns:
            List of TTNN feature tensors in (B, N, C, H, W) format.
        """
        if img is None:
            return None

        B = img.shape[0] if len(img.shape) == 5 else 1

        # Handle input shape
        if len(img.shape) == 5 and img.shape[0] == 1:
            img = ttnn.squeeze(img, 0)
        elif len(img.shape) == 4 and img.shape[0] > 1:
            B, N, C, H, W = img.shape
            img = ttnn.reshape(img, (B * N, C, H, W))

        # Permute to NHWC format for TTNN conv
        img = ttnn.permute(img, (0, 2, 3, 1))
        N, H, W, C = img.shape
        batch_size = N

        # Flatten for backbone
        img = ttnn.reshape(img, (1, 1, N * H * W, C))

        # Backbone forward (TTNN)
        img_feats = self.img_backbone(img, batch_size=batch_size)

        # FPN forward (TTNN)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # Reshape features to (B, N, C, H, W) format
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.unsqueeze(img_feat, 0)
            img_feat = ttnn.to_layout(img_feat, layout=ttnn.ROW_MAJOR_LAYOUT)
            img_feat = ttnn.sharded_to_interleaved(img_feat)

            # Reshape to (N, H, W, C) then permute to (N, C, H, W)
            feat_shape = img_feat.shape
            if len(feat_shape) == 4:
                img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))

            # Reshape to (B, N, C, H, W) - for single camera, N=1
            # For multi-camera, we need to handle this properly
            BN, C, H, W = img_feat.shape
            N_per_batch = BN // B if B > 0 else BN
            img_feat = ttnn.reshape(img_feat, (B, N_per_batch, C, H, W))
            img_feats_reshaped.append(img_feat)

        return img_feats_reshaped

    def extract_feat(
        self,
        img,
        img_metas: Optional[List[Dict]] = None,
        len_queue: Optional[int] = None,
    ):
        """Extract features from images.

        Args:
            img: Input images (TTNN tensor).
            img_metas: Image meta information.
            len_queue: Length of temporal queue.

        Returns:
            List of feature maps in (B, N, C, H, W) format.
        """
        return self.extract_img_feat_ttnn(img, img_metas, len_queue=len_queue)

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
        img: Optional[ttnn.Tensor] = None,
        points: Optional[ttnn.Tensor] = None,
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
        x: List[ttnn.Tensor],
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
        # Convert features to format expected by head
        # x is in (B, N, C, H, W) format, head expects list of (B*N, H, W, C) in NHWC
        tt_feats = []
        for feat in x:
            B, N, C, H, W = feat.shape
            # Reshape to (BN, C, H, W) then permute to (BN, H, W, C)
            feat_reshaped = ttnn.reshape(feat, (B * N, C, H, W))
            feat_reshaped = ttnn.permute(feat_reshaped, (0, 2, 3, 1))
            tt_feats.append((feat_reshaped,))  # Wrap in tuple to match expected format

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
