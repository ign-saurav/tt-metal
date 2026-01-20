# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
import os
import torch
import ttnn

from models.experimental.MapTR.tt.backbone import TtResNet50
from models.experimental.MapTR.tt.fpn import TtFPN
from models.experimental.MapTR.tt.head import TtMapTRHead
from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.decoder import TtMapTRDecoder


def pred2result(bboxes, scores, labels, pts, attrs=None):
    """Convert detection results to a dictionary.

    Args:
        bboxes: Bounding boxes tensor.
        scores: Prediction scores tensor.
        labels: Label tensor.
        pts: Points tensor.
        attrs: Optional attributes tensor.

    Returns:
        Dictionary with detection results.
    """
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        pts_3d=pts.to("cpu"),
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


class TtMapTR:
    """TTNN implementation of MapTR for map element detection.

    Args:
        device: TTNN device.
        params: Preprocessed model parameters.
        use_grid_mask: Whether to use grid mask (training only).
        img_backbone: Whether to use image backbone.
        img_neck: Whether to use image neck (FPN).
        pts_bbox_head: Whether to use bbox head.
        video_test_mode: Whether to use temporal information during inference.
        modality: Input modality ('vision' or 'fusion').
        bev_h: BEV height.
        bev_w: BEV width.
        pc_range: Point cloud range.
        num_vec: Number of map vectors.
        num_pts_per_vec: Number of points per vector.
        num_classes: Number of classes.
        embed_dims: Embedding dimensions.
    """

    def __init__(
        self,
        device,
        params,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        modality="vision",
        bev_h=200,
        bev_w=100,
        pc_range=None,
        num_vec=50,
        num_pts_per_vec=20,
        num_classes=3,
        embed_dims=256,
    ):
        super(TtMapTR, self).__init__()
        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.params = params
        self.device = device
        self.modality = modality
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.embed_dims = embed_dims

        # Temporal info for video test mode
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.with_img_neck = img_neck is not None and img_neck

        # Initialize backbone
        if img_backbone:
            self.img_backbone = TtResNet50(
                params.conv_args["img_backbone"],
                params.img_backbone,
                device,
            )
        else:
            self.img_backbone = None

        # Initialize FPN neck
        if self.with_img_neck:
            # conv_args["img_neck"] has structure: lateral_convs[0], fpn_convs[0] (Conv2dConfiguration objects)
            img_neck_args = params.conv_args["img_neck"]
            lateral_conv_config = img_neck_args["lateral_convs"][0]
            fpn_conv_config = img_neck_args["fpn_convs"][0]
            self.img_neck = TtFPN(
                lateral_conv_config,
                fpn_conv_config,
                device,
            )
        else:
            self.img_neck = None

        # Initialize transformer components
        self.transformer = None
        if pts_bbox_head and hasattr(params, "head") and hasattr(params.head, "transformer"):
            transformer_params = params.head.transformer

            encoder = None
            decoder = None

            # Try to initialize encoder
            try:
                encoder_params = transformer_params.encoder
                encoder = TtBEVFormerEncoder(
                    params=encoder_params,
                    device=device,
                    num_layers=1,  # num_layers=1 for tiny model
                    pc_range=pc_range,
                    embed_dims=embed_dims,
                )
            except (KeyError, AttributeError):
                pass

            # Try to initialize decoder
            try:
                decoder_params = transformer_params.decoder
                decoder = TtMapTRDecoder(
                    num_layers=6,
                    embed_dims=embed_dims,
                    num_heads=8,
                    params=decoder_params,
                    params_branches=params.head.branches,
                    device=device,
                )
            except (KeyError, AttributeError):
                pass

            if encoder is not None:
                # Initialize transformer (decoder can be None for partial mode)
                self.transformer = TtMapTRPerceptionTransformer(
                    params=transformer_params,
                    device=device,
                    encoder=encoder,
                    decoder=decoder,
                    embed_dims=embed_dims,
                )

        # Initialize head with transformer
        if pts_bbox_head:
            self.pts_bbox_head = TtMapTRHead(
                params=params.head,
                device=device,
                transformer=self.transformer,
                positional_encoding=None,
                embed_dims=embed_dims,
                num_classes=num_classes,
                num_reg_fcs=2,
                code_size=2,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
                num_vec=num_vec,
                num_pts_per_vec=num_pts_per_vec,
                num_decoder_layers=6,
                query_embed_type="instance_pts",
                transform_method="minmax",
                bev_encoder_type="BEVFormerEncoder",
                with_box_refine=True,
                as_two_stage=False,
            )
        else:
            self.pts_bbox_head = None

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images.

        Args:
            img: Input image tensor.
            img_metas: Image metadata.
            len_queue: Queue length for temporal processing.

        Returns:
            List of image feature tensors.
        """
        import logging

        logger = logging.getLogger(__name__)

        B = img.shape[0]
        logger.info(f"[TT] extract_img_feat input shape: {img.shape}")

        if img is not None:
            if img.shape[0] == 1 and len(img.shape) == 5:
                img = ttnn.squeeze(img, 0)
                logger.info(f"[TT] After squeeze: {img.shape}")
            elif len(img.shape) == 4 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = ttnn.reshape(img, (B * N, C, H, W))
                logger.info(f"[TT] After reshape 4D: {img.shape}")

            img = ttnn.permute(img, (0, 2, 3, 1))
            N, H, W, C = img.shape
            batch_size = img.shape[0]
            img = ttnn.reshape(img, (1, 1, N * H * W, C))
            logger.info(f"[TT] Before backbone: {img.shape}")

            img_feats = self.img_backbone(img, batch_size=batch_size)
            logger.info(f"[TT] After backbone: type={type(img_feats)}")

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())

            # Log backbone output
            for i, feat in enumerate(img_feats):
                feat_torch = ttnn.to_torch(feat)
                logger.info(
                    f"[TT] backbone[{i}] shape: {feat_torch.shape}, sample: {feat_torch.flatten()[:3].tolist()}"
                )
                # Save for comparison
                import torch

                torch.save(feat_torch, f"models/experimental/MapTR/tt/dumps/backbone_{i}.pt")
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # Log FPN output
            for i, feat in enumerate(img_feats):
                feat_torch = ttnn.to_torch(feat)
                logger.info(f"[TT] fpn[{i}] shape: {feat_torch.shape}, sample: {feat_torch.flatten()[:3].tolist()}")
                import torch

                torch.save(feat_torch, f"models/experimental/MapTR/tt/dumps/fpn_{i}.pt")

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.unsqueeze(img_feat, 0)
            img_feat = ttnn.to_layout(img_feat, layout=ttnn.ROW_MAJOR_LAYOUT)
            img_feat = ttnn.sharded_to_interleaved(img_feat)
            img_feat = ttnn.reshape(img_feat, (6, 12, 20, img_feat.shape[-1]))
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                img_feat = ttnn.reshape(img_feat, (int(B / len_queue), len_queue, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)
            else:
                img_feat = ttnn.reshape(img_feat, (B, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)

        ttnn.deallocate(img_feats[0])
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images.

        Args:
            img: Input image tensor.
            img_metas: Image metadata.
            len_queue: Queue length for temporal processing.

        Returns:
            List of feature tensors.
        """
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def __call__(self, return_loss=True, **kwargs):
        """Forward function.

        Args:
            return_loss: Whether to return loss (training mode).
            **kwargs: Additional arguments.

        Returns:
            Detection results.
        """
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        """Test forward function.

        Args:
            img_metas: Image metadata.
            img: Input image tensor.
            points: LiDAR points (optional).
            **kwargs: Additional arguments.

        Returns:
            Detection results.
        """
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        img = [img] if img is None else img

        # Handle scene token for temporal processing
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Handle ego motion (can_bus)
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        img = ttnn.unsqueeze(img[0][0], 0)
        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img,
            points=points,
            prev_bev=self.prev_frame_info["prev_bev"],
            **kwargs,
        )

        # Update temporal info
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Simple test without augmentation.

        Args:
            img_metas: Image metadata.
            img: Input image tensor.
            points: LiDAR points (optional).
            prev_bev: Previous BEV features.
            rescale: Whether to rescale results.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (new_prev_bev, bbox_results).
        """
        lidar_feat = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats,
            lidar_feat,
            img_metas,
            prev_bev=prev_bev,
            rescale=rescale,
            **kwargs,
        )

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return new_prev_bev, bbox_list

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False, **kwargs):
        """Test function for point features.

        Args:
            x: Image features.
            lidar_feat: LiDAR features (optional).
            img_metas: Image metadata.
            prev_bev: Previous BEV features.
            rescale: Whether to rescale results.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (bev_embed, bbox_results).
        """
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        # Convert outputs to torch
        outs["bev_embed"] = ttnn.to_torch(outs["bev_embed"]).float()
        outs["all_cls_scores"] = ttnn.to_torch(outs["all_cls_scores"]).float()
        outs["all_bbox_preds"] = ttnn.to_torch(outs["all_bbox_preds"]).float()
        outs["all_pts_preds"] = ttnn.to_torch(outs["all_pts_preds"]).float()

        # Save outputs for comparison (following vadv2 pattern)
        save_path = "models/experimental/MapTR/tt/dumps"
        os.makedirs(save_path, exist_ok=True)

        keys_to_save = [
            "bev_embed",
            "all_cls_scores",
            "all_bbox_preds",
            "all_pts_preds",
        ]

        for key in keys_to_save:
            tensor = outs[key]
            torch.save(tensor, os.path.join(save_path, f"{key}.pt"))

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]

        return outs["bev_embed"], bbox_results
