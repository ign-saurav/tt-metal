# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test file for MapTR Head - loads pretrained weights and runs forward pass."""

import pytest
import torch
import numpy as np
from loguru import logger

from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_transformer import MapTRPerceptionTransformer
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.modules.decoder import MapDetectionTransformerDecoder
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding


MAPTR_WEIGHTS_PATH = "models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_head_forward_with_weights(device, reset_seeds):
    """Test MapTR head forward pass with pretrained weights.

    Loads weights from checkpoint and runs full forward pass through:
    - Positional encoding
    - Transformer (encoder + decoder)
    - Classification and regression branches
    - get_bboxes decoding
    """
    torch.manual_seed(42)

    # Config (maptr_tiny_r50_24e_bevformer)
    embed_dims = 256
    bev_h, bev_w = 200, 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_cams = 6
    batch_size = 1
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6

    # Load weights
    checkpoint = torch.load(MAPTR_WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    head_weights = {k[len("pts_bbox_head.") :]: v for k, v in state_dict.items() if k.startswith("pts_bbox_head.")}
    logger.info(f"Loaded {len(head_weights)} head weight keys")

    # Create encoder
    encoder = BEVFormerEncoder(
        num_layers=1,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=embed_dims,
        num_heads=4,
        feedforward_channels=512,
        ffn_dropout=0.1,
    )

    # Create decoder
    decoder = MapDetectionTransformerDecoder(
        num_layers=6,
        embed_dim=embed_dims,
        num_heads=8,
    )

    # Create transformer
    transformer = MapTRPerceptionTransformer(
        encoder=encoder,
        decoder=decoder,
        embed_dims=embed_dims,
        num_feature_levels=4,
        num_cams=num_cams,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )

    # Create positional encoding
    positional_encoding = LearnedPositionalEncoding(
        num_feats=embed_dims // 2,
        row_num_embed=bev_h,
        col_num_embed=bev_w,
    )

    # Create head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=positional_encoding,
        embed_dims=embed_dims,
        num_classes=num_classes,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
    )

    # Load weights
    missing, unexpected = head.load_state_dict(head_weights, strict=False)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    assert len(missing) == 0, f"Missing {len(missing)} keys: {missing[:5]}"
    head.eval()

    # Prepare inputs
    torch.manual_seed(123)
    feat_h, feat_w = 28, 50
    mlvl_feats = [torch.randn(batch_size, num_cams, embed_dims, feat_h, feat_w)]
    img_metas = [
        {
            "can_bus": np.zeros(18, dtype=np.float32),
            "lidar2img": np.eye(4, dtype=np.float32)[np.newaxis].repeat(num_cams, axis=0),
            "img_shape": [(900, 1600)] * num_cams,
        }
    ]

    # Forward pass
    logger.info("Running MapTRHead forward pass...")
    with torch.no_grad():
        outputs = head(mlvl_feats, lidar_feat=None, img_metas=img_metas)

    # Verify output shapes
    logger.info(f"✓ bev_embed: {outputs['bev_embed'].shape}")
    logger.info(f"✓ all_cls_scores: {outputs['all_cls_scores'].shape}")
    logger.info(f"✓ all_bbox_preds: {outputs['all_bbox_preds'].shape}")
    logger.info(f"✓ all_pts_preds: {outputs['all_pts_preds'].shape}")

    assert outputs["bev_embed"].shape == (bev_h * bev_w, batch_size, embed_dims)
    assert outputs["all_cls_scores"].shape == (num_decoder_layers, batch_size, num_vec, num_classes)
    assert outputs["all_bbox_preds"].shape == (num_decoder_layers, batch_size, num_vec, 4)
    assert outputs["all_pts_preds"].shape == (num_decoder_layers, batch_size, num_vec, num_pts_per_vec, 2)

    # Test get_bboxes decoding
    results = head.get_bboxes(outputs, img_metas)
    assert len(results) == batch_size
    bboxes, scores, labels, pts = results[0]

    logger.info(f"✓ bboxes: {bboxes.shape}, scores: {scores.shape}, labels: {labels.shape}, pts: {pts.shape}")
    assert bboxes.shape == (num_vec, 4)
    assert scores.shape == (num_vec,)
    assert labels.shape == (num_vec,)
    assert pts.shape == (num_vec, num_pts_per_vec, 2)

    logger.info("✓ MapTRHead forward pass and decoding completed successfully!")
