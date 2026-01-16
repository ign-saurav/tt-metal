# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import numpy as np
from loguru import logger

from models.experimental.mapTR.reference.pytorch_transformer import MapTRPerceptionTransformer
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.modules.decoder import MapDetectionTransformerDecoder


MAPTR_WEIGHTS_PATH = "models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_transformer_forward_with_weights(device, reset_seeds):
    """Test MapTR transformer forward pass with pretrained weights.

    Loads weights from checkpoint and runs full forward pass.
    """
    torch.manual_seed(42)

    # Config (maptr_tiny_r50_24e_bevformer)
    embed_dims = 256
    bev_h, bev_w = 200, 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_cams = 6
    batch_size = 1
    num_vec = 50
    num_pts_per_vec = 20
    num_query = num_vec * num_pts_per_vec

    # Load weights
    assert os.path.exists(MAPTR_WEIGHTS_PATH), f"Weights not found: {MAPTR_WEIGHTS_PATH}"
    checkpoint = torch.load(MAPTR_WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Extract transformer weights
    prefix = "pts_bbox_head.transformer."
    transformer_weights = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    logger.info(f"Loaded {len(transformer_weights)} transformer weight keys")

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

    # Load weights
    missing, unexpected = transformer.load_state_dict(transformer_weights, strict=False)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    assert len(missing) == 0, f"Missing {len(missing)} keys: {missing[:5]}"
    transformer.eval()

    # Prepare inputs
    torch.manual_seed(123)
    feat_h, feat_w = 28, 50
    mlvl_feats = [torch.randn(batch_size, num_cams, embed_dims, feat_h, feat_w)]
    bev_queries = torch.randn(bev_h * bev_w, embed_dims)
    object_query_embed = torch.randn(num_query, embed_dims * 2)
    bev_pos = torch.randn(batch_size, embed_dims, bev_h, bev_w)
    img_metas = [
        {
            "can_bus": np.zeros(18, dtype=np.float32),
            "lidar2img": np.eye(4, dtype=np.float32)[np.newaxis].repeat(num_cams, axis=0),
            "img_shape": [(900, 1600)] * num_cams,
        }
    ]

    # Forward pass
    logger.info("Running transformer forward pass...")
    with torch.no_grad():
        bev_embed, inter_states, init_reference, inter_references = transformer(
            mlvl_feats=mlvl_feats,
            lidar_feat=None,
            bev_queries=bev_queries,
            object_query_embed=object_query_embed,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
        )

    # Log results
    logger.info(f"✓ bev_embed: {bev_embed.shape}")
    logger.info(f"✓ inter_states: {inter_states.shape}")
    logger.info(f"✓ init_reference: {init_reference.shape}")
    logger.info(f"✓ inter_references: {inter_references.shape}")

    # Verify shapes
    assert bev_embed.shape == (bev_h * bev_w, batch_size, embed_dims)
    assert inter_states.shape[0] == 6  # 6 decoder layers
    assert init_reference.shape == (batch_size, num_query, 2)

    logger.info("✓ Transformer forward pass completed successfully!")
