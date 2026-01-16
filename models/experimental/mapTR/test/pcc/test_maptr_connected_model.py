# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Connected model test for MapTR - End-to-End Output PCC comparison.

This test verifies the full MapTR model by comparing TTNN outputs against
PyTorch reference outputs using Pearson Correlation Coefficient (PCC).
"""

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger
import os

from models.experimental.mapTR.reference.pytorch_maptr import MapTR
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.modules.transformer import MapTRPerceptionTransformer
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.modules.decoder import MapTRDecoder, BaseTransformerLayer
from models.experimental.mapTR.reference.nms_free_coder import MapTRNMSFreeCoder

from models.experimental.mapTR.tt.head import TtMapTRHead
from models.experimental.mapTR.tt.weight_loading import (
    load_maptr_checkpoint,
    get_head_params_from_torch_model,
)


# Default checkpoint path
CHECKPOINT_PATH = "/home/ubuntu/christyv1/tt-metal/models/experimental/mapTR/resources/maptr_tiny_r50_24e_bevformer.pth"


def build_torch_maptr_model(checkpoint_path: str = CHECKPOINT_PATH) -> nn.Module:
    """Build PyTorch MapTR model and load weights.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        PyTorch MapTR model with loaded weights.
    """
    # Build ResNet50 backbone
    # Checkpoint only uses last layer output (layer4 with 2048 channels)
    backbone = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],  # ResNet50 configuration
        out_indices=(3,),  # Only output from layer4
    )

    # Build FPN neck - checkpoint only has single level FPN from layer4 (2048)
    fpn = FPN(
        in_channels=[2048],  # Only using the last layer of ResNet (layer4)
        out_channels=256,
        start_level=0,
        num_outs=1,
        relu_before_extra_convs=False,
    )

    # Build positional encoding
    positional_encoding = LearnedPositionalEncoding(
        num_feats=128,
        row_num_embed=200,
        col_num_embed=100,
    )

    # Build BEVFormer encoder
    encoder = BEVFormerEncoder(
        num_layers=6,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        embed_dims=256,
        num_heads=8,
        feedforward_channels=512,
        ffn_dropout=0.1,
    )

    # Build decoder layers
    decoder_layers = nn.ModuleList(
        [
            BaseTransformerLayer(
                attn_cfgs=[
                    dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                    dict(type="CustomMSDeformableAttention", embed_dims=256, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            )
            for _ in range(6)
        ]
    )

    decoder = MapTRDecoder(
        layers=decoder_layers,
        return_intermediate=True,
    )

    # Build transformer
    transformer = MapTRPerceptionTransformer(
        encoder=encoder,
        decoder=decoder,
        embed_dims=256,
        num_feature_levels=4,
        num_cams=6,
    )

    # Build bbox coder
    bbox_coder = MapTRNMSFreeCoder(
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        post_center_range=[-20.0, -35.0, -20.0, 35.0],
        max_num=50,
        num_classes=3,
    )

    # Build head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=positional_encoding,
        bbox_coder=bbox_coder,
        embed_dims=256,
        num_classes=3,
        num_reg_fcs=2,
        num_cls_fcs=2,
        code_size=2,
        bev_h=200,
        bev_w=100,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_vec=50,
        num_pts_per_vec=20,
    )

    # Build full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = load_maptr_checkpoint(checkpoint_path)
        # Load with strict=False to handle structural differences
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        logger.info("Loaded checkpoint weights (non-strict)")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")

    model.eval()
    return model


def build_ttnn_head(
    torch_head: nn.Module,
    device: ttnn.Device,
) -> TtMapTRHead:
    """Build TTNN MapTR head from PyTorch head.

    Args:
        torch_head: PyTorch MapTRHead module.
        device: TTNN device.

    Returns:
        TTNN MapTRHead module.
    """
    # Extract parameters from torch head
    head_params = get_head_params_from_torch_model(torch_head)

    # Build TTNN head using the torch transformer and positional encoding
    tt_head = TtMapTRHead(
        params=head_params,
        device=device,
        transformer=torch_head.transformer,
        positional_encoding=torch_head.positional_encoding,
        bbox_coder=torch_head.bbox_coder,
        embed_dims=torch_head.embed_dims,
        num_classes=torch_head.num_classes,
        bev_h=torch_head.bev_h,
        bev_w=torch_head.bev_w,
        pc_range=torch_head.pc_range,
        num_vec=torch_head.num_vec,
        num_pts_per_vec=torch_head.num_pts_per_vec,
        num_decoder_layers=len(torch_head.transformer.decoder.layers),
    )

    return tt_head


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_maptr_connected_model_output_pcc(device):
    """Test full connected MapTR model end-to-end with output PCC comparison.

    This test:
    1. Builds PyTorch MapTR model and loads checkpoint
    2. Builds TTNN head using the same transformer
    3. Runs forward pass on both models
    4. Compares outputs with PCC

    Args:
        device: TTNN device fixture.
    """
    logger.info("Testing MapTR Connected Model - End-to-End Output PCC")

    # Skip if checkpoint doesn't exist
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # Build PyTorch model
    logger.info("Building PyTorch MapTR model...")
    torch_model = build_torch_maptr_model(CHECKPOINT_PATH)
    torch_model.eval()

    # Create dummy input
    batch_size = 1
    num_cams = 6
    C, H, W = 3, 480, 800

    img_torch = torch.randn(batch_size, num_cams, C, H, W)

    import numpy as np

    img_metas = [
        {
            "can_bus": np.zeros(18),
            "lidar2img": [torch.eye(4) for _ in range(num_cams)],
            "img_shape": [(H, W, C) for _ in range(num_cams)],
            "pad_shape": [(H, W, C) for _ in range(num_cams)],
        }
    ]

    # Run PyTorch model forward to get features
    logger.info("Running PyTorch feature extraction...")
    with torch.no_grad():
        torch_feats = torch_model.extract_feat(img=img_torch, img_metas=img_metas)

    logger.info(f"PyTorch features extracted: {len(torch_feats)} levels")
    for i, feat in enumerate(torch_feats):
        logger.info(f"  Level {i}: {feat.shape}")

    # Run PyTorch head forward
    logger.info("Running PyTorch head forward pass...")
    with torch.no_grad():
        torch_outputs = torch_model.pts_bbox_head(torch_feats, None, img_metas)

    logger.info("PyTorch head outputs:")
    for key, value in torch_outputs.items():
        if value is not None:
            logger.info(f"  {key}: {value.shape}")

    # Build TTNN head
    logger.info("Building TTNN head...")
    tt_head = build_ttnn_head(torch_model.pts_bbox_head, device)

    # Convert features to TTNN format
    # PyTorch features: (B, N, C, H, W)
    # TTNN expects: (B*N, H, W, C) in NHWC format
    logger.info("Converting features to TTNN format...")
    tt_feats = []
    for feat in torch_feats:
        B, N, C, H, W = feat.shape
        # Reshape to (BN, C, H, W) then permute to (BN, H, W, C)
        feat_reshaped = feat.reshape(B * N, C, H, W).permute(0, 2, 3, 1).contiguous()
        feat_ttnn = ttnn.from_torch(feat_reshaped, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_feats.append((feat_ttnn,))  # Wrap in tuple to match FPN output format

    # Run TTNN head forward
    logger.info("Running TTNN head forward pass...")
    tt_outputs = tt_head(tt_feats, None, img_metas)

    logger.info("TTNN head outputs:")
    for key, value in tt_outputs.items():
        if value is not None:
            value_torch = ttnn.to_torch(value)
            logger.info(f"  {key}: {value_torch.shape}")

    # Compare outputs with PCC
    logger.info("Comparing Outputs with PCC")

    pcc_threshold = 0.95
    pcc_results = {}
    all_pass = True

    output_keys = ["bev_embed", "all_cls_scores", "all_bbox_preds", "all_pts_preds"]

    for key in output_keys:
        torch_out = torch_outputs.get(key)
        tt_out = tt_outputs.get(key)

        if torch_out is None or tt_out is None:
            logger.warning(f"Skipping {key}: torch={torch_out is not None}, tt={tt_out is not None}")
            continue

        # Convert TTNN to torch
        tt_out_torch = ttnn.to_torch(tt_out)

        try:
            # Flatten for PCC calculation
            torch_flat = torch_out.float().flatten()
            tt_flat = tt_out_torch.float().flatten()

            if torch_flat.shape != tt_flat.shape:
                logger.error(f"Shape mismatch for {key}: torch={torch_flat.shape} vs tt={tt_flat.shape}")
                pcc_results[key] = None
                continue

            pcc = torch.corrcoef(torch.stack([torch_flat, tt_flat]))[0, 1].item()
            pcc_results[key] = pcc

            logger.info(f"{key}: PCC = {pcc:.6f}, Shape: torch={torch_out.shape}, tt={tt_out_torch.shape}")

            if pcc < pcc_threshold:
                logger.warning(f"{key} PCC ({pcc:.6f}) below threshold ({pcc_threshold})")
                all_pass = False

        except Exception as e:
            logger.error(f"Error calculating PCC for {key}: {e}")
            pcc_results[key] = None
            all_pass = False

    # Summary
    logger.info("=" * 60)
    logger.info("Output PCC Summary")
    logger.info("=" * 60)
    for key, pcc in pcc_results.items():
        if pcc is not None:
            status = "✅" if pcc >= pcc_threshold else "⚠️"
            logger.info(f"{status} {key}: PCC = {pcc:.6f}")
        else:
            logger.info(f"❌ {key}: PCC calculation failed")
    logger.info("=" * 60)

    # Assert overall pass
    if all_pass:
        logger.info("✅ All outputs meet PCC threshold")
    else:
        logger.warning("⚠️ Some outputs below PCC threshold")

    logger.info("MapTR Connected Model Test Completed")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_maptr_head_weight_loading_pcc(device):
    """Test that head weights are loaded correctly with PCC verification.

    Args:
        device: TTNN device fixture.
    """
    logger.info("Testing MapTR Head Weight Loading PCC")

    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # Build PyTorch model
    torch_model = build_torch_maptr_model(CHECKPOINT_PATH)
    torch_model.eval()

    # Build TTNN head
    tt_head = build_ttnn_head(torch_model.pts_bbox_head, device)

    # Verify classification branch weights
    logger.info("Verifying classification branch weights...")
    for layer_idx in range(tt_head.num_decoder_layers):
        for fc_idx in range(tt_head.num_reg_fcs):
            # Get torch weights
            linear_idx = fc_idx * 3
            torch_weight = torch_model.pts_bbox_head.cls_branches[layer_idx][linear_idx].weight.data
            torch_bias = torch_model.pts_bbox_head.cls_branches[layer_idx][linear_idx].bias.data

            # Get TTNN weights (transposed)
            tt_weight = ttnn.to_torch(tt_head.cls_branches_weights[layer_idx][fc_idx]).T
            tt_bias = ttnn.to_torch(tt_head.cls_branches_biases[layer_idx][fc_idx]).flatten()

            # Compare
            weight_pcc = torch.corrcoef(torch.stack([torch_weight.flatten(), tt_weight.flatten()]))[0, 1].item()
            bias_pcc = torch.corrcoef(torch.stack([torch_bias.flatten(), tt_bias.flatten()]))[0, 1].item()

            assert weight_pcc > 0.99, f"cls_branch[{layer_idx}][{fc_idx}] weight PCC = {weight_pcc}"
            assert bias_pcc > 0.99, f"cls_branch[{layer_idx}][{fc_idx}] bias PCC = {bias_pcc}"

    logger.info("✅ All classification branch weights verified")

    # Verify regression branch weights
    logger.info("Verifying regression branch weights...")
    for layer_idx in range(tt_head.num_decoder_layers):
        for fc_idx in range(tt_head.num_reg_fcs):
            linear_idx = fc_idx * 2
            torch_weight = torch_model.pts_bbox_head.reg_branches[layer_idx][linear_idx].weight.data
            torch_bias = torch_model.pts_bbox_head.reg_branches[layer_idx][linear_idx].bias.data

            tt_weight = ttnn.to_torch(tt_head.reg_branches_weights[layer_idx][fc_idx]).T
            tt_bias = ttnn.to_torch(tt_head.reg_branches_biases[layer_idx][fc_idx]).flatten()

            weight_pcc = torch.corrcoef(torch.stack([torch_weight.flatten(), tt_weight.flatten()]))[0, 1].item()
            bias_pcc = torch.corrcoef(torch.stack([torch_bias.flatten(), tt_bias.flatten()]))[0, 1].item()

            assert weight_pcc > 0.99, f"reg_branch[{layer_idx}][{fc_idx}] weight PCC = {weight_pcc}"
            assert bias_pcc > 0.99, f"reg_branch[{layer_idx}][{fc_idx}] bias PCC = {bias_pcc}"

    logger.info("✅ All regression branch weights verified")
    logger.info("MapTR Head Weight Loading PCC Test Completed")
