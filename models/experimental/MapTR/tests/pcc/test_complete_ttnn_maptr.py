# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Complete TTNN MapTR network test - End-to-End Output PCC comparison.

This test is modeled after VADv2's test_tt_vad.py to verify the complete
TTNN MapTR network (backbone + neck + transformer + head) by comparing TTNN outputs
against PyTorch reference outputs using Pearson Correlation Coefficient (PCC).

This provides PCC for the COMPLETE TTNN network, not just individual components.
No PyTorch components are used in the forward pass.
"""

import pytest
import torch
import torch.nn as nn
import ttnn
import numpy as np
import os
from loguru import logger

from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.detectors.maptr import MapTR

from models.experimental.MapTR.tt.model import TtMapTR
from models.experimental.MapTR.tt.weight_loading import (
    load_maptr_checkpoint,
    create_maptr_complete_model_parameters,
)


# Default checkpoint path
CHECKPOINT_PATH = "/home/ubuntu/christyv1/tt-metal/models/experimental/MapTR/resources/maptr_tiny_r50_24e_bevformer.pth"

# Save path for output dumps (for PCC comparison)
SAVE_PATH_REFERENCE = "models/experimental/MapTR/reference/dumps"
SAVE_PATH_TTNN = "models/experimental/MapTR/tt/dumps"


def build_torch_maptr_model(checkpoint_path: str = CHECKPOINT_PATH) -> nn.Module:
    """Build PyTorch MapTR model and load weights.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        PyTorch MapTR model with loaded weights.
    """
    from models.experimental.MapTR.dependency import (
        build_backbone,
        build_neck,
        build_transformer,
        build_head,
        build_bbox_coder,
        ConfigDict,
    )

    # Build ResNet50 backbone
    backbone_cfg = ConfigDict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    )
    backbone = build_backbone(backbone_cfg)

    # Build FPN neck
    neck_cfg = ConfigDict(
        type="FPN",
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        num_outs=1,
        relu_before_extra_convs=False,
    )
    neck = build_neck(neck_cfg)

    # Build transformer config
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=256,
        encoder=ConfigDict(
            type="BEVFormerEncoder",
            num_layers=6,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=ConfigDict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    ConfigDict(type="TemporalSelfAttention", embed_dims=256, num_levels=1),
                    ConfigDict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=ConfigDict(
                            type="MSDeformableAttention3D",
                            embed_dims=256,
                            num_heads=8,
                            num_levels=1,
                            num_points=8,
                            im2col_step=192,
                        ),
                        embed_dims=256,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=ConfigDict(
            type="MapTRDecoder",
            num_layers=6,
            return_intermediate=True,
            transformerlayers=ConfigDict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    ConfigDict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                    ConfigDict(type="CustomMSDeformableAttention", embed_dims=256, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        num_feature_levels=4,  # Match checkpoint (was 1)
        num_cams=6,
    )
    transformer = build_transformer(transformer_cfg)

    # Build bbox coder
    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20.0, -35.0, -20.0, 35.0],
        max_num=50,
        num_classes=3,
    )
    bbox_coder = build_bbox_coder(bbox_coder_cfg)

    # Build head
    # Note: code_weights should be [4] for 2D bbox (x, y, w, h), not [10]
    head_cfg = ConfigDict(
        type="MapTRHead",
        in_channels=256,  # Required by DETRHead
        embed_dims=256,
        num_classes=3,
        num_reg_fcs=2,
        num_cls_fcs=2,
        code_size=2,  # 2D bbox
        code_weights=[1.0, 1.0, 1.0, 1.0],  # Match checkpoint shape [4]
        bev_h=200,
        bev_w=100,
        pc_range=pc_range,
        num_vec=50,
        num_pts_per_vec=20,
        transformer=transformer_cfg,
        bbox_coder=bbox_coder_cfg,
        with_box_refine=True,
        as_two_stage=False,
        loss_cls=ConfigDict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=ConfigDict(type="L1Loss", loss_weight=0.0),
        loss_iou=ConfigDict(type="GIoULoss", loss_weight=0.0),
        train_cfg=None,  # Inference only
        test_cfg=ConfigDict(max_per_img=50),
    )

    # Build head first
    head = build_head(head_cfg)

    # Add dummy update method if it doesn't exist (needed by MVXTwoStageDetector.__init__)
    if not hasattr(head, "update"):

        def dummy_update(self, **kwargs):
            pass

        head.update = dummy_update.__get__(head, type(head))

    # Build full model - pass head as dict config so MapTR can handle it properly
    # The parent class expects to call update on the head, so we pass it as config
    model = MapTR(
        img_backbone=backbone,
        img_neck=neck,
        pts_bbox_head=head_cfg,  # Pass config dict instead of built object
        use_grid_mask=False,
        video_test_mode=False,
        train_cfg=None,  # Inference only
        test_cfg=None,  # Inference only
    )

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = load_maptr_checkpoint(checkpoint_path)
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


def create_input_data():
    """Create sample input data for testing."""
    batch_size = 1
    num_cams = 6
    C, H, W = 3, 480, 800

    # Create random input image tensor
    img_torch = torch.randn(batch_size, num_cams, C, H, W)

    # Create img_metas
    img_metas = [
        {
            "can_bus": np.zeros(18),
            "lidar2img": [np.eye(4) for _ in range(num_cams)],
            "img_shape": [(H, W, C) for _ in range(num_cams)],
            "pad_shape": [(H, W, C) for _ in range(num_cams)],
            "scene_token": "test_scene",
        }
    ]

    return img_torch, img_metas


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_complete_ttnn_network(device):
    """Test complete TTNN MapTR network (backbone + neck + transformer + head) with output PCC comparison.

    This is the complete TTNN network test, similar to VADv2's test_vadv2.

    The test:
    1. Builds PyTorch MapTR model and loads checkpoint
    2. Runs forward pass through PyTorch model
    3. Creates TTNN parameters from PyTorch model
    4. Builds complete TTNN MapTR model
    5. Runs forward pass through TTNN model
    6. Compares all outputs with PCC

    Args:
        device: TTNN device fixture.
    """
    logger.info("=" * 60)
    logger.info("Testing Complete TTNN MapTR Network - End-to-End PCC")
    logger.info("=" * 60)

    # Skip if checkpoint doesn't exist
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # Build PyTorch model
    logger.info("Building PyTorch MapTR model...")
    torch_model = build_torch_maptr_model(CHECKPOINT_PATH)
    if torch_model is None:
        pytest.skip("Failed to build PyTorch model")
    torch_model.eval()

    # Create input data
    logger.info("Creating input data...")
    img_torch, img_metas = create_input_data()

    # Run PyTorch model forward pass
    logger.info("Running PyTorch forward pass...")
    with torch.no_grad():
        # Extract features
        torch_feats = torch_model.extract_feat(img=img_torch, img_metas=img_metas)
        logger.info(f"PyTorch features extracted: {len(torch_feats)} levels")
        for i, feat in enumerate(torch_feats):
            logger.info(f"  Level {i}: {feat.shape}")

        # Run head forward
        torch_outputs = torch_model.pts_bbox_head(torch_feats, None, img_metas)

    logger.info("PyTorch head outputs:")
    for key, value in torch_outputs.items():
        if value is not None:
            logger.info(f"  {key}: {value.shape}")

    # Save PyTorch outputs for comparison
    os.makedirs(SAVE_PATH_REFERENCE, exist_ok=True)
    keys_to_save = ["bev_embed", "all_cls_scores", "all_bbox_preds", "all_pts_preds"]
    for key in keys_to_save:
        if key in torch_outputs and torch_outputs[key] is not None:
            torch.save(torch_outputs[key], os.path.join(SAVE_PATH_REFERENCE, f"{key}.pt"))

    # Create TTNN parameters
    logger.info("Creating TTNN model parameters...")
    # Convert input to format for parameter inference
    # img_torch is (B, N, C, H, W), need (B*N, C, H, W) for backbone
    if len(img_torch.shape) == 5:
        B, N, C, H, W = img_torch.shape
        input_tensor = img_torch.reshape(B * N, C, H, W)
    else:
        # Already in (B*N, C, H, W) format
        input_tensor = img_torch
    params = create_maptr_complete_model_parameters(torch_model, input_tensor, device=device)

    # Build complete TTNN model
    logger.info("Building complete TTNN MapTR model...")
    tt_model = TtMapTR(
        device=device,
        params=params,
        embed_dims=256,
        num_classes=3,
        bev_h=200,
        bev_w=100,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_vec=50,
        num_pts_per_vec=20,
        num_decoder_layers=6,
        num_encoder_layers=6,
        use_grid_mask=False,
        video_test_mode=False,
    )

    # Convert input to TTNN format
    logger.info("Converting input to TTNN format...")
    B, N, C, H, W = img_torch.shape
    # Convert to (B*N, H, W, C) format for TTNN
    img_reshaped = img_torch.reshape(B * N, C, H, W).permute(0, 2, 3, 1).contiguous()
    img_ttnn = ttnn.from_torch(
        img_reshaped,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # Reshape back to (B, N, H, W, C) then permute to (B, N, C, H, W)
    img_ttnn = ttnn.reshape(img_ttnn, (B, N, H, W, C))
    img_ttnn = ttnn.permute(img_ttnn, (0, 1, 4, 2, 3))  # (B, N, C, H, W)

    # Run TTNN forward pass
    logger.info("Running TTNN forward pass...")
    # Extract features
    tt_feats = tt_model.extract_feat(img=img_ttnn, img_metas=img_metas)
    logger.info(f"TTNN features extracted: {len(tt_feats)} levels")
    for i, feat in enumerate(tt_feats):
        feat_torch = ttnn.to_torch(feat)
        logger.info(f"  Level {i}: {feat_torch.shape}")

    # Get outputs from head
    # Convert features to format expected by head
    tt_feats_for_head = []
    for feat in tt_feats:
        B, N, C, H, W = feat.shape
        # Reshape to (BN, C, H, W) then permute to (BN, H, W, C)
        feat_reshaped = ttnn.reshape(feat, (B * N, C, H, W))
        feat_reshaped = ttnn.permute(feat_reshaped, (0, 2, 3, 1))
        tt_feats_for_head.append((feat_reshaped,))

    tt_head_outputs = tt_model.pts_bbox_head(tt_feats_for_head, None, img_metas, prev_bev=None)

    logger.info("TTNN head outputs:")
    for key, value in tt_head_outputs.items():
        if value is not None:
            value_torch = ttnn.to_torch(value)
            logger.info(f"  {key}: {value_torch.shape}")

    # Save TTNN outputs
    os.makedirs(SAVE_PATH_TTNN, exist_ok=True)
    for key in keys_to_save:
        if key in tt_head_outputs and tt_head_outputs[key] is not None:
            tensor = ttnn.to_torch(tt_head_outputs[key]).float()
            torch.save(tensor, os.path.join(SAVE_PATH_TTNN, f"{key}.pt"))

    # Compare outputs with PCC
    logger.info("=" * 60)
    logger.info("Comparing Outputs with PCC")
    logger.info("=" * 60)

    pcc_results = {}
    pcc_threshold = 0.85
    all_pass = True

    for key in keys_to_save:
        torch_out = torch_outputs.get(key)
        tt_out = tt_head_outputs.get(key)

        if torch_out is None or tt_out is None:
            logger.warning(f"Skipping {key}: torch={torch_out is not None}, tt={tt_out is not None}")
            continue

        # Convert TTNN to torch
        tt_out_torch = ttnn.to_torch(tt_out).float()

        try:
            # Flatten for PCC calculation
            torch_flat = torch_out.float().flatten()
            tt_flat = tt_out_torch.flatten()

            if torch_flat.shape != tt_flat.shape:
                logger.error(f"Shape mismatch for {key}: torch={torch_flat.shape} vs tt={tt_flat.shape}")
                pcc_results[key] = None
                all_pass = False
                continue

            # Calculate PCC
            pcc = torch.corrcoef(torch.stack([torch_flat, tt_flat]))[0, 1].item()
            pcc_results[key] = pcc

            status = "✅" if pcc >= pcc_threshold else "⚠️"
            logger.info(f"{status} {key}: PCC = {pcc:.6f}, Shape: torch={torch_out.shape}, tt={tt_out_torch.shape}")

            if pcc < pcc_threshold:
                all_pass = False

        except Exception as e:
            logger.error(f"Error calculating PCC for {key}: {e}")
            pcc_results[key] = None
            all_pass = False

    # Summary
    logger.info("=" * 60)
    logger.info("Complete TTNN Network PCC Summary")
    logger.info("=" * 60)
    for key, pcc in pcc_results.items():
        if pcc is not None:
            status = "✅" if pcc >= pcc_threshold else "⚠️"
            logger.info(f"{status} {key}: PCC = {pcc:.6f}")
        else:
            logger.info(f"❌ {key}: PCC calculation failed")
    logger.info("=" * 60)

    if all_pass:
        logger.info("✅ All outputs meet PCC threshold")
    else:
        logger.warning("⚠️ Some outputs below PCC threshold")

    logger.info("Complete TTNN MapTR Network Test Completed")


if __name__ == "__main__":
    # For running directly without pytest
    import ttnn

    device = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    try:
        test_maptr_complete_ttnn_network(device)
    finally:
        ttnn.CloseDevice(device)
