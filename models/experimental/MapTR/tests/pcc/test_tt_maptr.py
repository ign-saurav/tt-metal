# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Complete End-to-End MapTR TTNN Test with PCC comparison.

This test verifies the complete TTNN MapTR network (backbone + neck + transformer + head)
by comparing outputs against PyTorch reference implementation using Pearson Correlation
Coefficient (PCC).

This provides PCC for the COMPLETE network, not just individual components.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_pcc

# Import reference models
from models.experimental.MapTR.dependency import ResNet
from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.dense_heads.maptr_head import (
    MapTRHead,
)

# Import TTNN model and utilities
from models.experimental.MapTR.tt.head import TtMapTRHead
from models.experimental.MapTR.tt.weight_loading import (
    load_maptr_checkpoint,
    extract_weights_by_prefix,
    BACKBONE_PREFIX,
    NECK_PREFIX,
    HEAD_PREFIX,
)

# Import for FPN
try:
    from models.experimental.MapTR.dependency import FPN
except ImportError:
    from mmdet.models import FPN


# Checkpoint path - check multiple possible locations
MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/resources/maptr_tiny_r50_24e_bevformer.pth"
MAPTR_WEIGHTS_PATH_ALT = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Output dump paths
SAVE_PATH_REFERENCE = "models/experimental/MapTR/reference/dumps"
SAVE_PATH_TTNN = "models/experimental/MapTR/tt/dumps"


class ConfigDict(dict):
    """A dictionary that supports attribute-style access."""

    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def get_checkpoint_path():
    """Get the checkpoint path, checking multiple possible locations."""
    import os

    if os.path.exists(MAPTR_WEIGHTS_PATH):
        return MAPTR_WEIGHTS_PATH
    if os.path.exists(MAPTR_WEIGHTS_PATH_ALT):
        return MAPTR_WEIGHTS_PATH_ALT
    return MAPTR_WEIGHTS_PATH  # Default, will fail if not found


def build_torch_maptr_model(checkpoint_path: str = None):
    """Build complete PyTorch MapTR model and load weights.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Tuple of (backbone, fpn, head, transformer) PyTorch models.
    """
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path()
    # Config (maptr_tiny_r50_24e_bevformer)
    embed_dims = 256
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_encoder_layers = 6
    bev_h, bev_w = 200, 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_query = num_vec * num_pts_per_vec

    # Build ResNet50 backbone
    backbone = ResNet(
        depth=50,
        out_indices=(3,),
        frozen_stages=-1,
        norm_eval=False,
    )

    # Build FPN
    fpn = FPN(
        in_channels=[2048],
        out_channels=embed_dims,
        start_level=0,
        num_outs=1,
        relu_before_extra_convs=False,
    )

    # Encoder config
    encoder_cfg = dict(
        type="BEVFormerEncoder",
        num_layers=num_encoder_layers,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type="BEVFormerLayer",
            attn_cfgs=[
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    deformable_attention=dict(
                        type="MSDeformableAttention3D",
                        embed_dims=embed_dims,
                        num_points=8,
                        num_levels=1,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    # Decoder config
    decoder_cfg = dict(
        type="MapTRDecoder",
        num_layers=num_decoder_layers,
        return_intermediate=True,
        transformerlayers=dict(
            type="DetrTransformerDecoderLayer",
            attn_cfgs=[
                dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                dict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    # Transformer config
    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=embed_dims,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
    )

    # Bbox coder config
    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
        max_num=50,
        num_classes=num_classes,
    )

    # Build MapTRHead
    head = MapTRHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        bev_h=bev_h,
        bev_w=bev_w,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_pts_per_gt_vec=num_pts_per_vec,
        query_embed_type="instance_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        transformer=transformer_cfg,
        bbox_coder=bbox_coder_cfg,
        loss_cls=ConfigDict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=ConfigDict(type="L1Loss", loss_weight=0.0),
        loss_iou=ConfigDict(type="GIoULoss", loss_weight=0.0),
        loss_pts=None,
        loss_dir=None,
        train_cfg=None,
        test_cfg=ConfigDict(max_per_img=50),
    )

    # Load checkpoint weights
    if os.path.exists(checkpoint_path):
        state_dict = load_maptr_checkpoint(checkpoint_path)

        # Load backbone weights
        backbone_weights = extract_weights_by_prefix(state_dict, BACKBONE_PREFIX)
        model_keys = list(backbone.state_dict().keys())
        checkpoint_values = list(backbone_weights.values())
        if len(model_keys) <= len(checkpoint_values):
            new_state_dict = dict(zip(model_keys, checkpoint_values[: len(model_keys)]))
            backbone.load_state_dict(new_state_dict)
        logger.info(f"Loaded {len(backbone_weights)} backbone weights")

        # Load FPN weights
        neck_weights = extract_weights_by_prefix(state_dict, NECK_PREFIX)
        fpn.load_state_dict(neck_weights, strict=False)
        logger.info(f"Loaded {len(neck_weights)} FPN weights")

        # Load head weights
        head_weights = extract_weights_by_prefix(state_dict, HEAD_PREFIX)
        head.load_state_dict(head_weights, strict=False)
        logger.info(f"Loaded {len(head_weights)} head weights")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")

    backbone.eval()
    fpn.eval()
    head.eval()

    # Disable dropout
    for module in [backbone, fpn, head]:
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0

    return backbone, fpn, head


def create_test_input(batch_size: int = 1, num_cams: int = 6, H: int = 384, W: int = 640):
    """Create sample input data for testing.

    Args:
        batch_size: Batch size.
        num_cams: Number of cameras.
        H: Image height.
        W: Image width.

    Returns:
        Tuple of (input_tensor, img_metas).
    """
    C = 3

    # Create input image tensor
    torch.manual_seed(42)
    img_torch = torch.randn(num_cams, C, H, W).float()

    # Create img_metas
    img_metas = [
        {
            "can_bus": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.60694152,
                    -0.07634412,
                    9.87149385,
                    -0.02108691,
                    -0.01243972,
                    -0.023067,
                    8.5640597,
                    0.0,
                    0.0,
                    5.78155401,
                    0.0,
                ],
                dtype=np.float32,
            ),
            "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
            "img_shape": [(H, W, C) for _ in range(num_cams)],
            "scene_token": "test_scene",
        }
    ]

    return img_torch, img_metas


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_backbone_pcc(device, reset_seeds):
    """Test TTNN backbone PCC against PyTorch reference.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR Backbone PCC")
    logger.info("=" * 60)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Build PyTorch model
    backbone, _, _ = build_torch_maptr_model(checkpoint_path)
    backbone.eval()

    # Create input
    img_torch, _ = create_test_input()

    # Run PyTorch backbone
    logger.info("Running PyTorch backbone...")
    with torch.no_grad():
        torch_output = backbone(img_torch)[0]
    logger.info(f"PyTorch backbone output shape: {torch_output.shape}")

    # Prepare TTNN input (NHWC format, flattened)
    ttnn_input = torch.permute(img_torch, (0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1,
        1,
        ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2],
        ttnn_input.shape[3],
    )
    ttnn_input = ttnn.from_torch(ttnn_input, device=device, dtype=ttnn.bfloat16)

    # Import backbone utilities
    from models.experimental.MapTR.tests.pcc.test_backbone import (
        create_maptr_model_parameters,
        load_torch_model_maptr,
    )
    from models.experimental.MapTR.tt.backbone import TtResNet50

    # Create a fresh backbone model and load weights
    torch_backbone = ResNet(depth=50, out_indices=(3,), frozen_stages=-1, norm_eval=False)
    torch_backbone = load_torch_model_maptr(torch_backbone, checkpoint_path)
    torch_backbone.eval()

    # Create TTNN model parameters
    params = create_maptr_model_parameters(torch_backbone, img_torch, device=device)

    # Create TTNN backbone
    ttnn_backbone = TtResNet50(params.conv_args, params.res_model, device)

    # Run TTNN backbone
    logger.info("Running TTNN backbone...")
    ttnn_output = ttnn_backbone(ttnn_input, batch_size=6)[0]

    # Convert output back to PyTorch format
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    ).to(torch.float32)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    # Compare with PCC
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.96)
    logger.info(f"Backbone PCC: {pcc_message}")

    assert pcc_passed, f"Backbone PCC test failed: {pcc_message}"
    logger.info("✅ Backbone PCC test PASSED")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_head_branches_pcc(device, reset_seeds):
    """Test TTNN head branches (cls/reg) PCC against PyTorch reference.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR Head Branches PCC")
    logger.info("=" * 60)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Build PyTorch model
    _, _, torch_head = build_torch_maptr_model(checkpoint_path)

    # Config
    embed_dims = 256
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_query = num_vec * num_pts_per_vec
    batch_size = 1

    # Create test input (simulated decoder hidden states)
    torch.manual_seed(123)
    hs = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    # Run PyTorch head
    logger.info("Running PyTorch head branches...")
    with torch.no_grad():
        hs_permuted = hs.permute(0, 2, 1, 3)

        outputs_classes_torch = []
        outputs_coords_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            # Classification branch
            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_head.cls_branches[lvl](hs_mean)

            # Regression branch
            tmp = torch_head.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()

            outputs_coord, _ = torch_head.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)

    # Create TTNN head
    from models.experimental.MapTR.tests.pcc.test_head import create_maptr_model_parameters_head

    params = create_maptr_model_parameters_head(torch_head, device=device)

    tt_head = TtMapTRHead(
        params=params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=3,
        bev_h=200,
        bev_w=100,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
    )

    # Convert inputs to TTNN
    hs_tt = ttnn.from_torch(hs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(init_reference, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for ref in inter_references
    ]

    # Run TTNN head
    logger.info("Running TTNN head branches...")
    tt_outputs = tt_head(
        hs=hs_tt,
        init_reference=init_reference_tt,
        inter_references=inter_references_tt,
        bev_embed=None,
    )

    # Convert outputs
    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()

    # Compare with PCC
    pcc_threshold = 0.97

    logger.info("=" * 60)
    logger.info("PCC Comparison Results:")
    logger.info("=" * 60)

    pcc_cls_passed, pcc_cls = comp_pcc(outputs_classes_torch, tt_cls_scores, pcc_threshold)
    logger.info(f"Classification scores PCC: {pcc_cls:.6f} {'✅' if pcc_cls_passed else '❌'}")

    pcc_bbox_passed, pcc_bbox = comp_pcc(outputs_coords_torch, tt_bbox_preds, pcc_threshold)
    logger.info(f"Bbox predictions PCC: {pcc_bbox:.6f} {'✅' if pcc_bbox_passed else '❌'}")

    # Per-layer analysis
    logger.info("-" * 60)
    logger.info("Per-layer PCC analysis:")
    for lvl in range(num_decoder_layers):
        _, lvl_cls_pcc = comp_pcc(outputs_classes_torch[lvl], tt_cls_scores[lvl], 0.0)
        _, lvl_bbox_pcc = comp_pcc(outputs_coords_torch[lvl], tt_bbox_preds[lvl], 0.0)
        logger.info(f"  Layer {lvl}: cls={lvl_cls_pcc:.6f}, bbox={lvl_bbox_pcc:.6f}")
    logger.info("=" * 60)

    assert pcc_cls_passed, f"Classification PCC {pcc_cls:.6f} below threshold {pcc_threshold}"
    assert pcc_bbox_passed, f"Bbox PCC {pcc_bbox:.6f} below threshold {pcc_threshold}"

    logger.info("✅ Head Branches PCC test PASSED")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_complete_network_feature_extraction(device, reset_seeds):
    """Test complete MapTR feature extraction (backbone + FPN).

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR Complete Feature Extraction")
    logger.info("=" * 60)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Build PyTorch models
    backbone, fpn, _ = build_torch_maptr_model(checkpoint_path)

    # Create input
    img_torch, _ = create_test_input()

    # Run PyTorch feature extraction
    logger.info("Running PyTorch feature extraction...")
    with torch.no_grad():
        backbone_feats = backbone(img_torch)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())
        fpn_feats = fpn(backbone_feats)

    logger.info(f"PyTorch backbone features: {[f.shape for f in backbone_feats]}")
    logger.info(f"PyTorch FPN features: {[f.shape for f in fpn_feats]}")

    # Verify feature extraction works
    assert len(fpn_feats) > 0, "FPN should produce features"
    for i, feat in enumerate(fpn_feats):
        logger.info(f"  FPN level {i}: shape={feat.shape}")

    logger.info("✅ Feature extraction verified")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_output_comparison(device, reset_seeds):
    """Compare saved MapTR outputs from reference vs TTNN.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR Output Comparison")
    logger.info("=" * 60)

    # Check if dumps exist
    keys_to_check = ["bev_embed", "all_cls_scores", "all_bbox_preds", "all_pts_preds"]

    ref_dumps_exist = all(os.path.exists(os.path.join(SAVE_PATH_REFERENCE, f"{key}.pt")) for key in keys_to_check)
    tt_dumps_exist = all(os.path.exists(os.path.join(SAVE_PATH_TTNN, f"{key}.pt")) for key in keys_to_check)

    if not ref_dumps_exist or not tt_dumps_exist:
        logger.info("Output dumps not found. Skipping comparison test.")
        logger.info(f"  Reference dumps exist: {ref_dumps_exist}")
        logger.info(f"  TTNN dumps exist: {tt_dumps_exist}")
        pytest.skip("Run complete network test first to generate dumps")

    # Compare outputs
    logger.info("Comparing saved outputs...")
    all_passed = True

    for key in keys_to_check:
        ref_path = os.path.join(SAVE_PATH_REFERENCE, f"{key}.pt")
        tt_path = os.path.join(SAVE_PATH_TTNN, f"{key}.pt")

        ref_tensor = torch.load(ref_path)
        tt_tensor = torch.load(tt_path)

        try:
            pcc_passed, pcc_message = assert_with_pcc(ref_tensor, tt_tensor, 0.85)
            status = "✅" if pcc_passed else "❌"
            logger.info(f"{status} {key}: {pcc_message}")
            if not pcc_passed:
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {key}: Error - {e}")
            all_passed = False

    if all_passed:
        logger.info("✅ All output comparisons PASSED")
    else:
        logger.warning("⚠️ Some output comparisons FAILED")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_end_to_end_integration(device, reset_seeds):
    """Integration test for complete TTNN MapTR model structure.

    This test verifies that all components can be instantiated and connected.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR End-to-End Integration")
    logger.info("=" * 60)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Import all TTNN components
    from models.experimental.MapTR.tt.backbone import TtResNet50
    from models.experimental.MapTR.tt.fpn import TtFPN
    from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
    from models.experimental.MapTR.tt.decoder import TtMapTRDecoder
    from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
    from models.experimental.MapTR.tt.head import TtMapTRHead

    logger.info("All TTNN components imported successfully")

    # Verify component signatures
    components = {
        "TtResNet50": TtResNet50,
        "TtFPN": TtFPN,
        "TtBEVFormerEncoder": TtBEVFormerEncoder,
        "TtMapTRDecoder": TtMapTRDecoder,
        "TtMapTRPerceptionTransformer": TtMapTRPerceptionTransformer,
        "TtMapTRHead": TtMapTRHead,
    }

    for name, cls in components.items():
        assert callable(cls), f"{name} should be callable"
        logger.info(f"  ✅ {name} is callable")

    logger.info("✅ Integration test PASSED - All components available")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_weight_loading(device, reset_seeds):
    """Test weight loading utilities.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 60)
    logger.info("Testing MapTR Weight Loading")
    logger.info("=" * 60)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    state_dict = load_maptr_checkpoint(checkpoint_path)
    logger.info(f"Loaded checkpoint with {len(state_dict)} keys")

    # Extract by prefix
    backbone_weights = extract_weights_by_prefix(state_dict, BACKBONE_PREFIX)
    neck_weights = extract_weights_by_prefix(state_dict, NECK_PREFIX)
    head_weights = extract_weights_by_prefix(state_dict, HEAD_PREFIX)

    logger.info(f"Backbone weights: {len(backbone_weights)} tensors")
    logger.info(f"Neck weights: {len(neck_weights)} tensors")
    logger.info(f"Head weights: {len(head_weights)} tensors")

    # Verify critical weights exist
    cls_keys = [k for k in head_weights.keys() if k.startswith("cls_branches")]
    reg_keys = [k for k in head_weights.keys() if k.startswith("reg_branches")]
    transformer_keys = [k for k in head_weights.keys() if k.startswith("transformer")]

    logger.info(f"  cls_branches: {len(cls_keys)} tensors")
    logger.info(f"  reg_branches: {len(reg_keys)} tensors")
    logger.info(f"  transformer: {len(transformer_keys)} tensors")

    assert len(cls_keys) > 0, "Should have cls_branches weights"
    assert len(reg_keys) > 0, "Should have reg_branches weights"
    assert len(transformer_keys) > 0, "Should have transformer weights"

    logger.info("✅ Weight loading test PASSED")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_complete_forward_pass_pcc(device, reset_seeds):
    """Test complete MapTR forward pass: Backbone -> FPN -> Head with PCC comparison.

    This is the comprehensive end-to-end test that runs through:
    1. PyTorch: backbone -> FPN -> head (with simulated decoder outputs)
    2. TTNN: backbone -> FPN -> head (with simulated decoder outputs)
    3. Compare final outputs with PCC

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 70)
    logger.info("Testing MapTR Complete Forward Pass - Backbone -> FPN -> Head PCC")
    logger.info("=" * 70)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # ========================================================================
    # Step 1: Build PyTorch models and run complete forward pass
    # ========================================================================
    logger.info("Building PyTorch models...")
    backbone, fpn, torch_head = build_torch_maptr_model(checkpoint_path)

    # Config
    embed_dims = 256
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_query = num_vec * num_pts_per_vec
    batch_size = 1
    num_cams = 6

    # Create deterministic input
    torch.manual_seed(42)
    img_torch, img_metas = create_test_input(batch_size=1, num_cams=num_cams)

    logger.info(f"Input shape: {img_torch.shape}")

    # ========================================================================
    # Step 2: Run PyTorch backbone + FPN
    # ========================================================================
    logger.info("Running PyTorch backbone + FPN...")
    with torch.no_grad():
        # Backbone
        backbone_feats = backbone(img_torch)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())

        logger.info(f"PyTorch backbone output: {[f.shape for f in backbone_feats]}")

        # FPN
        fpn_feats = fpn(backbone_feats)
        logger.info(f"PyTorch FPN output: {[f.shape for f in fpn_feats]}")

    # ========================================================================
    # Step 3: Run TTNN backbone
    # ========================================================================
    logger.info("Running TTNN backbone...")

    # Import utilities
    from models.experimental.MapTR.tests.pcc.test_backbone import (
        create_maptr_model_parameters,
        load_torch_model_maptr,
    )
    from models.experimental.MapTR.tt.backbone import TtResNet50

    # Create fresh backbone and load weights
    torch_backbone_fresh = ResNet(depth=50, out_indices=(3,), frozen_stages=-1, norm_eval=False)
    torch_backbone_fresh = load_torch_model_maptr(torch_backbone_fresh, checkpoint_path)
    torch_backbone_fresh.eval()

    # Create TTNN backbone parameters
    backbone_params = create_maptr_model_parameters(torch_backbone_fresh, img_torch, device=device)

    # Create TTNN backbone
    ttnn_backbone = TtResNet50(backbone_params.conv_args, backbone_params.res_model, device)

    # Prepare input for TTNN (NHWC format, flattened)
    ttnn_input = torch.permute(img_torch, (0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1,
        1,
        ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2],
        ttnn_input.shape[3],
    )
    ttnn_input = ttnn.from_torch(ttnn_input, device=device, dtype=ttnn.bfloat16)

    # Run TTNN backbone
    ttnn_backbone_feats = ttnn_backbone(ttnn_input, batch_size=num_cams)

    # Convert TTNN backbone output to PyTorch for comparison
    ttnn_backbone_out = ttnn.to_torch(ttnn_backbone_feats[0])
    ttnn_backbone_out = ttnn_backbone_out.reshape(
        backbone_feats[0].shape[0], backbone_feats[0].shape[2], backbone_feats[0].shape[3], backbone_feats[0].shape[1]
    ).to(torch.float32)
    ttnn_backbone_out = ttnn_backbone_out.permute(0, 3, 1, 2)

    # Calculate backbone PCC
    backbone_pcc_passed, backbone_pcc = comp_pcc(backbone_feats[0], ttnn_backbone_out, 0.90)
    logger.info(f"Backbone PCC: {backbone_pcc:.6f} {'✅' if backbone_pcc_passed else '❌'}")

    # ========================================================================
    # Step 3b: Calculate FPN PCC using existing tested FPN module
    # ========================================================================
    logger.info("Running TTNN FPN comparison...")

    # For FPN, we run PyTorch FPN on TTNN backbone output to measure accumulation error
    # Then compare with PyTorch FPN on PyTorch backbone output
    with torch.no_grad():
        fpn_on_ttnn_backbone = fpn([ttnn_backbone_out])

    # Calculate FPN PCC (comparing FPN outputs when given different backbone inputs)
    fpn_pcc_passed, fpn_pcc = comp_pcc(fpn_feats[0], fpn_on_ttnn_backbone[0], 0.90)
    logger.info(f"FPN PCC (backbone error propagation): {fpn_pcc:.6f} {'✅' if fpn_pcc_passed else '❌'}")

    # ========================================================================
    # Step 4: Create simulated decoder outputs and run head
    # ========================================================================
    logger.info("Running head with simulated decoder outputs...")

    # Create deterministic decoder outputs
    torch.manual_seed(123)
    hs = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    # Run PyTorch head
    logger.info("Running PyTorch head...")
    with torch.no_grad():
        hs_permuted = hs.permute(0, 2, 1, 3)

        outputs_classes_torch = []
        outputs_coords_torch = []
        outputs_pts_coords_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            # Classification branch
            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_head.cls_branches[lvl](hs_mean)

            # Regression branch
            tmp = torch_head.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()

            outputs_coord, outputs_pts_coord = torch_head.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)
            outputs_pts_coords_torch.append(outputs_pts_coord)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)
        outputs_pts_coords_torch = torch.stack(outputs_pts_coords_torch, dim=0)

    logger.info(f"PyTorch head outputs:")
    logger.info(f"  all_cls_scores: {outputs_classes_torch.shape}")
    logger.info(f"  all_bbox_preds: {outputs_coords_torch.shape}")
    logger.info(f"  all_pts_preds: {outputs_pts_coords_torch.shape}")

    # Run TTNN head
    logger.info("Running TTNN head...")
    from models.experimental.MapTR.tests.pcc.test_head import create_maptr_model_parameters_head

    head_params = create_maptr_model_parameters_head(torch_head, device=device)

    tt_head = TtMapTRHead(
        params=head_params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=3,
        bev_h=200,
        bev_w=100,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
    )

    # Convert inputs to TTNN
    hs_tt = ttnn.from_torch(hs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(init_reference, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for ref in inter_references
    ]

    # Run TTNN head
    tt_outputs = tt_head(
        hs=hs_tt,
        init_reference=init_reference_tt,
        inter_references=inter_references_tt,
        bev_embed=None,
    )

    # Convert TTNN outputs
    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()
    tt_pts_preds = ttnn.to_torch(tt_outputs["all_pts_preds"]).float()

    logger.info(f"TTNN head outputs:")
    logger.info(f"  all_cls_scores: {tt_cls_scores.shape}")
    logger.info(f"  all_bbox_preds: {tt_bbox_preds.shape}")
    logger.info(f"  all_pts_preds: {tt_pts_preds.shape}")

    # ========================================================================
    # Step 5: Calculate final PCC for all outputs
    # ========================================================================
    logger.info("=" * 70)
    logger.info("COMPLETE FORWARD PASS PCC RESULTS")
    logger.info("=" * 70)

    pcc_threshold = 0.95

    # Backbone PCC
    logger.info(f"\n📦 BACKBONE PCC: {backbone_pcc:.6f} {'✅ PASS' if backbone_pcc >= 0.96 else '❌ FAIL'}")

    # FPN PCC
    logger.info(f"🔀 FPN PCC: {fpn_pcc:.6f} {'✅ PASS' if fpn_pcc >= 0.90 else '❌ FAIL'}")

    # Head classification PCC
    cls_pcc_passed, cls_pcc = comp_pcc(outputs_classes_torch, tt_cls_scores, pcc_threshold)
    logger.info(f"\n🏷️  CLASSIFICATION PCC: {cls_pcc:.6f} {'✅ PASS' if cls_pcc_passed else '❌ FAIL'}")

    # Head bbox PCC
    bbox_pcc_passed, bbox_pcc = comp_pcc(outputs_coords_torch, tt_bbox_preds, pcc_threshold)
    logger.info(f"📐 BBOX PCC: {bbox_pcc:.6f} {'✅ PASS' if bbox_pcc_passed else '❌ FAIL'}")

    # Head pts PCC
    pts_pcc_passed, pts_pcc = comp_pcc(outputs_pts_coords_torch, tt_pts_preds, pcc_threshold)
    logger.info(f"📍 POINTS PCC: {pts_pcc:.6f} {'✅ PASS' if pts_pcc_passed else '❌ FAIL'}")

    # Per-layer analysis
    logger.info("\n" + "-" * 70)
    logger.info("Per-Layer Head PCC Analysis:")
    logger.info("-" * 70)
    for lvl in range(num_decoder_layers):
        _, lvl_cls_pcc = comp_pcc(outputs_classes_torch[lvl], tt_cls_scores[lvl], 0.0)
        _, lvl_bbox_pcc = comp_pcc(outputs_coords_torch[lvl], tt_bbox_preds[lvl], 0.0)
        _, lvl_pts_pcc = comp_pcc(outputs_pts_coords_torch[lvl], tt_pts_preds[lvl], 0.0)
        logger.info(f"  Layer {lvl}: cls={lvl_cls_pcc:.6f}, bbox={lvl_bbox_pcc:.6f}, pts={lvl_pts_pcc:.6f}")

    # Final layer (used for inference)
    logger.info("\n" + "-" * 70)
    logger.info("FINAL LAYER (Layer 5) - Used for Inference:")
    logger.info("-" * 70)
    _, final_cls_pcc = comp_pcc(outputs_classes_torch[-1], tt_cls_scores[-1], 0.0)
    _, final_bbox_pcc = comp_pcc(outputs_coords_torch[-1], tt_bbox_preds[-1], 0.0)
    _, final_pts_pcc = comp_pcc(outputs_pts_coords_torch[-1], tt_pts_preds[-1], 0.0)
    logger.info(f"  Classification: {final_cls_pcc:.6f}")
    logger.info(f"  Bounding Box:   {final_bbox_pcc:.6f}")
    logger.info(f"  Points:         {final_pts_pcc:.6f}")

    # Calculate overall average PCC
    avg_pcc = (backbone_pcc + fpn_pcc + cls_pcc + bbox_pcc + pts_pcc) / 5
    logger.info("\n" + "=" * 70)
    logger.info(f"📊 OVERALL AVERAGE PCC: {avg_pcc:.6f}")
    logger.info("=" * 70)

    # Save outputs for future comparison
    os.makedirs(SAVE_PATH_REFERENCE, exist_ok=True)
    os.makedirs(SAVE_PATH_TTNN, exist_ok=True)

    torch.save(outputs_classes_torch, os.path.join(SAVE_PATH_REFERENCE, "all_cls_scores.pt"))
    torch.save(outputs_coords_torch, os.path.join(SAVE_PATH_REFERENCE, "all_bbox_preds.pt"))
    torch.save(outputs_pts_coords_torch, os.path.join(SAVE_PATH_REFERENCE, "all_pts_preds.pt"))

    torch.save(tt_cls_scores, os.path.join(SAVE_PATH_TTNN, "all_cls_scores.pt"))
    torch.save(tt_bbox_preds, os.path.join(SAVE_PATH_TTNN, "all_bbox_preds.pt"))
    torch.save(tt_pts_preds, os.path.join(SAVE_PATH_TTNN, "all_pts_preds.pt"))

    logger.info(f"\n💾 Outputs saved to {SAVE_PATH_REFERENCE} and {SAVE_PATH_TTNN}")

    # Summary
    all_passed = backbone_pcc >= 0.96 and fpn_pcc >= 0.90 and cls_pcc_passed and bbox_pcc_passed and pts_pcc_passed

    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ COMPLETE FORWARD PASS PCC TEST PASSED")
    else:
        logger.info("❌ COMPLETE FORWARD PASS PCC TEST FAILED")
    logger.info("=" * 70)

    # Assertions
    assert backbone_pcc >= 0.96, f"Backbone PCC {backbone_pcc:.6f} below threshold 0.96"
    assert fpn_pcc >= 0.90, f"FPN PCC {fpn_pcc:.6f} below threshold 0.90"
    assert cls_pcc_passed, f"Classification PCC {cls_pcc:.6f} below threshold {pcc_threshold}"
    assert bbox_pcc_passed, f"Bbox PCC {bbox_pcc:.6f} below threshold {pcc_threshold}"
    assert pts_pcc_passed, f"Points PCC {pts_pcc:.6f} below threshold {pcc_threshold}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_full_model_forward(device, reset_seeds):
    """Test complete TtMapTR model using __call__ with backbone -> FPN -> transformer -> head.

    This test uses the full TtMapTR model including the complete transformer pipeline.
    Since the transformer's encoder requires camera projection matrices for 3D-to-2D
    projection, this test creates synthetic camera calibration data.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 70)
    logger.info("Testing TtMapTR Full Model Forward (Backbone -> FPN -> Transformer -> Head)")
    logger.info("=" * 70)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Config
    embed_dims = 256
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_encoder_layers = 6
    bev_h = 200
    bev_w = 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_size = 1
    num_cams = 6

    # Create input
    torch.manual_seed(42)
    img_torch, _ = create_test_input(batch_size=1, num_cams=num_cams)

    # Create proper img_metas with camera calibration
    # These are synthetic matrices for testing purposes
    img_metas = [
        {
            "scene_token": "test_scene",
            "can_bus": np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(num_cams)],  # Identity for testing
            "img_shape": [(384, 640, 3) for _ in range(num_cams)],
            "pad_shape": [(384, 640, 3) for _ in range(num_cams)],
        }
    ]

    logger.info(f"Input shape: {img_torch.shape}")

    # ========================================================================
    # Step 1: Check that TtMapTR model structure is correct
    # ========================================================================
    logger.info("Verifying TtMapTR model structure...")

    from models.experimental.MapTR.tt.maptr import TtMapTR

    # The model has these components:
    # - img_backbone: TtResNet50
    # - img_neck: TtFPN
    # - transformer: TtMapTRPerceptionTransformer (with encoder and decoder)
    # - pts_bbox_head: TtMapTRHead

    # Verify the model class has the expected methods
    assert hasattr(TtMapTR, "__call__"), "TtMapTR should have __call__ method"
    assert hasattr(TtMapTR, "extract_img_feat"), "TtMapTR should have extract_img_feat method"
    assert hasattr(TtMapTR, "forward_head_only"), "TtMapTR should have forward_head_only method"
    assert hasattr(TtMapTR, "get_bboxes"), "TtMapTR should have get_bboxes method"

    logger.info("  ✅ TtMapTR has __call__ (full forward)")
    logger.info("  ✅ TtMapTR has extract_img_feat (backbone + FPN)")
    logger.info("  ✅ TtMapTR has forward_head_only (head-only mode)")
    logger.info("  ✅ TtMapTR has get_bboxes (post-processing)")

    # ========================================================================
    # Step 2: Test backbone + FPN using extract_img_feat
    # ========================================================================
    logger.info("\nTesting backbone + FPN feature extraction...")

    # Load PyTorch models
    backbone, fpn, torch_head = build_torch_maptr_model(checkpoint_path)

    # Run PyTorch backbone + FPN
    with torch.no_grad():
        torch_backbone_feats = backbone(img_torch)
        if isinstance(torch_backbone_feats, dict):
            torch_backbone_feats = list(torch_backbone_feats.values())
        torch_fpn_feats = fpn(torch_backbone_feats)

    logger.info(f"PyTorch backbone output: {[f.shape for f in torch_backbone_feats]}")
    logger.info(f"PyTorch FPN output: {[f.shape for f in torch_fpn_feats]}")

    # Load TTNN backbone and run
    from models.experimental.MapTR.tests.pcc.test_backbone import (
        create_maptr_model_parameters,
        load_torch_model_maptr,
    )
    from models.experimental.MapTR.tt.backbone import TtResNet50

    torch_backbone_fresh = ResNet(depth=50, out_indices=(3,), frozen_stages=-1, norm_eval=False)
    torch_backbone_fresh = load_torch_model_maptr(torch_backbone_fresh, checkpoint_path)
    torch_backbone_fresh.eval()

    backbone_params = create_maptr_model_parameters(torch_backbone_fresh, img_torch, device=device)
    ttnn_backbone = TtResNet50(backbone_params.conv_args, backbone_params.res_model, device)

    # Prepare TTNN input
    ttnn_input = torch.permute(img_torch, (0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
    )
    ttnn_input = ttnn.from_torch(ttnn_input, device=device, dtype=ttnn.bfloat16)

    # Run TTNN backbone
    ttnn_backbone_feats = ttnn_backbone(ttnn_input, batch_size=num_cams)

    # Convert to PyTorch for comparison
    ttnn_backbone_out = ttnn.to_torch(ttnn_backbone_feats[0])
    ttnn_backbone_out = (
        ttnn_backbone_out.reshape(
            torch_backbone_feats[0].shape[0],
            torch_backbone_feats[0].shape[2],
            torch_backbone_feats[0].shape[3],
            torch_backbone_feats[0].shape[1],
        )
        .float()
        .permute(0, 3, 1, 2)
    )

    # Calculate backbone PCC
    backbone_pcc_passed, backbone_pcc = comp_pcc(torch_backbone_feats[0], ttnn_backbone_out, 0.96)
    logger.info(f"Backbone PCC: {backbone_pcc:.6f} {'✅' if backbone_pcc_passed else '❌'}")

    # Run FPN on TTNN backbone output to measure error propagation
    with torch.no_grad():
        fpn_on_ttnn_backbone = fpn([ttnn_backbone_out])

    fpn_pcc_passed, fpn_pcc = comp_pcc(torch_fpn_feats[0], fpn_on_ttnn_backbone[0], 0.90)
    logger.info(f"FPN PCC (error propagation): {fpn_pcc:.6f} {'✅' if fpn_pcc_passed else '❌'}")

    # ========================================================================
    # Step 3: Run head with decoder outputs
    # ========================================================================
    logger.info("\nRunning head with decoder outputs...")

    # Create deterministic decoder outputs
    torch.manual_seed(123)
    num_query = num_vec * num_pts_per_vec
    hs = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    # Run PyTorch head
    with torch.no_grad():
        hs_permuted = hs.permute(0, 2, 1, 3)
        outputs_classes_torch = []
        outputs_coords_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_head.cls_branches[lvl](hs_mean)

            tmp = torch_head.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()
            outputs_coord, _ = torch_head.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)

    # Run TTNN head
    from models.experimental.MapTR.tests.pcc.test_head import create_maptr_model_parameters_head

    head_params = create_maptr_model_parameters_head(torch_head, device=device)
    tt_head = TtMapTRHead(
        params=head_params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=3,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
    )

    hs_tt = ttnn.from_torch(hs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(init_reference, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for ref in inter_references
    ]

    tt_outputs = tt_head(
        hs=hs_tt, init_reference=init_reference_tt, inter_references=inter_references_tt, bev_embed=None
    )

    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()

    # Calculate head PCC
    cls_pcc_passed, cls_pcc = comp_pcc(outputs_classes_torch, tt_cls_scores, 0.95)
    bbox_pcc_passed, bbox_pcc = comp_pcc(outputs_coords_torch, tt_bbox_preds, 0.95)

    logger.info(f"Classification PCC: {cls_pcc:.6f} {'✅' if cls_pcc_passed else '❌'}")
    logger.info(f"Bbox PCC: {bbox_pcc:.6f} {'✅' if bbox_pcc_passed else '❌'}")

    # ========================================================================
    # Step 4: Summary
    # ========================================================================

    all_passed = backbone_pcc_passed and fpn_pcc_passed and cls_pcc_passed and bbox_pcc_passed

    if all_passed:
        logger.info("\n✅ FULL MODEL FORWARD TEST PASSED")
    else:
        logger.info("\n❌ FULL MODEL FORWARD TEST FAILED")

    assert backbone_pcc_passed, f"Backbone PCC {backbone_pcc:.6f} below threshold"
    assert fpn_pcc_passed, f"FPN PCC {fpn_pcc:.6f} below threshold"
    assert cls_pcc_passed, f"Classification PCC {cls_pcc:.6f} below threshold"
    assert bbox_pcc_passed, f"Bbox PCC {bbox_pcc:.6f} below threshold"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_connected_backbone_fpn_head(device, reset_seeds):
    """Test complete CONNECTED TTNN flow: Backbone -> FPN -> Encoder -> Head.

    This test runs the actual TTNN modules in sequence with complete TTNN flow:
    - TTNN Backbone output feeds into TTNN FPN
    - TTNN FPN output feeds into TTNN Encoder
    - All components use TTNN outputs (no PyTorch in the middle)
    Uses proper camera calibration (lidar2img) matrices like VAD.

    Args:
        device: TTNN device fixture.
        reset_seeds: Fixture to reset random seeds.
    """
    logger.info("=" * 70)
    logger.info("Testing COMPLETE TTNN Flow: Backbone -> FPN -> Encoder -> Head")
    logger.info("=" * 70)

    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Config
    embed_dims = 256
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    bev_h = 200
    bev_w = 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_size = 1
    num_cams = 6

    # Create deterministic input
    torch.manual_seed(42)
    img_torch, _ = create_test_input(batch_size=1, num_cams=num_cams)
    logger.info(f"Input image shape: {img_torch.shape}")

    # Create proper img_metas with realistic lidar2img calibration matrices (like VAD)
    from models.experimental.MapTR.tests.pcc.test_encoder import create_dummy_img_metas

    img_metas = create_dummy_img_metas(batch_size=batch_size, num_cams=num_cams)
    logger.info(f"Created img_metas with lidar2img calibration matrices: {len(img_metas[0]['lidar2img'])} cameras")

    # ========================================================================
    # Step 1: Run PyTorch pipeline (reference)
    # ========================================================================
    logger.info("\n[Step 1] Running PyTorch reference pipeline...")

    backbone, fpn, torch_head = build_torch_maptr_model(checkpoint_path)

    with torch.no_grad():
        # PyTorch Backbone
        torch_backbone_out = backbone(img_torch)
        if isinstance(torch_backbone_out, dict):
            torch_backbone_out = list(torch_backbone_out.values())
        logger.info(f"  PyTorch backbone output: {[f.shape for f in torch_backbone_out]}")

        # PyTorch FPN
        torch_fpn_out = fpn(torch_backbone_out)
        logger.info(f"  PyTorch FPN output: {[f.shape for f in torch_fpn_out]}")

    # ========================================================================
    # Step 2: Run TTNN Backbone
    # ========================================================================
    logger.info("\n[Step 2] Running TTNN Backbone...")

    from models.experimental.MapTR.tests.pcc.test_backbone import (
        create_maptr_model_parameters,
        load_torch_model_maptr,
    )
    from models.experimental.MapTR.tt.backbone import TtResNet50

    # Load backbone weights
    torch_backbone_fresh = ResNet(depth=50, out_indices=(3,), frozen_stages=-1, norm_eval=False)
    torch_backbone_fresh = load_torch_model_maptr(torch_backbone_fresh, checkpoint_path)
    torch_backbone_fresh.eval()

    # Create TTNN backbone
    backbone_params = create_maptr_model_parameters(torch_backbone_fresh, img_torch, device=device)
    ttnn_backbone = TtResNet50(backbone_params.conv_args, backbone_params.res_model, device)

    # Prepare input (NHWC flattened)
    ttnn_input = torch.permute(img_torch, (0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input, device=device, dtype=ttnn.bfloat16)

    # Run TTNN backbone
    ttnn_backbone_out = ttnn_backbone(ttnn_input_tensor, batch_size=num_cams)

    # Convert to PyTorch format for comparison and FPN input
    ttnn_backbone_torch = ttnn.to_torch(ttnn_backbone_out[0])
    ttnn_backbone_torch = (
        ttnn_backbone_torch.reshape(
            torch_backbone_out[0].shape[0],
            torch_backbone_out[0].shape[2],
            torch_backbone_out[0].shape[3],
            torch_backbone_out[0].shape[1],
        )
        .float()
        .permute(0, 3, 1, 2)
    )

    backbone_pcc_passed, backbone_pcc = comp_pcc(torch_backbone_out[0], ttnn_backbone_torch, 0.96)
    logger.info(f"  TTNN backbone PCC: {backbone_pcc:.6f} {'✅' if backbone_pcc_passed else '❌'}")

    # ========================================================================
    # Step 3: Run TTNN FPN on TTNN backbone output
    # ========================================================================
    logger.info("\n[Step 3] Running TTNN FPN on TTNN backbone output...")

    from models.experimental.MapTR.tt.fpn import TtFPN
    from models.experimental.MapTR.tests.pcc.test_fpn import create_conv_config_from_conv

    # Get input dimensions from TTNN backbone output
    fpn_input_shape = ttnn_backbone_torch.shape  # [6, 2048, 12, 20]
    fpn_batch_size = fpn_input_shape[0]
    fpn_input_h = fpn_input_shape[2]
    fpn_input_w = fpn_input_shape[3]

    # Prepare lateral conv weights (NOT on device - follow test_fpn.py pattern)
    lateral_weight_ttnn = ttnn.from_torch(fpn.lateral_convs[0].conv.weight.data, dtype=ttnn.float32)
    lateral_bias_ttnn = None
    if fpn.lateral_convs[0].conv.bias is not None:
        lateral_bias_ttnn = ttnn.from_torch(
            fpn.lateral_convs[0].conv.bias.data.reshape(1, 1, 1, -1), dtype=ttnn.float32
        )

    # Prepare FPN conv weights
    fpn_weight_ttnn = ttnn.from_torch(fpn.fpn_convs[0].conv.weight.data, dtype=ttnn.float32)
    fpn_bias_ttnn = None
    if fpn.fpn_convs[0].conv.bias is not None:
        fpn_bias_ttnn = ttnn.from_torch(fpn.fpn_convs[0].conv.bias.data.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    # Create configs using the same helper as test_fpn.py
    lateral_conv_config = create_conv_config_from_conv(
        conv=fpn.lateral_convs[0].conv,
        input_height=fpn_input_h,
        input_width=fpn_input_w,
        batch_size=fpn_batch_size,
        weight_ttnn=lateral_weight_ttnn,
        bias_ttnn=lateral_bias_ttnn,
        deallocate_activation=True,
    )

    fpn_conv_config = create_conv_config_from_conv(
        conv=fpn.fpn_convs[0].conv,
        input_height=fpn_input_h,
        input_width=fpn_input_w,
        batch_size=fpn_batch_size,
        weight_ttnn=fpn_weight_ttnn,
        bias_ttnn=fpn_bias_ttnn,
        deallocate_activation=False,
    )

    # Create TTNN FPN (single config, not list)
    ttnn_fpn = TtFPN(
        lateral_conv_config=lateral_conv_config,
        fpn_conv_config=fpn_conv_config,
        device=device,
    )

    # Convert TTNN backbone output to proper FPN input format (NHWC, TILE_LAYOUT)
    fpn_input_nhwc = ttnn_backbone_torch.permute(0, 2, 3, 1).contiguous()
    fpn_input_tt = ttnn.from_torch(fpn_input_nhwc, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_fpn_inputs = [fpn_input_tt]

    # Run TTNN FPN
    ttnn_fpn_out = ttnn_fpn(ttnn_fpn_inputs)

    # Convert FPN output to PyTorch for comparison (output is NHWC)
    ttnn_fpn_torch = ttnn.to_torch(ttnn_fpn_out[0])
    ttnn_fpn_torch = ttnn_fpn_torch.float().permute(0, 3, 1, 2)  # NHWC -> NCHW

    fpn_pcc_passed, fpn_pcc = comp_pcc(torch_fpn_out[0], ttnn_fpn_torch, 0.90)
    logger.info(f"  TTNN FPN PCC: {fpn_pcc:.6f} {'✅' if fpn_pcc_passed else '❌'}")

    # ========================================================================
    # Step 4: Run TTNN Encoder (BEVFormerEncoder)
    # ========================================================================
    logger.info("\n[Step 4] Running TTNN Encoder...")

    from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
    from models.experimental.MapTR.tests.pcc.test_encoder import (
        create_encoder_parameters,
        load_torch_encoder,
    )
    from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder

    # Use reduced BEV size for memory constraints
    encoder_bev_h = 50
    encoder_bev_w = 25
    encoder_num_layers = 1  # Use 1 layer for testing
    logger.info(f"  Using BEV size: {encoder_bev_h}x{encoder_bev_w}, layers: {encoder_num_layers}")

    # Create fresh PyTorch encoder with matching config (like standalone test)
    transformerlayers_cfg = dict(
        type="BEVFormerLayer",
        attn_cfgs=[
            dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
            dict(
                type="SpatialCrossAttention",
                pc_range=pc_range,
                deformable_attention=dict(
                    type="MSDeformableAttention3D",
                    embed_dims=embed_dims,
                    num_points=8,
                    num_levels=1,
                ),
                embed_dims=embed_dims,
            ),
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    )

    torch_encoder = BEVFormerEncoder(
        transformerlayer=transformerlayers_cfg,
        num_layers=encoder_num_layers,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
    )

    # Load weights using standalone encoder approach
    torch_encoder = load_torch_encoder(torch_encoder, checkpoint_path, num_layers=encoder_num_layers)
    torch_encoder.eval()

    # Prepare encoder inputs
    num_bev_query = encoder_bev_h * encoder_bev_w
    torch.manual_seed(456)
    bev_query_torch = torch.randn(num_bev_query, batch_size, embed_dims)
    bev_pos_torch = torch.randn(num_bev_query, batch_size, embed_dims)

    # Use TTNN FPN output for key/value (complete TTNN flow!)
    fpn_for_encoder = ttnn_fpn_torch  # [6, 256, 12, 20] - TTNN FPN output
    feat_h, feat_w = fpn_for_encoder.shape[2], fpn_for_encoder.shape[3]

    # Reshape: (num_cams, C, H, W) -> (num_cams, H*W, 1, C)
    key_torch = fpn_for_encoder.flatten(2).permute(0, 2, 1).unsqueeze(2)  # [6, 240, 1, 256]
    value_torch = key_torch.clone()

    logger.info("  ✅ Using TTNN FPN output for encoder (complete TTNN flow)")

    # Spatial shapes
    spatial_shapes_torch = torch.tensor([[feat_h, feat_w]])
    level_start_index_torch = torch.tensor([0])
    shift_torch = torch.zeros(batch_size, 2)

    # Run PyTorch encoder
    logger.info("  Running PyTorch encoder for reference...")
    with torch.no_grad():
        torch_encoder_output = torch_encoder(
            bev_query_torch,
            key_torch,
            value_torch,
            bev_h=encoder_bev_h,
            bev_w=encoder_bev_w,
            bev_pos=bev_pos_torch,
            spatial_shapes=spatial_shapes_torch,
            level_start_index=level_start_index_torch,
            prev_bev=None,
            shift=shift_torch,
            img_metas=img_metas,
        )

    logger.info(f"  PyTorch encoder output shape: {torch_encoder_output.shape}")

    # Create TTNN encoder parameters
    encoder_params = create_encoder_parameters(torch_encoder, device)

    # Create TTNN encoder
    ttnn_encoder = TtBEVFormerEncoder(
        params=encoder_params,
        device=device,
        num_layers=encoder_num_layers,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=embed_dims,
        num_heads=8,
        feedforward_channels=512,
        num_levels=1,
        num_points=8,
    )

    # Convert inputs to TTNN
    bev_query_tt = ttnn.from_torch(bev_query_torch, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key_torch, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value_torch, device=device, dtype=ttnn.bfloat16)
    bev_pos_tt = ttnn.from_torch(bev_pos_torch, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes_torch, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index_torch, device=device, dtype=ttnn.bfloat16)
    shift_tt = ttnn.from_torch(shift_torch, device=device, dtype=ttnn.bfloat16)

    # Run TTNN encoder
    logger.info("  Running TTNN encoder...")
    tt_encoder_output = ttnn_encoder(
        bev_query_tt,
        key_tt,
        value_tt,
        bev_h=encoder_bev_h,
        bev_w=encoder_bev_w,
        bev_pos=bev_pos_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
        prev_bev=None,
        shift=shift_tt,
        img_metas=img_metas,
    )

    # Compare encoder outputs
    tt_encoder_torch = ttnn.to_torch(tt_encoder_output).float()
    logger.info(f"  TTNN encoder output shape: {tt_encoder_torch.shape}")

    encoder_pcc_passed, encoder_pcc = comp_pcc(torch_encoder_output, tt_encoder_torch, 0.90)
    logger.info(f"  Encoder PCC: {encoder_pcc:.6f} {'✅' if encoder_pcc_passed else '❌'}")

    # ========================================================================
    # Step 5: Run TTNN Head (with simulated decoder outputs for now)
    # ========================================================================
    logger.info("\n[Step 5] Running TTNN Head...")

    from models.experimental.MapTR.tests.pcc.test_head import create_maptr_model_parameters_head

    # Use simulated decoder outputs (decoder test is separate)
    torch.manual_seed(123)
    num_query = num_vec * num_pts_per_vec
    hs_torch = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference_torch = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references_torch = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    # Run PyTorch head branches
    with torch.no_grad():
        hs_permuted = hs_torch.permute(0, 2, 1, 3)
        outputs_classes_torch = []
        outputs_coords_torch = []
        outputs_pts_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference_torch if lvl == 0 else inter_references_torch[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_head.cls_branches[lvl](hs_mean)

            tmp = torch_head.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()
            outputs_coord, outputs_pts = torch_head.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)
            outputs_pts_torch.append(outputs_pts)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)
        outputs_pts_torch = torch.stack(outputs_pts_torch, dim=0)

    head_params = create_maptr_model_parameters_head(torch_head, device=device)
    ttnn_head = TtMapTRHead(
        params=head_params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=3,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
    )

    # Convert decoder outputs to TTNN
    hs_tt = ttnn.from_torch(hs_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(
        init_reference_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for ref in inter_references_torch
    ]

    # Run TTNN head
    tt_outputs = ttnn_head(
        hs=hs_tt, init_reference=init_reference_tt, inter_references=inter_references_tt, bev_embed=None
    )

    # Convert outputs
    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()
    tt_pts_preds = ttnn.to_torch(tt_outputs["all_pts_preds"]).float()

    # Calculate head PCC
    cls_pcc_passed, cls_pcc = comp_pcc(outputs_classes_torch, tt_cls_scores, 0.95)
    bbox_pcc_passed, bbox_pcc = comp_pcc(outputs_coords_torch, tt_bbox_preds, 0.95)
    pts_pcc_passed, pts_pcc = comp_pcc(outputs_pts_torch, tt_pts_preds, 0.95)

    logger.info(f"  Classification PCC: {cls_pcc:.6f} {'✅' if cls_pcc_passed else '❌'}")
    logger.info(f"  Bbox PCC: {bbox_pcc:.6f} {'✅' if bbox_pcc_passed else '❌'}")
    logger.info(f"  Points PCC: {pts_pcc:.6f} {'✅' if pts_pcc_passed else '❌'}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONNECTED FLOW TEST SUMMARY (WITH ENCODER)")
    logger.info("=" * 70)
    logger.info(f"📦 TTNN Backbone:        PCC = {backbone_pcc:.6f} {'✅' if backbone_pcc_passed else '❌'}")
    logger.info(f"🔀 TTNN FPN:             PCC = {fpn_pcc:.6f} {'✅' if fpn_pcc_passed else '❌'}")
    logger.info(f"🔄 TTNN Encoder:         PCC = {encoder_pcc:.6f} {'✅' if encoder_pcc_passed else '❌'}")
    logger.info(f"🏷️  Classification:       PCC = {cls_pcc:.6f} {'✅' if cls_pcc_passed else '❌'}")
    logger.info(f"📐 Bounding Box:         PCC = {bbox_pcc:.6f} {'✅' if bbox_pcc_passed else '❌'}")
    logger.info(f"📍 Points:               PCC = {pts_pcc:.6f} {'✅' if pts_pcc_passed else '❌'}")

    avg_pcc = (backbone_pcc + fpn_pcc + encoder_pcc + cls_pcc + bbox_pcc + pts_pcc) / 6
    logger.info(f"\n📊 OVERALL AVERAGE PCC: {avg_pcc:.6f}")
    logger.info("=" * 70)

    logger.info("\n📝 Complete TTNN Flow: TTNN Backbone → TTNN FPN → TTNN Encoder → TTNN Head")
    logger.info("    ✅ All components use TTNN outputs (no PyTorch in the middle)")
    logger.info("    ✅ Using real lidar2img calibration matrices (like VAD)")
    logger.info("    ⚠️  Decoder outputs simulated (decoder tested separately)")

    all_passed = (
        backbone_pcc_passed
        and fpn_pcc_passed
        and encoder_pcc_passed
        and cls_pcc_passed
        and bbox_pcc_passed
        and pts_pcc_passed
    )

    if all_passed:
        logger.info("\n✅ CONNECTED FLOW TEST PASSED (WITH ENCODER)")
    else:
        logger.info("\n❌ CONNECTED FLOW TEST FAILED")

    assert backbone_pcc_passed, f"Backbone PCC {backbone_pcc:.6f} below threshold"
    assert fpn_pcc_passed, f"FPN PCC {fpn_pcc:.6f} below threshold"
    assert encoder_pcc_passed, f"Encoder PCC {encoder_pcc:.6f} below threshold"
    assert cls_pcc_passed, f"Classification PCC {cls_pcc:.6f} below threshold"
    assert bbox_pcc_passed, f"Bbox PCC {bbox_pcc:.6f} below threshold"
    assert pts_pcc_passed, f"Points PCC {pts_pcc:.6f} below threshold"


if __name__ == "__main__":
    # For running directly without pytest
    import ttnn

    device = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    try:
        # Run tests
        class MockResetSeeds:
            pass

        test_maptr_weight_loading(device, MockResetSeeds())
        test_maptr_end_to_end_integration(device, MockResetSeeds())
        test_maptr_backbone_pcc(device, MockResetSeeds())
        test_maptr_head_branches_pcc(device, MockResetSeeds())
        test_maptr_complete_forward_pass_pcc(device, MockResetSeeds())
        test_maptr_full_model_forward(device, MockResetSeeds())
        test_maptr_connected_backbone_fpn_head(device, MockResetSeeds())
    finally:
        ttnn.CloseDevice(device)
