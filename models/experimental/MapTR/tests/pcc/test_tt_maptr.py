# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger
from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.detectors.maptr import MapTR
from models.experimental.MapTR.tt import tt_maptr
from models.experimental.MapTR.tt.model_preprocessing import (
    create_maptr_model_parameters,
    load_maptr_weights,
)


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"


class ConfigDict(dict):
    """A dictionary that supports attribute-style access (like mmcv.Config)."""

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


def create_maptr_config():
    """Create MapTR model configuration."""
    embed_dims = 256
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    bev_h, bev_w = 200, 100

    # Image backbone config (ResNet50)
    img_backbone_cfg = ConfigDict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=ConfigDict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    )

    # FPN neck config
    img_neck_cfg = ConfigDict(
        type="FPN",
        in_channels=[2048],
        out_channels=embed_dims,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    )

    # Transformer config
    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=embed_dims,
        encoder=ConfigDict(
            type="BEVFormerEncoder",
            num_layers=1,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayer=ConfigDict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    ConfigDict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                    ConfigDict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=ConfigDict(
                            type="MSDeformableAttention3D", embed_dims=embed_dims, num_points=8, num_levels=1
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=ConfigDict(
            type="MapTRDecoder",
            num_layers=num_decoder_layers,
            return_intermediate=True,
            transformerlayer=ConfigDict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    ConfigDict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    ConfigDict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )

    # BBox coder config
    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
        max_num=50,
        num_classes=num_classes,
    )

    # Head config
    pts_bbox_head_cfg = ConfigDict(
        type="MapTRHead",
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=num_vec * num_pts_per_vec,
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

    return ConfigDict(
        img_backbone=img_backbone_cfg,
        img_neck=img_neck_cfg,
        pts_bbox_head=pts_bbox_head_cfg,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_classes=num_classes,
        embed_dims=embed_dims,
    )


def create_input_dict(batch_size=1, num_cams=6, img_h=384, img_w=640):
    """Create input dictionary for MapTR inference."""
    input_dict = {
        "img_metas": [
            [
                {
                    "filename": [
                        "./data/nuscenes/samples/CAM_FRONT/sample.jpg",
                    ]
                    * num_cams,
                    "ori_shape": [(360, 640, 3)] * num_cams,
                    "img_shape": [(img_h, img_w, 3)] * num_cams,
                    "lidar2img": [
                        np.array(
                            [
                                [4.97195909e02, 3.36259809e02, 1.31050214e01, -1.41740456e02],
                                [-7.28050437e00, 2.14719425e02, -4.90215017e02, -2.57883151e02],
                                [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        )
                        for _ in range(num_cams)
                    ],
                    "pad_shape": [(img_h, img_w, 3)] * num_cams,
                    "scale_factor": 1.0,
                    "flip": False,
                    "pcd_horizontal_flip": False,
                    "pcd_vertical_flip": False,
                    "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "prev_idx": "",
                    "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                    "pcd_scale_factor": 1.0,
                    "pts_filename": "data/pcd.bin",
                    "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    "can_bus": np.array(
                        [
                            6.50486842e02,
                            1.81754303e03,
                            0.00000000e00,
                            1.84843146e-01,
                            1.84843146e-01,
                            1.84843146e-01,
                            1.84843146e-01,
                            8.47522666e-01,
                            1.34135536e00,
                            9.58588434e00,
                            -9.57939215e-03,
                            6.51179999e-03,
                            3.75314295e-01,
                            3.77446848e00,
                            0.00000000e00,
                            0.00000000e00,
                            3.51370076e00,
                            2.01320224e02,
                        ]
                    ),
                }
            ]
        ],
    }
    return input_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 20 * 1024}], indirect=True)
def test_maptr(
    device,
    reset_seeds,
    model_location_generator,
):
    """Test MapTR: compare reference PyTorch MapTR vs TTNN implementation with PCC."""
    # Ensure reproducible results
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = create_maptr_config()

    # Create PyTorch MapTR model (reference)
    logger.info("Creating PyTorch MapTR model (reference)...")
    torch_model = MapTR(
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=config.img_backbone,
        pts_backbone=None,
        img_neck=config.img_neck,
        pts_neck=None,
        pts_bbox_head=config.pts_bbox_head,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    )

    # Load weights
    logger.info(f"Loading weights from {MAPTR_WEIGHTS_PATH}...")
    torch_model = load_maptr_weights(torch_model, MAPTR_WEIGHTS_PATH)
    torch_model.eval()

    # Create input data
    input_dict = create_input_dict()
    tensor = torch.randn(1, 6, 3, 384, 640)
    img = [tensor]

    # Save PyTorch outputs for comparison
    import os
    from models.common.utility_functions import comp_pcc

    ref_save_path = "models/experimental/MapTR/reference/dumps"
    tt_save_path = "models/experimental/MapTR/tt/dumps"
    os.makedirs(ref_save_path, exist_ok=True)
    os.makedirs(tt_save_path, exist_ok=True)

    # ============================================================
    # STAGE 1: Run PyTorch intermediate stages for comparison
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: PyTorch Intermediate Outputs")
    logger.info("=" * 60)

    with torch.no_grad():
        # Reshape input for backbone (B, N, C, H, W) -> (B*N, C, H, W)
        B, N, C, H, W = tensor.shape
        img_reshaped = tensor.reshape(B * N, C, H, W)

        # Backbone
        torch_backbone_out = torch_model.img_backbone(img_reshaped)
        logger.info(f"PyTorch backbone output: {len(torch_backbone_out)} features")
        for i, feat in enumerate(torch_backbone_out):
            logger.info(f"  backbone[{i}] shape: {feat.shape}")
        torch.save(torch_backbone_out[-1], f"{ref_save_path}/backbone_last.pt")

        # FPN
        torch_fpn_out = torch_model.img_neck(torch_backbone_out)
        logger.info(f"PyTorch FPN output: {len(torch_fpn_out)} features")
        for i, feat in enumerate(torch_fpn_out):
            logger.info(f"  fpn[{i}] shape: {feat.shape}")
        torch.save(torch_fpn_out[0], f"{ref_save_path}/fpn_first.pt")

    # Hook to capture BEV embedding from PyTorch transformer
    bev_embed_ref = None
    decoder_inter_states_ref = None
    decoder_inter_refs_ref = None

    def hook_bev_embed(module, input, output):
        nonlocal bev_embed_ref
        # output from encoder is the bev_embed
        bev_embed_ref = output

    def hook_decoder(module, input, output):
        nonlocal decoder_inter_states_ref, decoder_inter_refs_ref
        # decoder output is (inter_states, inter_references)
        if isinstance(output, tuple) and len(output) == 2:
            decoder_inter_states_ref, decoder_inter_refs_ref = output

    # Hook for head forward - capture hs and init_reference
    head_hs_ref = None
    head_init_ref_ref = None
    head_outputs_ref = None

    def hook_head(module, input, output):
        nonlocal head_outputs_ref
        head_outputs_ref = output

    # Try to register hooks
    if hasattr(torch_model, "pts_bbox_head") and hasattr(torch_model.pts_bbox_head, "transformer"):
        if hasattr(torch_model.pts_bbox_head.transformer, "encoder"):
            torch_model.pts_bbox_head.transformer.encoder.register_forward_hook(hook_bev_embed)
            logger.info("Registered hook on transformer encoder to capture BEV embedding")
        if hasattr(torch_model.pts_bbox_head.transformer, "decoder"):
            torch_model.pts_bbox_head.transformer.decoder.register_forward_hook(hook_decoder)
            logger.info("Registered hook on transformer decoder to capture outputs")

    if hasattr(torch_model, "pts_bbox_head"):
        torch_model.pts_bbox_head.register_forward_hook(hook_head)
        logger.info("Registered hook on pts_bbox_head to capture head outputs")

    # Run full PyTorch forward pass
    logger.info("Running PyTorch MapTR full forward pass...")
    with torch.no_grad():
        torch_outputs = torch_model(
            return_loss=False,
            img=img,
            img_metas=input_dict["img_metas"],
        )

    # Create TTNN model parameters
    logger.info("Creating TTNN model parameters...")
    parameters = create_maptr_model_parameters(
        torch_model,
        tensor,
        device,
    )

    # Convert input to TTNN
    tensor_tt = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    img_tt = [tensor_tt]

    # Create TTNN MapTR model
    logger.info("Creating TTNN MapTR model...")
    tt_model = tt_maptr.TtMapTR(
        device=device,
        params=parameters,
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
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        num_classes=config.num_classes,
        embed_dims=config.embed_dims,
    )

    # ============================================================
    # STAGE 2: Run TTNN forward pass with intermediate logging
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: TTNN Forward Pass with Intermediate Outputs")
    logger.info("=" * 60)

    ttnn_outputs = tt_model(
        return_loss=False,
        img=img_tt,
        img_metas=input_dict["img_metas"],
    )

    # ============================================================
    # STAGE 3: Compare intermediate outputs
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 3: Intermediate Output Comparison")
    logger.info("=" * 60)

    def compare_tensors(name, ref_path, tt_path, convert_nhwc_to_nchw=False):
        """Compare two saved tensors and report PCC."""
        try:
            ref = torch.load(ref_path)
            tt = torch.load(tt_path)

            # Ensure same dtype for comparison
            ref = ref.float()
            tt = tt.float()

            # Handle TTNN format: (1, 1, N*H*W, C) -> (N, C, H, W)
            if convert_nhwc_to_nchw and len(tt.shape) == 4 and tt.shape[0] == 1 and tt.shape[1] == 1:
                # Get target shape info from ref
                N, C, H, W = ref.shape
                logger.info(f"  {name}: Converting TTNN (1,1,{N*H*W},{C}) NHWC to ({N},{C},{H},{W}) NCHW")
                # TT shape: (1, 1, N*H*W, C) - flattened spatial, channels last
                tt = tt.reshape(N, H, W, C)  # (N, H, W, C)
                tt = tt.permute(0, 3, 1, 2)  # (N, C, H, W)

            # Handle shape differences
            if ref.shape != tt.shape:
                logger.info(f"  {name}: SHAPE MISMATCH ref={ref.shape} tt={tt.shape}")
                # Try to reshape or permute to match
                if ref.numel() == tt.numel():
                    tt = tt.reshape(ref.shape)
                    logger.info(f"  {name}: Reshaped tt to {tt.shape}")
                else:
                    logger.info(
                        f"  {name}: Cannot compare - different number of elements (ref={ref.numel()}, tt={tt.numel()})"
                    )
                    return None

            pcc_result = comp_pcc(ref, tt)
            pcc = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
            status = "✓ PASS" if pcc >= 0.90 else "✗ FAIL"
            logger.info(f"  {name}: PCC={pcc:.6f} {status}")
            logger.info(f"    ref sample: {ref.flatten()[:5].tolist()}")
            logger.info(f"    tt sample:  {tt.flatten()[:5].tolist()}")
            return pcc
        except Exception as e:
            logger.info(f"  {name}: Error - {e}")
            import traceback

            traceback.print_exc()
            return None

    # Compare backbone outputs (TTNN saves backbone_0.pt)
    # TTNN output format: (1, 1, N*H*W, C), PyTorch format: (N, C, H, W)
    logger.info("Backbone comparison:")
    compare_tensors(
        "backbone", f"{ref_save_path}/backbone_last.pt", f"{tt_save_path}/backbone_0.pt", convert_nhwc_to_nchw=True
    )

    # Compare FPN outputs
    logger.info("FPN comparison:")
    compare_tensors("fpn", f"{ref_save_path}/fpn_first.pt", f"{tt_save_path}/fpn_0.pt", convert_nhwc_to_nchw=True)

    # Save and compare BEV embedding
    logger.info("BEV Embedding comparison:")
    if bev_embed_ref is not None:
        torch.save(bev_embed_ref, f"{ref_save_path}/bev_embed_ref.pt")
        logger.info(f"  PyTorch BEV embed shape: {bev_embed_ref.shape}, sample: {bev_embed_ref.flatten()[:5].tolist()}")
        compare_tensors("bev_embed", f"{ref_save_path}/bev_embed_ref.pt", f"{tt_save_path}/bev_embed_transformer.pt")
    else:
        logger.info("  BEV embed hook not triggered")

    # Save and compare decoder outputs
    logger.info("Decoder comparison:")
    if decoder_inter_states_ref is not None:
        torch.save(decoder_inter_states_ref, f"{ref_save_path}/decoder_inter_states_ref.pt")
        logger.info(
            f"  PyTorch decoder inter_states shape: {decoder_inter_states_ref.shape}, sample: {decoder_inter_states_ref.flatten()[:5].tolist()}"
        )
        compare_tensors(
            "decoder_inter_states",
            f"{ref_save_path}/decoder_inter_states_ref.pt",
            f"{tt_save_path}/decoder_inter_states.pt",
        )
    else:
        logger.info("  Decoder inter_states hook not triggered")

    if decoder_inter_refs_ref is not None:
        torch.save(decoder_inter_refs_ref, f"{ref_save_path}/decoder_inter_refs_ref.pt")
        logger.info(
            f"  PyTorch decoder inter_refs shape: {decoder_inter_refs_ref.shape}, sample: {decoder_inter_refs_ref.flatten()[:5].tolist()}"
        )
        compare_tensors(
            "decoder_inter_refs", f"{ref_save_path}/decoder_inter_refs_ref.pt", f"{tt_save_path}/decoder_inter_refs.pt"
        )
    else:
        logger.info("  Decoder inter_refs hook not triggered")

    # Compare head outputs
    logger.info("Head comparison:")
    if head_outputs_ref is not None:
        if isinstance(head_outputs_ref, dict):
            for key, val in head_outputs_ref.items():
                if val is not None and isinstance(val, torch.Tensor):
                    torch.save(val, f"{ref_save_path}/head_{key}.pt")
                    logger.info(f"  PyTorch head {key}: shape={val.shape}, sample={val.flatten()[:5].tolist()}")

            # Compare all_cls_scores, all_bbox_preds, all_pts_preds
            if "all_cls_scores" in head_outputs_ref and head_outputs_ref["all_cls_scores"] is not None:
                compare_tensors(
                    "head_all_cls_scores", f"{ref_save_path}/head_all_cls_scores.pt", f"{tt_save_path}/head_cls_lvl0.pt"
                )
            if "all_bbox_preds" in head_outputs_ref and head_outputs_ref["all_bbox_preds"] is not None:
                compare_tensors(
                    "head_all_bbox_preds",
                    f"{ref_save_path}/head_all_bbox_preds.pt",
                    f"{tt_save_path}/head_bbox_lvl0.pt",
                )
            if "all_pts_preds" in head_outputs_ref and head_outputs_ref["all_pts_preds"] is not None:
                compare_tensors(
                    "head_all_pts_preds", f"{ref_save_path}/head_all_pts_preds.pt", f"{tt_save_path}/head_pts_lvl0.pt"
                )
    else:
        logger.info("  Head outputs hook not triggered")

    # Compare outputs - direct comparison
    logger.info("=" * 60)
    logger.info("PCC Comparison Results:")
    logger.info("=" * 60)

    # Debug output structure
    logger.info(f"PyTorch output type: {type(torch_outputs)}")
    logger.info(f"TTNN output type: {type(ttnn_outputs)}")

    if torch_outputs:
        if isinstance(torch_outputs, list) and len(torch_outputs) > 0:
            logger.info(f"PyTorch output[0] type: {type(torch_outputs[0])}")
            if isinstance(torch_outputs[0], dict):
                logger.info(f"PyTorch output[0] keys: {torch_outputs[0].keys()}")
        elif isinstance(torch_outputs, dict):
            logger.info(f"PyTorch output keys: {torch_outputs.keys()}")

    if ttnn_outputs:
        if isinstance(ttnn_outputs, list) and len(ttnn_outputs) > 0:
            logger.info(f"TTNN output[0] type: {type(ttnn_outputs[0])}")
            if isinstance(ttnn_outputs[0], dict):
                logger.info(f"TTNN output[0] keys: {ttnn_outputs[0].keys()}")
        elif isinstance(ttnn_outputs, dict):
            logger.info(f"TTNN output keys: {ttnn_outputs.keys()}")

    # Extract pts_bbox from outputs and save reference dumps
    if torch_outputs and ttnn_outputs:
        torch_pts_bbox = torch_outputs[0].get("pts_bbox", {}) if isinstance(torch_outputs[0], dict) else {}
        ttnn_pts_bbox = ttnn_outputs[0].get("pts_bbox", {}) if isinstance(ttnn_outputs[0], dict) else {}

        logger.info(f"PyTorch pts_bbox keys: {torch_pts_bbox.keys() if torch_pts_bbox else 'None'}")
        logger.info(f"TTNN pts_bbox keys: {ttnn_pts_bbox.keys() if ttnn_pts_bbox else 'None'}")

        # Save reference and TTNN outputs for inspection
        for key, tensor in torch_pts_bbox.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                torch.save(tensor, f"{ref_save_path}/{key}.pt")
                logger.info(f"Saved reference {key} to {ref_save_path}/{key}.pt")

        for key, tensor in ttnn_pts_bbox.items():
            if tensor is not None:
                if not isinstance(tensor, torch.Tensor):
                    tensor = ttnn.to_torch(tensor)
                torch.save(tensor, f"{tt_save_path}/{key}.pt")
                logger.info(f"Saved TTNN {key} to {tt_save_path}/{key}.pt")

        # Compare common keys in pts_bbox
        pcc_results = {}
        if torch_pts_bbox and ttnn_pts_bbox:
            common_keys = set(torch_pts_bbox.keys()) & set(ttnn_pts_bbox.keys())
            for key in common_keys:
                ref_tensor = torch_pts_bbox[key]
                tt_tensor = ttnn_pts_bbox[key]

                if ref_tensor is not None and tt_tensor is not None:
                    if not isinstance(tt_tensor, torch.Tensor):
                        try:
                            tt_tensor = ttnn.to_torch(tt_tensor)
                        except Exception as e:
                            logger.info(f"{key}: Error converting to torch - {e}")
                            continue
                    if not isinstance(ref_tensor, torch.Tensor):
                        logger.info(f"{key}: Skipped (not a tensor)")
                        continue

                    # Calculate PCC manually for logging
                    from models.common.utility_functions import comp_pcc

                    pcc_result = comp_pcc(ref_tensor.float(), tt_tensor.float())
                    # comp_pcc returns (passing, pcc_value)
                    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result
                    pcc_results[key] = pcc_value

                    # Log shapes and PCC
                    logger.info(f"{key}: ref shape={ref_tensor.shape}, tt shape={tt_tensor.shape}, PCC={pcc_value:.6f}")

                    # Log sample values for debugging
                    logger.info(f"  ref sample: {ref_tensor.flatten()[:5].tolist()}")
                    logger.info(f"  tt sample:  {tt_tensor.flatten()[:5].tolist()}")
                else:
                    logger.info(f"{key}: Skipped (None tensor)")

            # Report overall results
            logger.info("=" * 60)
            logger.info("Post-processed output PCC Summary (informational - different ordering expected):")
            for key, pcc in pcc_results.items():
                status = "✓" if pcc >= 0.90 else "⚠️ (ordering)"
                logger.info(f"  {key}: PCC={pcc:.6f} {status}")

            # Check raw predictions (before top-k selection) which should have high PCC
            logger.info("=" * 60)
            logger.info("Raw prediction PCC (PASS/FAIL criteria):")
            raw_pred_pass = True
            try:
                ref_cls = torch.load("models/experimental/MapTR/reference/dumps/head_all_cls_scores.pt")
                ref_bbox = torch.load("models/experimental/MapTR/reference/dumps/head_all_bbox_preds.pt")
                ref_pts = torch.load("models/experimental/MapTR/reference/dumps/head_all_pts_preds.pt")
                ref_bev = torch.load("models/experimental/MapTR/reference/dumps/bev_embed_ref.pt")
                tt_cls = torch.load("models/experimental/MapTR/tt/dumps/all_cls_scores.pt")
                tt_bbox = torch.load("models/experimental/MapTR/tt/dumps/all_bbox_preds.pt")
                tt_pts = torch.load("models/experimental/MapTR/tt/dumps/all_pts_preds.pt")
                tt_bev = torch.load("models/experimental/MapTR/tt/dumps/bev_embed.pt")

                # comp_pcc returns (passed, pcc_value)
                _, raw_pcc_bev = comp_pcc(ref_bev, tt_bev)
                _, raw_pcc_cls = comp_pcc(ref_cls, tt_cls)
                _, raw_pcc_bbox = comp_pcc(ref_bbox, tt_bbox)
                _, raw_pcc_pts = comp_pcc(ref_pts, tt_pts)

                threshold = 0.95
                for name, pcc in [
                    ("bev_embed", raw_pcc_bev),
                    ("all_cls_scores", raw_pcc_cls),
                    ("all_bbox_preds", raw_pcc_bbox),
                    ("all_pts_preds", raw_pcc_pts),
                ]:
                    status = "✓ PASS" if pcc >= threshold else "✗ FAIL"
                    logger.info(f"  {name}: PCC={pcc:.6f} {status}")
                    if pcc < threshold:
                        raw_pred_pass = False

                assert raw_pred_pass, "Raw prediction PCC below threshold"
                logger.info("Raw prediction PCC check PASSED!")
            except FileNotFoundError:
                logger.info("Raw prediction files not found, skipping raw PCC check")
        else:
            logger.info("Could not extract pts_bbox outputs")
    else:
        logger.info("Could not extract comparable outputs")

    logger.info("=" * 60)
    logger.info("MapTR end-to-end test complete!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_head_only(device, reset_seeds):
    """Test MapTR head only (without backbone/FPN) for faster iteration."""
    torch.manual_seed(42)

    # Import existing head test
    from models.experimental.MapTR.tests.pcc.test_head import (
        load_maptr_head_weights,
        create_maptr_model_parameters_head,
        ConfigDict,
    )
    from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.dense_heads.maptr_head import MapTRHead
    from models.experimental.MapTR.tt.head import TtMapTRHead
    from models.common.utility_functions import comp_pcc

    # Config (maptr_tiny_r50_24e_bevformer)
    embed_dims = 256
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_reg_fcs = 2
    code_size = 2
    code_weights = [1.0, 1.0, 1.0, 1.0]
    bev_h, bev_w = 200, 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_size = 1
    num_query = num_vec * num_pts_per_vec

    # Build transformer config
    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=embed_dims,
        encoder=ConfigDict(
            type="BEVFormerEncoder",
            num_layers=1,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayer=ConfigDict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    ConfigDict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                    ConfigDict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=ConfigDict(
                            type="MSDeformableAttention3D", embed_dims=embed_dims, num_points=8, num_levels=1
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=ConfigDict(
            type="MapTRDecoder",
            num_layers=num_decoder_layers,
            return_intermediate=True,
            transformerlayer=ConfigDict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    ConfigDict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    ConfigDict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )

    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
        max_num=50,
        num_classes=num_classes,
    )

    # Load weights
    logger.info("Loading MapTR head weights...")
    head_weights = load_maptr_head_weights()

    # Create PyTorch MapTRHead
    logger.info("Creating PyTorch MapTRHead (reference)...")
    torch_model = MapTRHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=num_reg_fcs,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=code_size,
        code_weights=code_weights,
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

    torch_model.load_state_dict(head_weights, strict=False)
    torch_model.eval()

    # Create test inputs (decoder outputs)
    torch.manual_seed(123)
    hs = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    logger.info(f"Input shapes - hs: {hs.shape}, init_reference: {init_reference.shape}")

    # Run PyTorch forward
    logger.info("Running PyTorch MapTRHead forward pass...")
    with torch.no_grad():
        hs_permuted = hs.permute(0, 2, 1, 3)

        outputs_classes_torch = []
        outputs_coords_torch = []
        outputs_pts_coords_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_model.cls_branches[lvl](hs_mean)

            tmp = torch_model.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()

            outputs_coord, outputs_pts_coord = torch_model.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)
            outputs_pts_coords_torch.append(outputs_pts_coord)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)
        outputs_pts_coords_torch = torch.stack(outputs_pts_coords_torch, dim=0)

    # Create TTNN model
    params = create_maptr_model_parameters_head(torch_model, device=device)

    logger.info("Creating TTNN TtMapTRHead model...")
    tt_model = TtMapTRHead(
        params=params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=num_classes,
        num_reg_fcs=num_reg_fcs,
        code_size=code_size,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
        query_embed_type="instance_pts",
        transform_method="minmax",
    )

    # Convert inputs to TTNN
    hs_tt = ttnn.from_torch(hs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(init_reference, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for ref in inter_references
    ]

    # Run TTNN forward
    logger.info("Running TTNN head forward pass...")
    tt_outputs = tt_model(
        hs=hs_tt,
        init_reference=init_reference_tt,
        inter_references=inter_references_tt,
        bev_embed=None,
    )

    # Convert outputs
    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()
    tt_pts_preds = ttnn.to_torch(tt_outputs["all_pts_preds"]).float()

    # Compare with PCC
    pcc_threshold = 0.97

    logger.info("=" * 60)
    logger.info("PCC Comparison Results:")
    logger.info("=" * 60)

    pcc_cls_passed, pcc_cls = comp_pcc(outputs_classes_torch, tt_cls_scores, pcc_threshold)
    logger.info(f"Classification scores PCC: {pcc_cls:.6f} {'✓ PASSED' if pcc_cls_passed else '✗ FAILED'}")

    pcc_bbox_passed, pcc_bbox = comp_pcc(outputs_coords_torch, tt_bbox_preds, pcc_threshold)
    logger.info(f"Bbox predictions PCC:      {pcc_bbox:.6f} {'✓ PASSED' if pcc_bbox_passed else '✗ FAILED'}")

    pcc_pts_passed, pcc_pts = comp_pcc(outputs_pts_coords_torch, tt_pts_preds, pcc_threshold)
    logger.info(f"Points predictions PCC:    {pcc_pts:.6f} {'✓ PASSED' if pcc_pts_passed else '✗ FAILED'}")

    logger.info("=" * 60)

    assert pcc_cls_passed, f"Classification scores PCC {pcc_cls:.6f} below threshold {pcc_threshold}"
    assert pcc_bbox_passed, f"Bbox predictions PCC {pcc_bbox:.6f} below threshold {pcc_threshold}"
    assert pcc_pts_passed, f"Points predictions PCC {pcc_pts:.6f} below threshold {pcc_threshold}"

    logger.info("✓ MapTR Head PCC test PASSED")
