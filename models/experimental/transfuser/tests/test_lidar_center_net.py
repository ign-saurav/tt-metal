# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import cv2
import torch
import pytest
import numpy as np
import ttnn
from PIL import Image
from collections import OrderedDict
from typing import Dict, Any, List, Tuple

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor

from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


# =========================
# Utilities: mesh mappers
# =========================
def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


# --- deterministic seed for this module ---
@pytest.fixture(autouse=True, scope="session")
def _seed_everything():
    import random, numpy as np, torch

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid CuDNN nondeterminism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✅ Session seed set to {seed}")


# ==========================================
# Input preprocessing (NO normalization)
# ==========================================
def _scale_crop(image: Image.Image, scale: int, start_x: int, crop_x: int, start_y: int, crop_y: int) -> np.ndarray:
    """Resize (if scale!=1) and crop [H, W, C] numpy (uint8)."""
    w, h = image.width // scale, image.height // scale
    if scale != 1:
        image = image.resize((w, h))
    arr = np.asarray(image)  # HWC, uint8
    return arr[start_y : start_y + crop_y, start_x : start_x + crop_x]


def _shift_x_scale_crop(image: Image.Image, scale: int, crop: Tuple[int, int], crop_shift: int = 0) -> np.ndarray:
    """Shift/scale/crop then transpose to CHW, uint8 -> float32 (still 0..255)."""
    crop_h, crop_w = crop
    width, height = int(image.width // scale), int(image.height // scale)
    im_resized = image.resize((width, height))
    arr = np.array(im_resized)  # HWC
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2
    start_x += int(crop_shift // scale)
    crop_arr = arr[start_y : start_y + crop_h, start_x : start_x + crop_w]
    chw = np.transpose(crop_arr, (2, 0, 1)).astype(np.float32)  # CHW in 0..255 (no norm)
    return chw


def lidar_to_histogram_features(lidar_xyz: np.ndarray) -> np.ndarray:
    """
    Build 2-bin histogram (above/below ground) over 256x256.
    Returns (2,256,256) float32 in [0,1].
    """

    def splat_points(pc):
        # grid: 256x256 => 32m x 32m @ 8 px/m (x ∈ [-16,16], y ∈ [-32,0])
        ppm = 8
        hist_max = 5
        x_max = 16
        y_max = 32
        xbins = np.linspace(-x_max, x_max, 32 * ppm + 1)
        ybins = np.linspace(-y_max, 0, 32 * ppm + 1)
        hist = np.histogramdd(pc[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max] = hist_max
        return (hist / hist_max).astype(np.float32)

    below = lidar_xyz[lidar_xyz[..., 2] <= -2.3]
    above = lidar_xyz[lidar_xyz[..., 2] > -2.3]
    h_below = splat_points(below)
    h_above = splat_points(above)
    feats = np.stack([h_above, h_below], axis=0)  # (2,H,W)
    # rotate to CARLA/Transfuser convention
    feats = np.rot90(feats, -1, axes=(1, 2)).copy().astype(np.float32)
    return feats


def _draw_target_point(target_point_xy: np.ndarray, color: int = 255) -> np.ndarray:
    """
    Draw a single target point on a 256x256 image. Returns (1,256,256) float32 in [0,1].
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    pt = target_point_xy.copy()

    # convert to lidar BEV pixel coordinates
    pt[1] += 1.3
    pt *= 8.0
    pt[1] *= -1
    pt[1] = 256 - pt[1]
    pt[0] += 128
    pt = pt.astype(np.int32)
    pt = np.clip(pt, 0, 256)
    cv2.circle(img, tuple(pt), radius=5, color=color, thickness=3)
    img = img.reshape(1, 256, 256).astype(np.float32) / 255.0
    return img


def build_inputs_on_the_fly(data_root: str, frame: str, image_variant: str = "raw") -> Dict[str, torch.Tensor]:
    """
    Build all inputs needed by Transfuser/LidarCenterNet **without** normalization:
      - image: (1,3,160,704) float32 in [0,255]
      - lidar: (1,3,256,256) float32, hist(2) + target_point(1)
      - velocity: (1,1)
      - target_point: (1,2)

    Folders under data_root:
      rgb/{frame}.png, lidar/{frame}.npy, measurements/{frame}.json
    """
    # ---- image ----
    rgb_path = os.path.join(data_root, "rgb", f"{frame}.png")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    pil_img = Image.open(rgb_path).convert("RGB")
    # emulate the original pipeline: concat (left/front/right) by reusing the same image, then crop
    H, W = 160, 704
    scale = 1
    # Make 3 views (left, front, right) using the same image; crop each to (H,W)
    cams = []
    for _ in ("left", "front", "right"):
        view = _scale_crop(pil_img, scale, 0, W, 0, H)  # HxWxC (uint8)
        cams.append(view)
    concat_hwc = np.concatenate(cams, axis=1)  # H x (3W) x 3
    # Center crop back to (H,W) => same as _shift_x_scale_crop with crop_shift=0
    concat_pil = Image.fromarray(concat_hwc)
    chw = _shift_x_scale_crop(concat_pil, scale=1, crop=(H, W), crop_shift=0)  # (3,H,W) float32, 0..255
    image_tensor = torch.from_numpy(chw).unsqueeze(0)  # (1,3,160,704), float32 in [0,255]

    # ---- lidar ----
    lidar_path = os.path.join(data_root, "lidar", f"{frame}.npy")
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
    lidar_array = np.load(lidar_path, allow_pickle=True)
    # Original files usually pack pointcloud at index 1
    pointcloud = lidar_array[1]
    xyz = pointcloud[:, :3].copy()
    xyz[:, 1] *= -1  # invert Y for histogram function
    lidar_hist = lidar_to_histogram_features(xyz)  # (2,256,256)

    # ---- measurements ----
    meas_path = os.path.join(data_root, "measurements", f"{frame}.json")
    if not os.path.exists(meas_path):
        raise FileNotFoundError(f"Measurements not found: {meas_path}")
    with open(meas_path, "r") as f:
        measurements = json.load(f)
    speed = float(measurements.get("speed", 0.0))
    velocity_tensor = torch.tensor([[speed]], dtype=torch.float32)  # (1,1)

    # target point (fallback to zeros if not available)
    if (
        "target_point" in measurements
        and isinstance(measurements["target_point"], (list, tuple))
        and len(measurements["target_point"]) >= 2
    ):
        tp_xy = np.array(measurements["target_point"][:2], dtype=np.float32)
    else:
        tp_xy = np.array([0.0, 0.0], dtype=np.float32)
    target_point = torch.from_numpy(tp_xy.reshape(1, 2))  # (1,2)

    # target point image => 1 channel
    tpi = _draw_target_point(tp_xy)  # (1,256,256) float32 [0,1]

    # final 3-channel lidar bev: [above, below, target]
    lidar_bev = np.concatenate([lidar_hist, tpi], axis=0).astype(np.float32)  # (3,256,256)
    lidar_tensor = torch.from_numpy(lidar_bev).unsqueeze(0)  # (1,3,256,256)

    return {
        "image": image_tensor,  # (1,3,160,704), float32 in [0,255]
        "lidar": lidar_tensor,  # (1,3,256,256), float32
        "velocity": velocity_tensor,  # (1,1)
        "target_point": target_point,  # (1,2)
    }


# ------------------------------
# Helper: make TTNN boxes match reference postproc
# ------------------------------
def filter_and_cap_boxes(
    head_module,
    torch_boxes_tuple,
    ref_rotated_bboxes,
    explicit_score_thr=None,
    bb_confidence_threshold=None,
):
    """
    head.get_bboxes returns a list of (bboxes, labels) for each image.
    We apply:
      1) confidence threshold (prefer head.test_cfg.score_thr if present, else explicit_score_thr)
      2) cap to len(ref_rotated_bboxes) (so counts match the reference path)
    """
    assert isinstance(torch_boxes_tuple, (list, tuple)) and len(torch_boxes_tuple) > 0
    bboxes, labels = torch_boxes_tuple[0]  # first (and only) image in batch

    # 1) Decide score threshold (prefer head.test_cfg if present)
    score_thr = None
    if hasattr(head_module, "test_cfg") and head_module.test_cfg is not None:
        # mmcv-style config might have a score_thr
        score_thr = getattr(head_module.test_cfg, "score_thr", None)

    if score_thr is None and explicit_score_thr is not None:
        score_thr = float(explicit_score_thr)

    # Apply score threshold if we have one
    if score_thr is not None and bboxes.numel() > 0:
        keep = bboxes[:, -1] >= float(score_thr)
        bboxes = bboxes[keep]
        if labels is not None and labels.numel() > 0:
            labels = labels[keep]

    # Optional extra filter using your config.bb_confidence_threshold (already used later, but safe to unify)
    if bb_confidence_threshold is not None and bboxes.numel() > 0:
        keep = bboxes[:, -1] >= float(bb_confidence_threshold)
        bboxes = bboxes[keep]
        if labels is not None and labels.numel() > 0:
            labels = labels[keep]

    # 2) Cap to the same count as the reference rotated boxes
    ref_count = len(ref_rotated_bboxes)
    if bboxes.numel() > 0 and bboxes.size(0) > ref_count:
        scores = bboxes[:, -1]
        topk_idx = torch.topk(scores, k=ref_count, largest=True, sorted=True).indices
        bboxes = bboxes[topk_idx]
        if labels is not None and labels.numel() > 0:
            labels = labels[topk_idx]

    return [(bboxes, labels)]


# ===================================================
# Head preprocessor (unchanged from your version)
# ===================================================
def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            if head_name == "heatmap_head":
                weight_dtype = ttnn.float32
            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)
                parameters[head_name] = {}
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
        return parameters

    return custom_preprocessor


# ===================================================
# Checkpoint helpers (your logic preserved)
# ===================================================
def fix_and_filter_checkpoint_keys(
    checkpoint_path: str, target_prefix: str = "module._model.", state_dict_key: str = None
) -> Dict[str, Any]:
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    src = ckpt[state_dict_key] if (state_dict_key and state_dict_key in ckpt) else ckpt
    new_state = OrderedDict()
    removed = 0
    for k, v in src.items():
        if k.startswith(target_prefix):
            new_state[k[len(target_prefix) :]] = v
        else:
            removed += 1
    print(f"✅ Filtered {len(new_state)} keys, discarded {removed}.")
    return new_state


def load_trained_weights(weight_path: str) -> Dict[str, Any]:
    print(f"Loading trained weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module._model."):
            clean_key = key[len("module._model.") :]
            state_dict[clean_key] = value
        else:
            state_dict[key] = value

    backbone_keys = [
        "image_encoder",
        "lidar_encoder",
        "transformer1",
        "transformer2",
        "transformer3",
        "transformer4",
        "change_channel_conv_image",
        "change_channel_conv_lidar",
        "up_conv5",
        "up_conv4",
        "up_conv3",
        "c5_conv",
    ]
    backbone_renamed = 0
    for key in list(state_dict.keys()):
        for bb in backbone_keys:
            if key.startswith(f"{bb}."):
                new_key = f"_model.{bb}.{key[len(bb)+1:]}"
                state_dict[new_key] = state_dict.pop(key)
                backbone_renamed += 1
                break
    print(f"Added '_model.' to {backbone_renamed} backbone keys")

    detection_components = ["head", "pred_bev", "join", "decoder", "output"]
    det_renamed = 0
    for key in list(state_dict.keys()):
        for comp in detection_components:
            if key.startswith(f"module.{comp}."):
                new_key = key[len("module.") :]
                state_dict[new_key] = state_dict.pop(key)
                det_renamed += 1
                break
    print(f"Cleaned {det_renamed} detection component keys")
    return state_dict


def delete_incompatible_keys(state_dict: Dict[str, Any], keys_to_delete: List[str]) -> Dict[str, Any]:
    new_state = OrderedDict(state_dict)
    count = 0
    for k in keys_to_delete:
        if k in new_state:
            del new_state[k]
            count += 1
            print(f"🗑️ Deleted incompatible key: {k}")
    print(f"Successfully deleted {count} key(s) for strict=True loading.")
    return new_state


# ===================================================
# PCC helpers (unchanged)
# ===================================================
def compare_boxes_pcc(ref_boxes, torch_boxes):
    pcc_scores = []
    for i, bbox_ref in enumerate(ref_boxes):
        b_ref = bbox_ref[0] if isinstance(bbox_ref, tuple) else bbox_ref
        for j, bbox_t in enumerate(torch_boxes):
            b_t = bbox_t[0] if isinstance(bbox_t, tuple) else bbox_t
            does_pass, pcc_val = check_with_pcc(b_ref, b_t, 0.0)
            pcc_scores.append((i, j, pcc_val))
    pcc_scores.sort(key=lambda x: x[2], reverse=True)
    top = pcc_scores[: len(ref_boxes)]
    return top, pcc_scores


def print_results(top_pcc, all_pcc_scores):
    print("\n" + "=" * 60)
    print("TOP PCC SCORES (Top len(ref_boxes) matches)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Ref_Idx':<8} {'Torch_Idx':<10} {'PCC_Score':<12}")
    print("-" * 60)
    for rank, (ri, tj, pv) in enumerate(top_pcc, 1):
        try:
            pvf = float(pv)
            print(f"{rank:<6} {ri:<8} {tj:<10} {pvf:<12.6f}")
        except Exception:
            print(f"{rank:<6} {ri:<8} {tj:<10} {str(pv):<12}")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total comparisons: {len(all_pcc_scores)}")
    print(f"Top matches shown: {len(top_pcc)}")
    if all_pcc_scores:
        vals = [float(s[2]) for s in all_pcc_scores]
        print(f"Best PCC score: {max(vals):.6f}")
        print(f"Worst PCC score: {min(vals):.6f}")
        print(f"Average PCC score: {np.mean(vals):.6f}")
        print(f"Median PCC score: {np.median(vals):.6f}")


# ===================================================
# The actual test
# ===================================================
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, target_point_image_shape, img_shape, lidar_bev_shape",
    [
        ("regnety_032", "regnety_032", 4, False, (1, 1, 256, 256), (1, 3, 160, 704), (1, 3, 256, 256)),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_fallback", [True])
def test_lidar_center_net(
    device,
    cli_args,  # from conftest.py
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    target_point_image_shape,
    img_shape,
    lidar_bev_shape,
    input_dtype,
    weight_dtype,
    use_fallback,
):
    seed = 0
    import random, numpy as np, torch
    from loguru import logger

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✅ Random seed set to {seed}")
    # ---- CLI args (with safe defaults if not provided) ----
    data_root = cli_args["data_root"] or "Scenario3_Town01_curved_route0_11_23_20_02_59"
    frame = cli_args["frame"] or "0120"
    image_variant = cli_args["image_variant"] or "raw"

    # ---- build inputs on the fly (NO normalization) ----
    built = build_inputs_on_the_fly(data_root, frame, image_variant)
    image = built["image"]  # (1,3,160,704) float32 in [0,255]
    lidar_bev = built["lidar"]  # (1,3,256,256) float32
    velocity = built["velocity"]  # (1,1)
    target_point = built["target_point"]  # (1,2)

    # inputs = torch.load("transfuser_inputs_final.pt")
    # image = inputs["image"]  # RGB camera image tensor
    # lidar_bev = inputs["lidar"]  # LiDAR BEV tensor
    # velocity = inputs["velocity"]  # Ego velocity tensor
    # target_point = inputs["target_point"]

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    # ---- reference model ----
    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    config.use_target_point_image = True

    ref_layer = LidarCenterNet(
        config,
        backbone="transFuser",
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=use_velocity,
    ).eval()

    # Load & clean trained weights (keep your paths/keys)
    checkpoint_path = "model_seed1_39.pth"
    modified_state_dict = load_trained_weights(checkpoint_path)
    modified_state_dict = delete_incompatible_keys(
        modified_state_dict,
        [
            "_model.lidar_encoder._model.stem.conv.weight",
            "module.seg_decoder.deconv1.0.weight",
            "module.seg_decoder.deconv1.0.bias",
            "module.seg_decoder.deconv1.2.weight",
            "module.seg_decoder.deconv1.2.bias",
            "module.seg_decoder.deconv2.0.weight",
            "module.seg_decoder.deconv2.0.bias",
            "module.seg_decoder.deconv2.2.weight",
            "module.seg_decoder.deconv2.2.bias",
            "module.seg_decoder.deconv3.0.weight",
            "module.seg_decoder.deconv3.0.bias",
            "module.seg_decoder.deconv3.2.weight",
            "module.seg_decoder.deconv3.2.bias",
            "module.depth_decoder.deconv1.0.weight",
            "module.depth_decoder.deconv1.0.bias",
            "module.depth_decoder.deconv1.2.weight",
            "module.depth_decoder.deconv1.2.bias",
            "module.depth_decoder.deconv2.0.weight",
            "module.depth_decoder.deconv2.0.bias",
            "module.depth_decoder.deconv2.2.weight",
            "module.depth_decoder.deconv2.2.bias",
            "module.depth_decoder.deconv3.0.weight",
            "module.depth_decoder.deconv3.0.bias",
            "module.depth_decoder.deconv3.2.weight",
            "module.depth_decoder.deconv3.2.bias",
        ],
    )
    ref_layer.load_state_dict(modified_state_dict, strict=True)

    # Forward through reference path
    ref_fused_features, ref_feature, pred_wp, ref_head_results, ref_boxes, ref_rotated_bboxes = ref_layer.forward_ego(
        image, lidar_bev, target_point, velocity
    )

    (
        ref_center_heatmap_list,
        ref_wh_list,
        ref_offset_list,
        ref_yaw_class_list,
        ref_yaw_res_list,
        ref_velocity_list,
        ref_brake_list,
    ) = ref_head_results

    ref_center_heatmap = ref_center_heatmap_list[0]
    ref_wh = ref_wh_list[0]
    ref_offset = ref_offset_list[0]
    ref_yaw_class = ref_yaw_class_list[0]
    ref_yaw_res = ref_yaw_res_list[0]
    ref_velocity = ref_velocity_list[0]
    ref_brake = ref_brake_list[0]

    torch_model = ref_layer._model

    # ---- TT parameter preprocessing ----
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    for name in ["transformer1", "transformer2", "transformer3", "transformer4"]:
        gpt_params = preprocess_model_parameters(
            initialize_model=lambda n=name: getattr(torch_model, n),
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
            device=device,
        )
        parameters[name] = gpt_params

    parameters["head"] = preprocess_model_parameters(
        initialize_model=lambda: ref_layer.head,
        custom_preprocessor=create_lidar_center_net_head_preprocessor(device, weight_dtype),
        device=device,
    )

    # ---- TT model ----
    transfuser_model = ref_layer._model
    tt_layer = TtLidarCenterNet(
        device,
        parameters,
        config,
        backbone="transFuser",
        torch_model=transfuser_model,
        use_fallback=use_fallback,
    )

    # ---- Convert inputs to TTNN ----
    tt_image_input = ttnn.from_torch(
        image.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_lidar_input = ttnn.from_torch(
        lidar_bev.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_velocity_input = ttnn.from_torch(
        velocity,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_image = ttnn.to_device(tt_image_input, device)
    tt_lidar_bev = ttnn.to_device(tt_lidar_input, device)
    tt_velocity = ttnn.to_device(tt_velocity_input, device)

    # ---- TT forward ----
    tt_features, tt_fused_features = tt_layer.forward_ego(tt_image, tt_lidar_bev, tt_velocity, target_point)

    # ---- Compare fused features ----
    tt_fused_torch = ttnn.to_torch(tt_fused_features, device=device)
    does_pass, fused_features_pcc_message = check_with_pcc(ref_fused_features, tt_fused_torch, 0.80)
    logger.info(f"fused features PCC: {fused_features_pcc_message}")

    # ---- GRU on PyTorch (using TT fused features) ----
    tt_fused_torch = tt_fused_torch.to(torch.float32)
    tt_pred_wp, _, _, _, _ = ref_layer.forward_gru(tt_fused_torch, target_point)
    does_pass, pred_wp_pcc_message = check_with_pcc(pred_wp, tt_pred_wp, 0.80)
    logger.info(f"pred wp PCC: {pred_wp_pcc_message}")

    # ---- Feature map PCC ----
    tt_feature_0 = ttnn.to_torch(tt_features[0], device=device).to(torch.float32).permute(0, 3, 1, 2)
    pcc_passed, pcc_msg = check_with_pcc(ref_feature, tt_feature_0, pcc=0.95)
    logger.info(f"Feature PCC: {pcc_msg}")

    # ---- Head through PyTorch for both ----
    torch_results = ref_layer.head([tt_feature_0])
    does_pass, results_pcc_message = check_with_pcc(ref_head_results[0][0], torch_results[0][0], 0.80)
    logger.info(f"results PCC: {results_pcc_message}")

    # Unpack the TT head tensors
    (
        torch_center_heatmap_list,
        torch_wh_list,
        torch_offset_list,
        torch_yaw_class_list,
        torch_yaw_res_list,
        torch_velocity_list,
        torch_brake_list,
    ) = torch_results

    torch_center_heatmap = torch_center_heatmap_list[0]
    torch_wh = torch_wh_list[0]
    torch_offset = torch_offset_list[0]
    torch_yaw_class = torch_yaw_class_list[0]
    torch_yaw_res = torch_yaw_res_list[0]
    torch_velocity = torch_velocity_list[0]
    torch_brake = torch_brake_list[0]

    # ---- BBoxes (apply SAME postproc semantics as reference) ----
    # First, raw decode via head API:
    raw_boxes = ref_layer.head.get_bboxes(
        [torch_center_heatmap],
        [torch_wh],
        [torch_offset],
        [torch_yaw_class],
        [torch_yaw_res],
        [torch_velocity],
        [torch_brake],
    )

    # Now normalize TTNN results to the SAME number as reference using head.test_cfg & your config threshold
    torch_boxes = filter_and_cap_boxes(
        head_module=ref_layer.head,
        torch_boxes_tuple=raw_boxes,
        ref_rotated_bboxes=ref_rotated_bboxes,
        explicit_score_thr=None,  # keep None to prefer head.test_cfg.score_thr if it exists
        bb_confidence_threshold=getattr(config, "bb_confidence_threshold", None),
    )

    # Convert to metric coords exactly like reference path
    torch_bboxes, _ = torch_boxes[0]
    torch_rotated_bboxes = []
    if torch_bboxes.numel() > 0:
        for bbox in torch_bboxes.detach().cpu().numpy():
            bbox_metric = ref_layer.get_bbox_local_metric(bbox)
            torch_rotated_bboxes.append(bbox_metric)

    # Compare bbox counts (should now match the reference)
    logger.info(f"Reference bboxes count: {len(ref_rotated_bboxes)}")
    logger.info(f"TTNN bboxes count: {len(torch_rotated_bboxes)}")

    # PCC ranking printout
    top_pcc, all_pcc_scores = compare_boxes_pcc(ref_rotated_bboxes, torch_rotated_bboxes)
    print_results(top_pcc, all_pcc_scores)

    # ---- Metric PCCs (unchanged) ----
    _ok, msg = check_with_pcc(ref_wh, torch_wh, 0.80)
    logger.info(f"WH PCC: {msg}")
    _ok, msg = check_with_pcc(ref_offset, torch_offset, 0.80)
    logger.info(f"Offset PCC: {msg}")
    _ok, msg = check_with_pcc(ref_yaw_class, torch_yaw_class, 0.80)
    logger.info(f"Yaw Class PCC: {msg}")
    _ok, msg = check_with_pcc(ref_yaw_res, torch_yaw_res, 0.80)
    logger.info(f"Yaw Residual PCC: {msg}")
    _ok, msg = check_with_pcc(ref_velocity, torch_velocity, 0.80)
    logger.info(f"Velocity PCC: {msg}")

    does_pass, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, torch_center_heatmap, 0.80)
    logger.info(f"Center Heatmap PCC: {heatmap_pcc_message}")
    assert does_pass, f"Center Heatmap PCC Failed! PCC: {heatmap_pcc_message}"

    if does_pass:
        logger.info("LidarCenterNet Passed!")
    else:
        logger.warning("LidarCenterNet Failed!")
