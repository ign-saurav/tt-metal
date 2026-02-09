# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import argparse
import os
import json
import sys
import torch
import numpy as np
import ttnn
import gc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.reference.bevformer_v2 import BEVFormerV2
from models.experimental.BEVFormerV2.common import load_torch_model
from models.experimental.BEVFormerV2.tt.model_preprocessing import create_bevformerv2_model_parameters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processing import (
    load_demo_data,
    prepare_demo_sample,
    EmbeddedNuScenesWrapper,
    Box,
    BoxVisibility,
    box_in_image,
)

try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None
    print(
        "Warning: pyquaternion not found. Coordinate transformations will use lidar space. "
        "Install with: pip install pyquaternion"
    )

CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

VIS_MIN_DET_SCORE = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description="BEVFormerV2 TTNN Demo - Run inference and visualize predictions")
    parser.add_argument(
        "--data-root",
        default="models/experimental/BEVFormerV2/demo/demo_data/nuscenes",
        help="data root directory",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="sample index to test (default: 0, use -1 for all)")
    parser.add_argument(
        "--out",
        default="models/experimental/BEVFormerV2/demo/outputs/results.json",
        help="output result file (JSON format)",
    )
    parser.add_argument("--score-thresh", default=0.35, type=float, help="score threshold for visualization")
    parser.add_argument(
        "--device-params",
        default='{"l1_small_size": 32768}',
        help="Device parameters as JSON string",
    )
    return parser.parse_args()


def get_color(name):
    colormap = {
        "bicycle": (255, 61, 99),
        "construction_vehicle": (255, 158, 0),
        "traffic_cone": (255, 99, 71),
        "car": (0, 0, 230),
        "truck": (255, 140, 0),
        "bus": (255, 127, 80),
        "trailer": (255, 20, 147),
        "pedestrian": (255, 255, 0),
        "motorcycle": (255, 61, 99),
        "barrier": (112, 128, 144),
    }
    for key in colormap:
        if key in name:
            return colormap[key]
    return (255, 255, 255)


def format_results_to_json(results, infos, output_path):
    nusc_annos = {}

    for sample_id, result in enumerate(results):
        if sample_id >= len(infos):
            continue

        info = infos[sample_id]
        sample_token = info.get("token", f"sample_{sample_id}")

        if "pts_bbox" not in result:
            nusc_annos[sample_token] = []
            continue

        pts_bbox = result["pts_bbox"]
        boxes_3d = pts_bbox.get("boxes_3d", None)
        scores_3d = pts_bbox.get("scores_3d", None)
        labels_3d = pts_bbox.get("labels_3d", None)

        if boxes_3d is None or scores_3d is None or labels_3d is None:
            nusc_annos[sample_token] = []
            continue

        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if isinstance(scores_3d, torch.Tensor):
            scores_3d = scores_3d.cpu().numpy()
        if isinstance(labels_3d, torch.Tensor):
            labels_3d = labels_3d.cpu().numpy()

        annos = []

        for i in range(len(boxes_3d)):
            box = boxes_3d[i]
            score = float(scores_3d[i])
            label = int(labels_3d[i])

            if label < 0 or label >= len(CLASSES):
                continue

            cx, cy, cz = box[0], box[1], box[2]
            w, l, h = box[3], box[4], box[5]
            rot = box[6]

            if len(box) > 8:
                vx, vy = box[7], box[8]
            else:
                vx, vy = 0.0, 0.0

            center_lidar = np.array([cx, cy, cz])

            try:
                if Quaternion is not None:
                    lidar2ego_rot_data = np.array(info["lidar2ego_rotation"])
                    ego2global_rot_data = np.array(info["ego2global_rotation"])

                    if lidar2ego_rot_data.shape == (4,):
                        lidar2ego_rot = Quaternion(lidar2ego_rot_data).rotation_matrix
                    elif lidar2ego_rot_data.shape == (3, 3):
                        lidar2ego_rot = lidar2ego_rot_data
                    else:
                        raise ValueError(f"Unexpected lidar2ego_rotation shape: {lidar2ego_rot_data.shape}")

                    if ego2global_rot_data.shape == (4,):
                        ego2global_rot = Quaternion(ego2global_rot_data).rotation_matrix
                    elif ego2global_rot_data.shape == (3, 3):
                        ego2global_rot = ego2global_rot_data
                    else:
                        raise ValueError(f"Unexpected ego2global_rotation shape: {ego2global_rot_data.shape}")

                    lidar2ego_trans = np.array(info["lidar2ego_translation"])
                    ego2global_trans = np.array(info["ego2global_translation"])

                    center_ego = center_lidar @ lidar2ego_rot.T + lidar2ego_trans
                    center_global = center_ego @ ego2global_rot.T + ego2global_trans

                    vel_lidar = np.array([vx, 0.0, vy])
                    vel_ego = vel_lidar @ lidar2ego_rot.T
                    vel_global = vel_ego @ ego2global_rot.T
                    velocity = [float(vel_global[0]), float(vel_global[1])]
                else:
                    center_global = center_lidar
                    velocity = [float(vx), float(vy)]
            except Exception:
                center_global = center_lidar
                velocity = [float(vx), float(vy)]

            dims = np.array([w, l, h])
            dims[[0, 1, 2]] = dims[[2, 0, 1]]

            yaw = -rot

            try:
                if Quaternion is not None:
                    q1 = Quaternion(axis=[0, 0, 1], radians=yaw)
                    q2 = Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
                    quat = q2 * q1
                    rotation = quat.elements.tolist()
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            except Exception:
                rotation = [0.0, 0.0, 0.0, 1.0]

            detection_name = CLASSES[label]

            anno = {
                "sample_token": sample_token,
                "translation": center_global.tolist(),
                "size": dims.tolist(),
                "rotation": rotation,
                "velocity": velocity,
                "detection_name": detection_name,
                "detection_score": score,
                "attribute_name": "",
            }
            annos.append(anno)

        nusc_annos[sample_token] = annos

    nusc_submissions = {
        "meta": {"use_lidar": False, "use_camera": True, "use_radar": False, "use_map": False, "use_external": False},
        "results": nusc_annos,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(nusc_submissions, f, indent=2)

    return output_path


def get_predicted_data(sample_data_token, box_vis_level, pred_anns, nusc, info):
    """
    Transform predicted boxes from global to sensor coordinates.
    Based on BEVFormer's get_predicted_data function.
    """
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record.get("modality") == "camera" or "CAM" in sd_record.get("channel", ""):
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record.get("width", 1600), sd_record.get("height", 900))
    else:
        cam_intrinsic = None
        imsize = None

    boxes = []
    for box in pred_anns:
        box_copy = Box(box.center, box.wlh, box.orientation, name=box.name, token=box.token)

        box_copy.translate(-np.array(pose_record["translation"]))
        box_copy.rotate(Quaternion(pose_record["rotation"]).inverse)

        box_copy.translate(-np.array(cs_record["translation"]))
        box_copy.rotate(Quaternion(cs_record["rotation"]).inverse)

        if cam_intrinsic is not None:
            if not box_in_image(box_copy, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

        boxes.append(box_copy)

    return data_path, boxes, cam_intrinsic


def render_sample_visualization(sample_token, pred_data, data_root, out_path, score_thresh=0.35):
    """
    Render camera views with projected 3D bounding boxes.
    Based on BEVFormer's render_sample_data function.
    """
    from processing import SAMPLE_0_INFO

    nusc = EmbeddedNuScenesWrapper(version="v1.0-mini", dataroot=data_root)
    sample = nusc.get("sample", sample_token)

    if not sample or "data" not in sample:
        print(f"Warning: Sample {sample_token} not found in embedded data")
        return

    if "results" not in pred_data or sample_token not in pred_data["results"]:
        print(f"Warning: No predictions found for sample {sample_token}")
        return

    predictions = pred_data["results"][sample_token]
    info = SAMPLE_0_INFO

    boxes_global = []
    for record in predictions:
        if record.get("detection_score", 0) > score_thresh:
            box = Box(
                record["translation"],
                record["size"],
                Quaternion(record["rotation"]),
                name=record["detection_name"],
                token="predicted",
            )
            boxes_global.append(box)

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    num_boxes_total = 0

    for idx, cam in enumerate(CAMS):
        if idx >= len(axes):
            break

        if cam not in sample["data"]:
            axes[idx].text(0.5, 0.5, f"{cam}\n(not available)", ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].axis("off")
            continue

        sample_data_token = sample["data"][cam]

        try:
            data_path, boxes_pred, camera_intrinsic = get_predicted_data(
                sample_data_token, BoxVisibility.ANY, boxes_global, nusc, info
            )

            img = Image.open(data_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"PRED: {cam}", fontsize=12, fontweight="bold")
            axes[idx].axis("off")
            axes[idx].set_xlim(0, img.width)
            axes[idx].set_ylim(img.height, 0)
            axes[idx].set_aspect("equal")

            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(axes[idx], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                num_boxes_total += 1

        except Exception as e:
            print(f"Warning: Error processing {cam}: {e}")
            import traceback

            traceback.print_exc()
            axes[idx].text(0.5, 0.5, f"{cam}\n(error)", ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].axis("off")

    plt.tight_layout()
    vis_path = f"{out_path}_camera.png"
    plt.savefig(vis_path, bbox_inches="tight", pad_inches=0, dpi=200)
    print(f"Visualization saved to: {vis_path} ({num_boxes_total} boxes drawn across all cameras)")
    plt.close()


def main():
    args = parse_args()

    print("=" * 80)
    print("TTNN BEVFormerV2 Demo")
    print("=" * 80)

    device_params = json.loads(args.device_params)
    device = ttnn.open_device(device_id=0, **device_params)

    torch_model = BEVFormerV2(
        use_grid_mask=False,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=200, bev_w=200, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    print("Loading weights using load_torch_model...")
    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=None)

    for m in torch_model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = 6
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = 6

    print("Loading demo data...")
    infos = load_demo_data(sample_idx=args.sample_idx if args.sample_idx >= 0 else 0)
    print(f"Loaded {len(infos)} samples")

    if args.sample_idx >= len(infos):
        print(f"Error: sample_idx {args.sample_idx} >= {len(infos)}")
        ttnn.close_device(device)
        return

    if args.sample_idx < 0:
        sample_indices = range(len(infos))
    else:
        sample_indices = [args.sample_idx]

    first_sample = True
    outputs = []

    for idx in sample_indices:
        info = infos[idx]
        print(f"\nProcessing sample {idx}: {info.get('token', 'unknown')}")

        imgs, img_metas = prepare_demo_sample(sample_idx=idx, data_root=args.data_root)

        if first_sample:
            print("Preprocessing parameters for TTNN...")
            if isinstance(imgs, torch.Tensor) and imgs.dim() == 5:
                B, N, C, H, W = imgs.shape
                if B == 1:
                    imgs_for_preprocessing = imgs.squeeze(0)
                else:
                    imgs_for_preprocessing = imgs.reshape(B * N, C, H, W)
            else:
                imgs_for_preprocessing = imgs
            img_list = [imgs_for_preprocessing]
            encoder_num_layers = torch_model.pts_bbox_head.transformer.encoder.num_layers
            decoder_num_layers = torch_model.pts_bbox_head.transformer.decoder.num_layers
            parameters = create_bevformerv2_model_parameters(
                torch_model,
                [
                    False,
                    img_list,
                    img_metas,
                ],
                device,
            )

            del torch_model
            gc.collect()

            print("Creating TTNN model...")
            ttnn_model = TtBevFormerV2(
                device=device,
                params=parameters,
                use_grid_mask=False,
                img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
                img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
                pts_bbox_head=dict(
                    bev_h=200,
                    bev_w=200,
                    num_query=900,
                    num_classes=10,
                    in_channels=256,
                    encoder_num_layers=encoder_num_layers,
                    decoder_num_layers=decoder_num_layers,
                ),
                video_test_mode=False,
            )
            first_sample = False

        print("Converting images to TTNN format...")
        if isinstance(imgs, torch.Tensor) and imgs.dim() == 5:
            B, N, C, H, W = imgs.shape
            if B == 1:
                imgs_torch = imgs.squeeze(0)
            else:
                imgs_torch = imgs.reshape(B * N, C, H, W)
        else:
            imgs_torch = imgs
        imgs_ttnn = ttnn.from_torch(imgs_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        imgs_ttnn = [imgs_ttnn]

        print("Running TTNN inference...")
        with torch.no_grad():
            result = ttnn_model.forward_test(img_metas=img_metas, img=imgs_ttnn)[0]

        if isinstance(result, dict):
            outputs.append(result)
        elif isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)

    print(f"\n{'='*80}")
    print(f"Inference completed. Processed {len(outputs)} samples")
    print(f"{'='*80}")

    if len(outputs) > 0 and isinstance(outputs[0], dict):
        print(f"\nResult keys: {list(outputs[0].keys())}")
        if "pts_bbox" in outputs[0]:
            pts_bbox = outputs[0]["pts_bbox"]
            if "boxes_3d" in pts_bbox:
                num_dets = len(pts_bbox["boxes_3d"]) if hasattr(pts_bbox["boxes_3d"], "__len__") else "N/A"
                print(f"Number of detections: {num_dets}")
            if "scores_3d" in pts_bbox:
                scores = pts_bbox["scores_3d"]
                if hasattr(scores, "min") and hasattr(scores, "max"):
                    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"Score mean: {scores.mean():.4f}, Score std: {scores.std():.4f}")
                    if len(scores) > 0:
                        top_scores = (
                            torch.topk(scores, min(10, len(scores)))[0]
                            if isinstance(scores, torch.Tensor)
                            else sorted(scores, reverse=True)[:10]
                        )
                        print(f"Top 10 scores: {top_scores}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        json_path = format_results_to_json(outputs, infos, args.out)
        print(f"\n✓ Results saved to JSON: {json_path}")

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    show_dir = os.path.join(demo_dir, "outputs", "visualizations")
    os.makedirs(show_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")

    bevformer_results = {"results": {}}
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            bevformer_results = json.load(f)

    for idx in sample_indices:
        info = infos[idx]
        sample_token = info.get("token", f"sample_{idx}")
        vis_out_path = os.path.join(show_dir, sample_token)

        print(f"Generating visualization for sample {idx}: {sample_token}")
        render_sample_visualization(
            sample_token, bevformer_results, args.data_root, vis_out_path, score_thresh=args.score_thresh
        )

    print(f"\n{'='*80}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*80}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
