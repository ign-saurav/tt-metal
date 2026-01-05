# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from loguru import logger

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "..", "resources", "nuScenes")

IMG_KEYS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

SHOW_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


def parse_args():
    parser = ArgumentParser(description="BEVDepth Demo - Torch and TTNN visualization")
    parser.add_argument(
        "--mode", choices=["torch", "ttnn", "both", "precomputed"], default="both", help="Inference mode"
    )
    parser.add_argument("--output", default="bevdepth_demo_output.png", help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--show-range", type=float, default=60.0, help="Show range in meters")
    return parser.parse_args()


# Import common utilities
from models.experimental.BevDepth.common import download_bevdepth_weights, load_reference_model


def load_infos():
    import pickle

    infos_path = os.path.join(RESOURCES_DIR, "infos.pkl")
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)
    return infos


def load_images_and_mats(info):
    img_mean = np.array([123.675, 116.28, 103.53], np.float32)
    img_std = np.array([58.395, 57.12, 57.375], np.float32)

    sweep_imgs = []
    sweep_sensor2ego_mats = []
    sweep_intrin_mats = []
    sweep_ida_mats = []
    sweep_sensor2sensor_mats = []

    cam_info = info["cam_infos"]

    ida_aug_conf = {
        "H": 900,
        "W": 1600,
        "final_dim": [256, 704],
        "bot_pct_lim": [0.0, 0.0],
        "resize_lim": [0.386, 0.55],
        "rot_lim": [0.0, 0.0],
        "rand_flip": False,
    }

    H, W = ida_aug_conf["H"], ida_aug_conf["W"]
    fH, fW = ida_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(ida_aug_conf["bot_pct_lim"])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    for cam in IMG_KEYS:
        img_path = os.path.join(RESOURCES_DIR, cam_info[cam]["filename"])
        img = Image.open(img_path)
        img = img.resize(resize_dims)
        img = img.crop(crop)

        ida_mat = torch.eye(4)
        ida_mat[0, 0] = resize
        ida_mat[1, 1] = resize
        ida_mat[0, 3] = -crop[0]
        ida_mat[1, 3] = -crop[1]

        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np - img_mean) / img_std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Sensor to ego transformation (camera to vehicle frame)
        w, x, y, z = cam_info[cam]["calibrated_sensor"]["rotation"]
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(cam_info[cam]["calibrated_sensor"]["translation"])
        sensor2ego = torch.eye(4)
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, 3] = sensor2ego_tran

        # For key frame, sensor2ego_mats is just the sensor2ego transform
        sweepsensor2keyego = sensor2ego

        intrin_mat = torch.zeros((4, 4))
        intrin_mat[3, 3] = 1
        intrin_mat[:3, :3] = torch.Tensor(cam_info[cam]["calibrated_sensor"]["camera_intrinsic"])

        sweep_imgs.append(img_tensor)
        sweep_sensor2ego_mats.append(sweepsensor2keyego)
        sweep_intrin_mats.append(intrin_mat)
        sweep_ida_mats.append(ida_mat)
        sweep_sensor2sensor_mats.append(torch.eye(4))

    # Stack and reshape to [B, num_sweeps, num_cameras, ...]
    imgs = torch.stack(sweep_imgs).unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 3, H, W]
    sensor2ego_mats = torch.stack(sweep_sensor2ego_mats).unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 4, 4]
    intrin_mats = torch.stack(sweep_intrin_mats).unsqueeze(0).unsqueeze(0)
    ida_mats = torch.stack(sweep_ida_mats).unsqueeze(0).unsqueeze(0)
    sensor2sensor_mats = torch.stack(sweep_sensor2sensor_mats).unsqueeze(0).unsqueeze(0)

    # For 2-key model, duplicate sweep to have 2 sweeps
    imgs = imgs.repeat(1, 2, 1, 1, 1, 1)
    sensor2ego_mats = sensor2ego_mats.repeat(1, 2, 1, 1, 1)
    intrin_mats = intrin_mats.repeat(1, 2, 1, 1, 1)
    ida_mats = ida_mats.repeat(1, 2, 1, 1, 1)
    sensor2sensor_mats = sensor2sensor_mats.repeat(1, 2, 1, 1, 1)

    mats_dict = {
        "sensor2ego_mats": sensor2ego_mats,
        "intrin_mats": intrin_mats,
        "ida_mats": ida_mats,
        "sensor2sensor_mats": sensor2sensor_mats,
        "bda_mat": torch.eye(4).unsqueeze(0),
    }

    ego2global_rotation = np.mean([cam_info[cam]["ego_pose"]["rotation"] for cam in IMG_KEYS], 0)
    ego2global_translation = np.mean([cam_info[cam]["ego_pose"]["translation"] for cam in IMG_KEYS], 0)

    return imgs, mats_dict, ego2global_rotation, ego2global_translation


def load_lidar_points(info):
    lidar_path = info["lidar_infos"]["LIDAR_TOP"]["filename"]
    lidar_points = np.fromfile(os.path.join(RESOURCES_DIR, lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[
        ..., :4
    ]
    lidar_calibrated_sensor = info["lidar_infos"]["LIDAR_TOP"]["calibrated_sensor"]

    from nuscenes.utils.data_classes import LidarPointCloud

    pts = LidarPointCloud(lidar_points.T)
    pts.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    pts.translate(np.array(lidar_calibrated_sensor["translation"]))
    return pts.points.T


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    from nuscenes.utils.data_classes import Box

    box = Box(
        box_dict["translation"],
        box_dict["size"],
        Quaternion(box_dict["rotation"]),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    template = (
        np.array(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        )
        / 2
    )
    corners3d = np.tile(boxes3d[:, None, 3:6], [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]], [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]], [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners


# load_reference_model is now imported from test_utils


def decode_predictions(preds, class_names, score_threshold=0.3):
    boxes_list = []
    classes_list = []
    scores_list = []

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.8, 0.8, 8.0]
    out_size_factor = 1

    for task_idx, task_pred in enumerate(preds):
        if isinstance(task_pred, list):
            pred_dict = task_pred[0]
        else:
            pred_dict = task_pred

        heatmap = pred_dict["heatmap"].sigmoid()
        reg = pred_dict["reg"]
        height = pred_dict["height"]
        dim = pred_dict["dim"]
        rot = pred_dict["rot"]
        vel = pred_dict.get("vel", None)

        batch_size, num_classes, H, W = heatmap.shape

        for b in range(batch_size):
            for c in range(num_classes):
                heat = heatmap[b, c]
                mask = heat > score_threshold

                if mask.sum() == 0:
                    continue

                ys, xs = torch.where(mask)
                scores = heat[mask]

                for i in range(len(xs)):
                    x_idx = xs[i].item()
                    y_idx = ys[i].item()
                    score = scores[i].item()

                    # Decode position
                    x = (x_idx + reg[b, 0, y_idx, x_idx].item()) * voxel_size[0] * out_size_factor + pc_range[0]
                    y = (y_idx + reg[b, 1, y_idx, x_idx].item()) * voxel_size[1] * out_size_factor + pc_range[1]
                    z = height[b, 0, y_idx, x_idx].item()

                    # Decode dimensions
                    dx = dim[b, 0, y_idx, x_idx].item()
                    dy = dim[b, 1, y_idx, x_idx].item()
                    dz = dim[b, 2, y_idx, x_idx].item()

                    # Decode rotation
                    rot_sin = rot[b, 0, y_idx, x_idx].item()
                    rot_cos = rot[b, 1, y_idx, x_idx].item()
                    yaw = np.arctan2(rot_sin, rot_cos)

                    # Velocity
                    vx, vy = 0, 0
                    if vel is not None:
                        vx = vel[b, 0, y_idx, x_idx].item()
                        vy = vel[b, 1, y_idx, x_idx].item()

                    boxes_list.append([x, y, z, dx, dy, dz, yaw, vx, vy])
                    classes_list.append(class_names[task_idx][c])
                    scores_list.append(score)

    return boxes_list, classes_list, scores_list


def run_torch_inference(model, imgs, mats_dict):
    logger.info("Running Torch inference...")
    with torch.no_grad():
        preds = model.model(imgs, mats_dict)
    return preds


def prepare_ttnn_parameters(device):
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.experimental.BevDepth.tt.custom_preprocessing import (
        create_custom_mesh_preprocessor,
        extract_backbone_state_dict,
        extract_neck_state_dict,
        extract_depthnet_state_dict,
        fuse_batchnorm_into_conv,
        prepare_ttnn_parameters as prep_backbone_params,
        prepare_depthnet_parameters as prep_depthnet,
    )
    from models.experimental.BevDepth.tt.ttnn_secondfpn import prepare_secondfpn_parameters

    logger.info("Preparing TTNN parameters...")

    reference_model = load_reference_model()
    if reference_model is None:
        return None, None

    checkpoint_path = download_bevdepth_weights()

    backbone_state = extract_backbone_state_dict(checkpoint_path)
    backbone_state = fuse_batchnorm_into_conv(backbone_state)
    backbone_params = prep_backbone_params(backbone_state)

    neck_state = extract_neck_state_dict(checkpoint_path)
    neck_params = prepare_secondfpn_parameters(
        neck_state,
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2],
    )

    depthnet_state = extract_depthnet_state_dict(checkpoint_path)
    depthnet_params = prep_depthnet(
        depthnet_state,
        in_channels=512,
        mid_channels=512,
        depth_channels=112,
    )

    torch_head = reference_model.model.head
    torch_head.eval()
    head_params = preprocess_model_parameters(
        initialize_model=lambda: torch_head,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )

    params = {
        "backbone": backbone_params,
        "neck": neck_params,
        "depthnet": depthnet_params,
        "head": head_params,
    }

    return params, reference_model


def run_ttnn_inference(device, params, imgs, mats_dict):
    import ttnn
    from models.experimental.BevDepth.tt.ttnn_bevdepth_backbone import TtBaseLSSFPN
    from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations

    logger.info("Running TTNN inference...")

    # Get actual image dimensions from input
    _, _, _, _, img_h, img_w = imgs.shape
    logger.info(f"TTNN input image size: {img_h}x{img_w}")

    # LSS configuration matching BEVDepth official config (256x704)
    lss_conf = {
        "x_bound": [-51.2, 51.2, 0.8],
        "y_bound": [-51.2, 51.2, 0.8],
        "z_bound": [-5.0, 3.0, 0.2],
        "d_bound": [2.0, 58.0, 0.5],
        "final_dim": [img_h, img_w],
        "downsample_factor": 16,
        "output_channels": 80,
    }

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "batch_size": 1,
        "neck_in_channels": [256, 512, 1024, 2048],
        "neck_out_channels": [128, 128, 128, 128],
        "neck_upsample_strides": [0.25, 0.5, 1, 2],
        "depthnet_in_channels": 512,
        "depthnet_mid_channels": 512,
        "depthnet_context_channels": 80,
        "depthnet_depth_channels": 112,
        "use_torch_fallback": True,
    }

    ttnn_backbone = TtBaseLSSFPN(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    head_model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
    }
    ttnn_head = TtBEVDepthHead(
        parameters=params["head"],
        model_config=head_model_config,
        layer_optimisations=head_optimisations,
        device=device,
    )

    ttnn_bev_feature = ttnn_backbone(imgs, mats_dict, is_return_depth=False)

    ttnn_bev_input = ttnn.from_torch(
        ttnn_bev_feature.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_bev_input = ttnn.to_device(ttnn_bev_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_output = ttnn_head(ttnn_bev_input, device=device)

    # Convert TTNN output to torch format
    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    torch_preds = []

    for task_idx in range(len(ttnn_output)):
        task_dict = {}
        for key in output_keys:
            ttnn_tensor, shape = ttnn_output[task_idx][key]
            tensor_torch = ttnn.to_torch(ttnn_tensor)
            # TTNN output is [N, H, W, C] format - permute to [N, C, H, W]
            if len(tensor_torch.shape) == 4:
                # shape is (out_h, out_w) from the new builder API
                tensor_torch = tensor_torch.permute(0, 3, 1, 2).contiguous()
            task_dict[key] = tensor_torch
        torch_preds.append([task_dict])

    return torch_preds


def visualize_results(
    info,
    pts,
    pred_corners_torch,
    pred_classes_torch,
    pred_corners_ttnn,
    pred_classes_ttnn,
    gt_corners,
    output_path,
    show_range=60,
):
    cam_info = info["cam_infos"]

    if pred_corners_ttnn is not None:
        fig = plt.figure(figsize=(24, 16))
        num_rows = 4
    else:
        fig = plt.figure(figsize=(24, 8))
        num_rows = 2

    # Draw camera views with Torch predictions (top row)
    for i, k in enumerate(IMG_KEYS):
        fig_idx = i + 1 if i < 3 else i + 2
        ax = plt.subplot(num_rows, 4, fig_idx)
        ax.set_title(f"{k} (Torch)" if pred_corners_ttnn is not None else k)
        ax.axis("off")
        ax.set_xlim(0, 1600)
        ax.set_ylim(900, 0)

        img_path = os.path.join(RESOURCES_DIR, cam_info[k]["filename"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        for corners, cls in zip(pred_corners_torch, pred_classes_torch):
            cam_corners = get_cam_corners(
                corners,
                cam_info[k]["calibrated_sensor"]["translation"],
                cam_info[k]["calibrated_sensor"]["rotation"],
                cam_info[k]["calibrated_sensor"]["camera_intrinsic"],
            )
            lines = get_3d_lines(cam_corners)
            for line in lines:
                ax.plot(
                    line[0], line[1], c=plt.colormaps["tab10"](SHOW_CLASSES.index(cls) if cls in SHOW_CLASSES else 0)
                )

    # BEV for Torch
    ax_bev_torch = plt.subplot(num_rows, 4, 4)
    ax_bev_torch.set_title("BEV (Torch)")
    ax_bev_torch.axis("equal")
    ax_bev_torch.set_xlim(-40, 40)
    ax_bev_torch.set_ylim(-40, 40)

    ax_bev_torch.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap="gray")

    for corners in gt_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            ax_bev_torch.plot([-x for x in line[1]], line[0], c="r", label="ground truth")

    for corners in pred_corners_torch:
        lines = get_bev_lines(corners)
        for line in lines:
            ax_bev_torch.plot([-x for x in line[1]], line[0], c="g", label="torch prediction")

    handles, labels = ax_bev_torch.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_bev_torch.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=1)

    # If TTNN predictions available, add bottom rows
    if pred_corners_ttnn is not None:
        for i, k in enumerate(IMG_KEYS):
            fig_idx = i + 9 if i < 3 else i + 10
            ax = plt.subplot(num_rows, 4, fig_idx)
            ax.set_title(f"{k} (TTNN)")
            ax.axis("off")
            ax.set_xlim(0, 1600)
            ax.set_ylim(900, 0)

            img_path = os.path.join(RESOURCES_DIR, cam_info[k]["filename"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            for corners, cls in zip(pred_corners_ttnn, pred_classes_ttnn):
                cam_corners = get_cam_corners(
                    corners,
                    cam_info[k]["calibrated_sensor"]["translation"],
                    cam_info[k]["calibrated_sensor"]["rotation"],
                    cam_info[k]["calibrated_sensor"]["camera_intrinsic"],
                )
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    ax.plot(
                        line[0],
                        line[1],
                        c=plt.colormaps["tab10"](SHOW_CLASSES.index(cls) if cls in SHOW_CLASSES else 0),
                    )

        ax_bev_ttnn = plt.subplot(num_rows, 4, 12)
        ax_bev_ttnn.set_title("BEV (TTNN)")
        ax_bev_ttnn.axis("equal")
        ax_bev_ttnn.set_xlim(-40, 40)
        ax_bev_ttnn.set_ylim(-40, 40)

        ax_bev_ttnn.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap="gray")

        for corners in gt_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_ttnn.plot([-x for x in line[1]], line[0], c="r", label="ground truth")

        for corners in pred_corners_ttnn:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_ttnn.plot([-x for x in line[1]], line[0], c="b", label="ttnn prediction")

        handles, labels = ax_bev_ttnn.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_bev_ttnn.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=1)

    plt.tight_layout(w_pad=0, h_pad=2)
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved visualization to {output_path}")


def boxes_to_corners(boxes_list, classes_list, show_range):
    pred_corners = []
    pred_classes = []

    for box, cls in zip(boxes_list, classes_list):
        if cls not in SHOW_CLASSES:
            continue
        box_np = np.array(box[:9])
        if np.linalg.norm(box_np[:2]) <= show_range:
            corners = get_corners(box_np[None])[0]
            pred_corners.append(corners)
            pred_classes.append(cls)

    return pred_corners, pred_classes


def get_gt_corners(info, ego2global_rotation, ego2global_translation, show_range):
    gt_corners = []
    for ann in info["ann_infos"]:
        if map_name_from_general_to_detection.get(ann["category_name"], "ignore") in SHOW_CLASSES:
            box = get_ego_box(
                dict(
                    size=ann["size"],
                    rotation=ann["rotation"],
                    translation=ann["translation"],
                ),
                ego2global_rotation,
                ego2global_translation,
            )
            if np.linalg.norm(box[:2]) <= show_range:
                corners = get_corners(box[None])[0]
                gt_corners.append(corners)

    return gt_corners


def load_precomputed_results(info, ego2global_rotation, ego2global_translation, score_threshold=0.3):
    results_path = os.path.join(RESOURCES_DIR, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    sample_token = info["sample_token"]
    detections = results["results"].get(sample_token, [])

    boxes_list = []
    classes_list = []
    scores_list = []

    for det in detections:
        if det["detection_score"] < score_threshold:
            continue

        box = get_ego_box(
            dict(
                size=det["size"],
                rotation=det["rotation"],
                translation=det["translation"],
            ),
            ego2global_rotation,
            ego2global_translation,
        )
        vx = det.get("velocity", [0, 0])[0]
        vy = det.get("velocity", [0, 0])[1]
        boxes_list.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], vx, vy])
        classes_list.append(det["detection_name"])
        scores_list.append(det["detection_score"])

    return boxes_list, classes_list, scores_list


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BEVDepth Demo - Torch and TTNN Visualization")
    logger.info("=" * 60)

    # Load sample data
    infos = load_infos()
    info = infos[0]
    logger.info(f"Loaded sample with token: {info['sample_token']}")

    imgs, mats_dict, ego2global_rotation, ego2global_translation = load_images_and_mats(info)
    logger.info(f"Input images shape: {imgs.shape}")

    # Load lidar points for BEV visualization
    pts = load_lidar_points(info)
    logger.info(f"Loaded {len(pts)} lidar points")

    # Get ground truth corners
    gt_corners = get_gt_corners(info, ego2global_rotation, ego2global_translation, args.show_range)
    logger.info(f"Found {len(gt_corners)} ground truth boxes")

    # Class names for each task head
    class_names = [
        ["car"],
        ["truck", "construction_vehicle"],
        ["bus", "trailer"],
        ["barrier"],
        ["motorcycle", "bicycle"],
        ["pedestrian", "traffic_cone"],
    ]

    pred_corners_torch = []
    pred_classes_torch = []
    pred_corners_ttnn = None
    pred_classes_ttnn = None

    if args.mode == "precomputed":
        boxes_pre, classes_pre, scores_pre = load_precomputed_results(
            info, ego2global_rotation, ego2global_translation, args.threshold
        )
        pred_corners_torch, pred_classes_torch = boxes_to_corners(boxes_pre, classes_pre, args.show_range)
        logger.info(f"Precomputed: Loaded {len(pred_corners_torch)} detections")

    if args.mode in ["torch", "both"]:
        model = load_reference_model()
        if model is not None:
            torch_preds = run_torch_inference(model, imgs, mats_dict)
            boxes_torch, classes_torch, scores_torch = decode_predictions(torch_preds, class_names, args.threshold)
            pred_corners_torch, pred_classes_torch = boxes_to_corners(boxes_torch, classes_torch, args.show_range)
            logger.info(f"Torch: Detected {len(pred_corners_torch)} objects")
            for cls, score in zip(classes_torch, scores_torch):
                logger.info(f"  Torch: {cls} score={score:.4f}")

    if args.mode in ["ttnn", "both"]:
        try:
            import ttnn

            device = ttnn.open_device(device_id=0, l1_small_size=32768)
            try:
                params, _ = prepare_ttnn_parameters(device)
                if params is not None:
                    ttnn_preds = run_ttnn_inference(device, params, imgs, mats_dict)
                    boxes_ttnn, classes_ttnn, scores_ttnn = decode_predictions(ttnn_preds, class_names, args.threshold)
                    pred_corners_ttnn, pred_classes_ttnn = boxes_to_corners(boxes_ttnn, classes_ttnn, args.show_range)
                    logger.info(f"TTNN: Detected {len(pred_corners_ttnn)} objects")
                    for cls, score in zip(classes_ttnn, scores_ttnn):
                        logger.info(f"  TTNN: {cls} score={score:.4f}")
            finally:
                ttnn.close_device(device)
        except ImportError:
            logger.warning("TTNN not available, skipping TTNN inference")
        except Exception as e:
            logger.error(f"TTNN inference failed: {e}")
            import traceback

            traceback.print_exc()

    # Visualize results
    output_path = os.path.join(SCRIPT_DIR, args.output)
    visualize_results(
        info,
        pts,
        pred_corners_torch,
        pred_classes_torch,
        pred_corners_ttnn,
        pred_classes_ttnn,
        gt_corners,
        output_path,
        args.show_range,
    )

    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
