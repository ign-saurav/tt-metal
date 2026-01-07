# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import json
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box, LidarPointCloud

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


def load_infos():
    """Load sample information from pickle file."""
    infos_path = os.path.join(RESOURCES_DIR, "infos.pkl")
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)
    return infos


def load_images_and_mats(info):
    """
    Load and preprocess camera images and transformation matrices.

    Args:
        info: Sample information dictionary containing camera info

    Returns:
        imgs: Preprocessed image tensor [B, num_sweeps, num_cameras, C, H, W]
        mats_dict: Dictionary containing transformation matrices
        ego2global_rotation: Ego to global rotation
        ego2global_translation: Ego to global translation
    """
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
    """
    Load and transform LiDAR points to ego frame.

    Args:
        info: Sample information dictionary containing LiDAR info

    Returns:
        points: LiDAR points in ego frame [N, 4]
    """
    lidar_path = info["lidar_infos"]["LIDAR_TOP"]["filename"]
    lidar_points = np.fromfile(os.path.join(RESOURCES_DIR, lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[
        ..., :4
    ]
    lidar_calibrated_sensor = info["lidar_infos"]["LIDAR_TOP"]["calibrated_sensor"]

    pts = LidarPointCloud(lidar_points.T)
    pts.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    pts.translate(np.array(lidar_calibrated_sensor["translation"]))
    return pts.points.T


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    """
    Transform a box from global frame to ego frame.

    Args:
        box_dict: Dictionary containing box translation, size, and rotation
        ego2global_rotation: Ego to global rotation quaternion
        ego2global_translation: Ego to global translation

    Returns:
        box_ego: Box in ego frame [x, y, z, dx, dy, dz, yaw, vx, vy]
    """
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
    """
    Rotate points along the z-axis.

    Args:
        points: Points to rotate [N, 8, 3]
        angle: Rotation angle [N]

    Returns:
        points_rot: Rotated points [N, 8, 3]
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
    Convert 3D boxes to corner points.

    Args:
        boxes3d: 3D boxes [N, 7] (x, y, z, dx, dy, dz, yaw)

    Returns:
        corners3d: Corner points [N, 8, 3]
    """
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
    """
    Extract BEV (bird's eye view) lines from corner points.

    Args:
        corners: Corner points [8, 3]

    Returns:
        lines: List of line segments for BEV visualization
    """
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]], [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    """
    Extract 3D lines from corner points for camera view visualization.

    Args:
        corners: Corner points [8, 3]

    Returns:
        lines: List of line segments for 3D visualization
    """
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]], [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    """
    Transform corner points to camera frame and project to image plane.

    Args:
        corners: Corner points in global/ego frame [8, 3]
        translation: Camera translation
        rotation: Camera rotation quaternion
        cam_intrinsics: Camera intrinsic matrix [3, 3]

    Returns:
        cam_corners: Projected corner points [8, 3] (x, y, depth)
    """
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners


def decode_predictions(preds, class_names, score_threshold=0.3):
    """
    Decode model predictions to bounding boxes.

    Args:
        preds: Model predictions (list of task predictions)
        class_names: Class names for each task
        score_threshold: Score threshold for filtering detections

    Returns:
        boxes_list: List of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        classes_list: List of class names
        scores_list: List of detection scores
    """
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


def boxes_to_corners(boxes_list, classes_list, show_range):
    """
    Convert boxes to corner format for visualization.

    Args:
        boxes_list: List of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        classes_list: List of class names
        show_range: Maximum range to show (meters)

    Returns:
        pred_corners: List of corner points
        pred_classes: List of filtered class names
    """
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
    """
    Get ground truth corners from annotations.

    Args:
        info: Sample information dictionary
        ego2global_rotation: Ego to global rotation
        ego2global_translation: Ego to global translation
        show_range: Maximum range to show (meters)

    Returns:
        gt_corners: List of ground truth corner points
    """
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
    """
    Load precomputed detection results from JSON file.

    Args:
        info: Sample information dictionary
        ego2global_rotation: Ego to global rotation
        ego2global_translation: Ego to global translation
        score_threshold: Score threshold for filtering detections

    Returns:
        boxes_list: List of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        classes_list: List of class names
        scores_list: List of detection scores
    """
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
    """
    Visualize BEVDepth predictions and ground truth.

    Args:
        info: Sample information dictionary
        pts: LiDAR points
        pred_corners_torch: Torch prediction corners
        pred_classes_torch: Torch prediction classes
        pred_corners_ttnn: TTNN prediction corners (can be None)
        pred_classes_ttnn: TTNN prediction classes (can be None)
        gt_corners: Ground truth corners
        output_path: Output file path for visualization
        show_range: Maximum range to show (meters)
    """
    import cv2
    import matplotlib.pyplot as plt

    cam_info = info["cam_infos"]

    show_torch = len(pred_corners_torch) > 0

    if pred_corners_ttnn is not None:
        if show_torch:
            fig = plt.figure(figsize=(24, 16))
            num_rows = 4
        else:
            fig = plt.figure(figsize=(24, 8))
            num_rows = 2
    else:
        fig = plt.figure(figsize=(24, 8))
        num_rows = 2

    if show_torch:
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
                        line[0],
                        line[1],
                        c=plt.colormaps["tab10"](SHOW_CLASSES.index(cls) if cls in SHOW_CLASSES else 0),
                    )

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

    if pred_corners_ttnn is not None:
        start_idx = 9 if show_torch else 1

        for i, k in enumerate(IMG_KEYS):
            if show_torch:
                fig_idx = i + 9 if i < 3 else i + 10
            else:
                fig_idx = i + 1 if i < 3 else i + 2
            ax = plt.subplot(num_rows, 4, fig_idx)
            ax.set_title(f"{k} (TTNN)" if show_torch else k)
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

        # BEV for TTNN
        bev_fig_idx = 12 if show_torch else 4
        ax_bev_ttnn = plt.subplot(num_rows, 4, bev_fig_idx)
        ax_bev_ttnn.set_title("BEV (TTNN)")
        ax_bev_ttnn.axis("equal")
        ax_bev_ttnn.set_xlim(-40, 40)
        ax_bev_ttnn.set_ylim(-40, 40)

        ax_bev_ttnn.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap="gray")

        # Always show ground truth in TTNN visualization
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
