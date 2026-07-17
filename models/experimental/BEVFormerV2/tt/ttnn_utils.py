# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os
import numpy as np
import cv2
from PIL import Image
import torch


def multi_scale_deformable_attn(value, value_spatial_shapes, sampling_locations, attention_weights, device):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels_sampling, num_points, _ = sampling_locations.shape
    num_levels_actual = value_spatial_shapes.shape[0]

    if num_levels_actual == 1:
        value_list = [value]
    else:
        split_sizes = [
            int(value_spatial_shapes[level, 0].item() * value_spatial_shapes[level, 1].item())
            for level in range(num_levels_actual)
        ]
        value_list = ttnn.split(value, split_sizes, dim=1)

    sampling_locations = sampling_locations[:, :, :, :num_levels_actual, :, :]
    sampling_locations_torch = ttnn.to_torch(sampling_locations)
    sampling_grids_torch = 2 * sampling_locations_torch - 1
    sampling_value_list = []

    for level in range(num_levels_actual):
        H_ = value_spatial_shapes[level, 0]
        W_ = value_spatial_shapes[level, 1]
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [value_l_.shape[0], value_l_.shape[1], value_l_.shape[2] * value_l_.shape[3]])
        value_l_ = ttnn.permute(value_l_, (0, 2, 1))

        value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, int(H_.item()), int(W_.item())])

        sampling_grid_l_torch = sampling_grids_torch[:, :, :, level]
        sampling_grid_l_torch = sampling_grid_l_torch.permute(0, 2, 1, 3, 4)
        sampling_grid_l_torch = sampling_grid_l_torch.reshape(
            sampling_grid_l_torch.shape[0] * sampling_grid_l_torch.shape[1],
            sampling_grid_l_torch.shape[2],
            sampling_grid_l_torch.shape[3],
            sampling_grid_l_torch.shape[4],
        )
        sampling_grid_l_ = ttnn.from_torch(
            sampling_grid_l_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        value_l_ = ttnn.permute(value_l_, (0, 2, 3, 1))
        value_l_ = ttnn.to_layout(value_l_, layout=ttnn.ROW_MAJOR_LAYOUT)
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        ttnn.deallocate(value_l_)
        ttnn.deallocate(sampling_grid_l_)
        sampling_value_l_ = ttnn.permute(sampling_value_l_, (0, 3, 1, 2))
        sampling_value_l_torch = ttnn.to_torch(sampling_value_l_)
        ttnn.deallocate(sampling_value_l_)
        sampling_value_list.append(sampling_value_l_torch)

    attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))
    attention_weights = attention_weights[:, :, :, :num_levels_actual, :]
    attention_weights_torch = ttnn.to_torch(attention_weights)
    attention_weights_torch = attention_weights_torch.view(
        bs * num_heads, 1, num_queries, num_levels_actual * num_points
    )

    output_torch = torch.stack(sampling_value_list, -2)
    output_torch = output_torch.view(
        output_torch.shape[0],
        output_torch.shape[1],
        output_torch.shape[2],
        output_torch.shape[3] * output_torch.shape[4],
    )
    output_torch = output_torch * attention_weights_torch
    output_torch = output_torch.sum(3)
    output_torch = output_torch.view(bs, num_heads * embed_dims, num_queries)
    output_torch = output_torch.permute(0, 2, 1)

    output = ttnn.from_torch(output_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(attention_weights)
    ttnn.deallocate(value)

    return output


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def bbox_xyxy_to_cxcywh(bbox):
    bbox = ttnn.unsqueeze(bbox, 0)
    bbox = ttnn.to_layout(bbox, layout=ttnn.ROW_MAJOR_LAYOUT)
    x1, y1, x2, y2 = ttnn.split(bbox, (1, 1, 1, 1), 2)
    bbox_new = [ttnn.div((x1 + x2), 2), ttnn.div((y1 + y2), 2), (x2 - x1), (y2 - y1)]
    return ttnn.concat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = ttnn.split(bbox, (1, 1, 1, 1), dim=-1)

    bbox_new = [ttnn.mul((cx - 0.5), w), ttnn.mul((cy - 0.5), h), ttnn.mul((cx + 0.5), w), ttnn.mul((cy + 0.5), h)]
    return ttnn.concat(bbox_new, dim=-1)


def tt_denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)

    bboxes_reshaped = ttnn.reshape(bboxes, (bboxes.shape[0], 2, 2))
    bboxes_even = bboxes_reshaped[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxex_odd = bboxes_reshaped[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]

    bboxes_combined = ttnn.stack([bboxes_even, bboxex_odd], dim=-1)
    bboxes = ttnn.reshape(bboxes_combined, [bboxes.shape[0], -1])

    return bboxes


def tt_denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts_slice1 = new_pts[..., 0:1]
    new_pts_slice2 = new_pts[..., 1:2]
    new_pts_slice1 = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts_slice2 = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts = ttnn.concat([new_pts_slice1, new_pts_slice2], dim=-1)
    return new_pts


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img.astype(np.float32)


def normalize_image(img, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0]):
    img = img.copy()
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img


def pad_image(img, size_divisor=32):
    h, w = img.shape[:2]
    pad_h = (size_divisor - h % size_divisor) % size_divisor
    pad_w = (size_divisor - w % size_divisor) % size_divisor
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    return img


def crop_resize_flip_image(img, crop=(0, 260, 1600, 900), resize_h=640, flip=False):
    x1, y1, x2, y2 = crop
    img_pil = Image.fromarray(np.uint8(img))
    img_pil = img_pil.crop(crop)

    original_h = y2 - y1
    resize = resize_h / original_h
    resize_w = int((x2 - x1) * resize)
    resize_dims = (resize_w, resize_h)

    img_pil = img_pil.resize(resize_dims, Image.BILINEAR)
    if flip:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    ida_rot = np.eye(2) * resize
    ida_tran = -np.array(crop[:2]) * resize
    ida_mat = np.eye(3)
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 2] = ida_tran

    img = np.array(img_pil).astype(np.float32)
    return img, ida_mat, resize_dims


def prepare_sample_images(info, data_root):
    cam_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    ida_aug_conf = {
        "resize": [640],
        "crop": (0, 260, 1600, 900),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
    crop = ida_aug_conf["crop"]
    resized_h = ida_aug_conf["resize"][0]
    flip = False

    img_paths = []
    lidar2cam_rts = []
    cam2img_list = []
    ori_img_shapes = []

    for cam_type in cam_types:
        cam_info = info["cams"][cam_type]
        data_path = cam_info["data_path"]

        if data_path.startswith("./data/nuscenes"):
            remaining_path = data_path[len("./data/nuscenes") :].lstrip("/")
            full_path = os.path.join(data_root, remaining_path)
        elif not os.path.isabs(data_path):
            full_path = os.path.join(data_root, data_path)
        else:
            full_path = data_path

        img = load_image(full_path)
        ori_img_shapes.append(img.shape[:2])

        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = cam_info["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

        img_paths.append(full_path)
        cam2img_list.append(viewpad.copy())
        lidar2cam_rts.append(lidar2cam_rt.T)

    processed_imgs = []
    img_shapes = []
    lidar2img_rts = []

    for img_path, cam2img, lidar2cam in zip(img_paths, cam2img_list, lidar2cam_rts):
        img = load_image(img_path)
        img, ida_mat, actual_resize_dims = crop_resize_flip_image(img, crop=crop, resize_h=resized_h, flip=flip)

        cam2img[:3, :3] = np.matmul(ida_mat, cam2img[:3, :3])
        lidar2img_rt = np.matmul(cam2img, lidar2cam)

        img = normalize_image(img, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0])
        img = pad_image(img, size_divisor=32)

        img_shapes.append(img.shape[:2])
        lidar2img_rts.append(lidar2img_rt)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if len(processed_imgs) == 0:
            processed_imgs = img.unsqueeze(0)
        else:
            processed_imgs = torch.cat([processed_imgs, img.unsqueeze(0)], dim=0)

    if processed_imgs.dim() == 4:
        processed_imgs = processed_imgs.unsqueeze(0)

    can_bus = info.get("can_bus", np.zeros(18))
    ego2global_translation = info.get("ego2global_translation", np.zeros(3))
    ego2global_rotation = info.get("ego2global_rotation", np.eye(3))
    lidar2ego_translation = info.get("lidar2ego_translation", np.zeros(3))
    lidar2ego_rotation = info.get("lidar2ego_rotation", np.eye(3))
    timestamp = info.get("timestamp", 0.0) / 1e6

    img_metas = [
        {
            "sample_idx": info.get("token", "unknown"),
            "img_shape": img_shapes,
            "ori_shape": ori_img_shapes,
            "lidar2img": [lidar2img_rt.tolist() for lidar2img_rt in lidar2img_rts],
            "can_bus": can_bus.tolist() if isinstance(can_bus, np.ndarray) else can_bus,
            "ego2global_translation": ego2global_translation.tolist()
            if isinstance(ego2global_translation, np.ndarray)
            else ego2global_translation,
            "ego2global_rotation": ego2global_rotation.tolist()
            if isinstance(ego2global_rotation, np.ndarray)
            else ego2global_rotation,
            "lidar2ego_translation": lidar2ego_translation.tolist()
            if isinstance(lidar2ego_translation, np.ndarray)
            else lidar2ego_translation,
            "lidar2ego_rotation": lidar2ego_rotation.tolist()
            if isinstance(lidar2ego_rotation, np.ndarray)
            else lidar2ego_rotation,
            "timestamp": timestamp,
            "pad_shape": [(img_shapes[0][0], img_shapes[0][1], 3)] * len(img_shapes),
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "img_norm_cfg": {
                "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                "to_rgb": True,
            },
        }
    ]

    return processed_imgs, img_metas
