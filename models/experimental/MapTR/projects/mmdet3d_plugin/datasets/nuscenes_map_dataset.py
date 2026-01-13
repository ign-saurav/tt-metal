import copy

import numpy as np

# from mmdet.datasets import DATASETS
from models.experimental.MapTR.dependency import DATASETS

# from mmdet3d.datasets import NuScenesDataset
import mmcv
from models.experimental.MapTR.dependency import ProgressBar
import os
from os import path as osp

# from mmdet.datasets import DATASETS
from models.experimental.MapTR.dependency import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

# from mmcv.parallel import DataContainer as DC
from models.experimental.MapTR.dependency import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from models.experimental.MapTR.dependency import to_tensor
import json


def add_rotation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise_angle = torch.normal(mean, std=std, size=(3,))
    # extrinsics[:, 0:3, 0:3] *= (1 + noise)
    sin_noise = torch.sin(noise_angle)
    cos_noise = torch.cos(noise_angle)
    rotation_matrix = torch.eye(4).view(4, 4)
    #  rotation_matrix[]
    rotation_matrix_x = rotation_matrix.clone()
    rotation_matrix_x[1, 1] = cos_noise[0]
    rotation_matrix_x[1, 2] = sin_noise[0]
    rotation_matrix_x[2, 1] = -sin_noise[0]
    rotation_matrix_x[2, 2] = cos_noise[0]

    rotation_matrix_y = rotation_matrix.clone()
    rotation_matrix_y[0, 0] = cos_noise[1]
    rotation_matrix_y[0, 2] = -sin_noise[1]
    rotation_matrix_y[2, 0] = sin_noise[1]
    rotation_matrix_y[2, 2] = cos_noise[1]

    rotation_matrix_z = rotation_matrix.clone()
    rotation_matrix_z[0, 0] = cos_noise[2]
    rotation_matrix_z[0, 1] = sin_noise[2]
    rotation_matrix_z[1, 0] = -sin_noise[2]
    rotation_matrix_z[1, 1] = cos_noise[2]

    rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    rotation = torch.from_numpy(extrinsics.astype(np.float32))
    rotation[:3, -1] = 0.0
    # import pdb;pdb.set_trace()
    rotation = rotation_matrix @ rotation
    extrinsics[:3, :3] = rotation[:3, :3].numpy()
    return extrinsics


def add_translation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise = torch.normal(mean, std=std, size=(3,))
    extrinsics[0:3, -1] += noise.numpy()
    return extrinsics


class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates"""

    def __init__(
        self,
        instance_line_list,
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=None,
    ):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(dtype=torch.float32)
        instance_se_points_tensor[:, 0] = torch.clamp(instance_se_points_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 1] = torch.clamp(instance_se_points_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_se_points_tensor[:, 2] = torch.clamp(instance_se_points_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 3] = torch.clamp(instance_se_points_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 1] = torch.clamp(instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_bbox_tensor[:, 2] = torch.clamp(instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 3] = torch.clamp(instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0, 2, 1)
            sampled_pts = torch.nn.functional.interpolate(
                poly_pts, size=(self.fixed_num), mode="linear", align_corners=True
            )
            sampled_pts = sampled_pts.permute(0, 2, 1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list, dim=0)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num - shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y
            )
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full(
                    [final_shift_num - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value
                )
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            if shifts_num > 2 * final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index + shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts, flip1_shifts_pts), axis=0)

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y
            )
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full(
                    [final_shift_num * 2 - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value
                )
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num * 2, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num * 2 - shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False

        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor


class VectorizedLocalMap(object):
    CLASS2LABEL = {"road_divider": 0, "lane_divider": 0, "ped_crossing": 1, "contours": 2, "others": -1}

    def __init__(
        self,
        dataroot,
        patch_size,
        map_classes=["divider", "ped_crossing", "boundary"],
        line_classes=["road_divider", "lane_divider"],
        ped_crossing_classes=["ped_crossing"],
        contour_classes=["road_segment", "lane"],
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
    ):
        """
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        """
        super().__init__()
        self.data_root = dataroot

        # Try to find the actual NuScenes dataset root (where maps/ directory exists)
        # Check common locations: data_root itself, parent directories, or common nuscenes paths
        map_dataroot = self.data_root
        possible_paths = [
            self.data_root,
            os.path.join(self.data_root, ".."),
            os.path.join(self.data_root, "../.."),
            os.path.join(os.path.expanduser("~"), "data", "nuscenes"),
            "/data/nuscenes",
            "data/nuscenes",
        ]

        # Find the first path that has maps directory (with or without expansion subdirectory)
        for path in possible_paths:
            path = os.path.abspath(path)
            maps_dir = os.path.join(path, "maps")
            if os.path.exists(maps_dir) and os.path.isdir(maps_dir):
                map_dataroot = path
                break

        self.MAPS = ["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}

        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=map_dataroot, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        """
        use lidar2global to get gt map layers
        """

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # Debug: Check coordinate transformation
        import os

        debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
        if debug_enabled:
            print(f"\n=== GT Coordinate Transformation Debug ===")
            print(f"lidar2global_translation: {lidar2global_translation}")
            print(f"map_pose (patch center): ({map_pose[0]:.2f}, {map_pose[1]:.2f})")
            print(f"patch_box: {patch_box}")
            print(f"patch_angle: {patch_angle:.2f} degrees")
            print(f"GT should be transformed to ego frame (centered at 0, 0)")
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == "divider":
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == "ped_crossing":
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get("ped_crossing", -1)))
            elif vec_class == "boundary":
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get("contours", -1)))
            else:
                raise ValueError(f"WRONG vec_class: {vec_class}")

        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)

        gt_instance = LiDARInstanceLines(
            gt_instance,
            self.sample_dist,
            self.num_samples,
            self.padding,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )

        # Debug: Check GT coordinates after transformation
        import os

        debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
        if debug_enabled and len(gt_instance.instance_list) > 0:
            # Get first few GT instances to check coordinates
            for i, inst in enumerate(gt_instance.instance_list[:3]):
                coords = np.array(list(inst.coords))
                print(f"GT instance {i} after transformation:")
                print(f"  Coords shape: {coords.shape}")
                print(f"  X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                print(f"  Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                print(f"  Center: ({coords[:, 0].mean():.2f}, {coords[:, 1].mean():.2f})")
                print(f"  Should be centered near (0, 0) in ego frame")

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
        )
        return anns_results

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == "LineString":
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != "MultiPolygon":
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError("{} is not a polygonal layer".format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == "drivable_area":
            for record in records:
                polygons = [
                    self.map_explorer[location].map_api.extract_polygon(polygon_token)
                    for polygon_token in record["polygon_tokens"]
                ]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                        )
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record["polygon_token"])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                        )
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == "traffic_light":
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record["line_token"])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, "ped_crossing")
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record["polygon_token"])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(
                        new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                    )
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == "Polygon":
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[: self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid


@DATASETS.register_module()
class CustomNuScenesLocalMapDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset add static map elements
    """
    MAPCLASSES = ("divider",)

    def __init__(
        self,
        map_ann_file=None,
        queue_length=4,
        bev_size=(200, 200),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        overlap_test=False,
        fixed_ptsnum_per_line=-1,
        eval_use_same_gt_sample_num_flag=False,
        padding_value=-10000,
        map_classes=None,
        noise="None",
        noise_std=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_ann_file = map_ann_file

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.vector_map = VectorizedLocalMap(
            kwargs["data_root"],
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
        )
        self.is_vis_on_test = False
        self.noise = noise
        self.noise_std = noise_std

        # Initialize nusc if not already initialized (needed for get_data_info)
        if not hasattr(self, "nusc") or self.nusc is None:
            from nuscenes import NuScenes

            self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f"Unsupported type {type(map_classes)} of map classes.")

        return class_names

    def vectormap_pipeline(self, example, input_dict):
        """
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        """
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = input_dict["lidar2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = input_dict["ego2global_translation"]

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        location = input_dict["map_location"]
        ego2global_translation = input_dict["ego2global_translation"]
        ego2global_rotation = input_dict["ego2global_rotation"]
        anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)

        """
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        """
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
        else:
            gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train,
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example["gt_labels_3d"] = DC(gt_vecs_label, cpu_only=False)
        example["gt_bboxes_3d"] = DC(gt_vecs_pts_loc, cpu_only=True)
        return example

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []

        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            example = self.vectormap_pipeline(example, input_dict)
            if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each["img"].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if metas_map[i]["scene_token"] != prev_scene_token:
                metas_map[i]["prev_bev_exists"] = False
                prev_scene_token = metas_map[i]["scene_token"]
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
            else:
                metas_map[i]["prev_bev_exists"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # Extract map_location from scene token if not in info
        if "map_location" not in info:
            # Get map location from scene's log token
            scene_token = info["scene_token"]
            scene = self.nusc.get("scene", scene_token)
            log = self.nusc.get("log", scene["log_token"])
            map_location = log["location"]
        else:
            map_location = info["map_location"]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"],
            map_location=map_location,
        )
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            for cam_type, cam_info in info["cams"].items():
                img_path = cam_info["data_path"]

                # Try multiple path resolution strategies
                resolved_path = None

                # Strategy 1: Use original path if it exists
                if os.path.exists(img_path):
                    resolved_path = img_path
                else:
                    # Strategy 2: Fix relative path with hardcoded base
                    if not os.path.isabs(img_path):
                        hardcoded_base = "models/experimental/MapTR/"
                        test_path = os.path.join(hardcoded_base, img_path.lstrip("./"))
                        if os.path.exists(test_path):
                            resolved_path = test_path

                    # Strategy 3: Try to find in camera directory using sample token
                    if not resolved_path:
                        # Extract sample token from lidar path or use sample_idx
                        sample_token = input_dict.get("sample_idx", "")
                        if not sample_token and "lidar_path" in input_dict:
                            lidar_basename = os.path.basename(input_dict["lidar_path"])
                            # Extract token from filename like: n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
                            if "__" in lidar_basename:
                                sample_token = lidar_basename.split("__")[0]

                        # Try common base paths
                        base_paths = [
                            "models/experimental/MapTR/data/nuscenes/samples",
                            "data/nuscenes/samples",
                        ]

                        for base_path in base_paths:
                            if os.path.exists(base_path):
                                cam_dir = os.path.join(base_path, cam_type)
                                if os.path.exists(cam_dir):
                                    # Try to find image matching sample token
                                    if sample_token:
                                        # Look for files starting with sample_token
                                        for f in os.listdir(cam_dir):
                                            if f.startswith(sample_token) and f.endswith((".jpg", ".png", ".jpeg")):
                                                resolved_path = os.path.join(cam_dir, f)
                                                break

                                    # If not found, try any image in the directory
                                    if not resolved_path:
                                        img_files = [
                                            f for f in os.listdir(cam_dir) if f.endswith((".jpg", ".png", ".jpeg"))
                                        ]
                                        if img_files:
                                            # Prefer files matching the expected pattern
                                            matching_files = (
                                                [f for f in img_files if sample_token in f] if sample_token else []
                                            )
                                            if matching_files:
                                                resolved_path = os.path.join(cam_dir, matching_files[0])
                                            else:
                                                resolved_path = os.path.join(cam_dir, img_files[0])

                                    if resolved_path:
                                        break

                    # Strategy 4: Try original directory structure
                    if not resolved_path:
                        img_dir = os.path.dirname(img_path)
                        if os.path.exists(img_dir):
                            img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
                            if img_files:
                                resolved_path = os.path.join(img_dir, img_files[0])

                if resolved_path and os.path.exists(resolved_path):
                    image_paths.append(resolved_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2cam_rt_t = lidar2cam_rt.T

                if self.noise == "rotation":
                    lidar2cam_rt_t = add_rotation_noise(lidar2cam_rt_t, std=self.noise_std)
                elif self.noise == "translation":
                    lidar2cam_rt_t = add_translation_noise(lidar2cam_rt_t, std=self.noise_std)

                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt_t
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt_t)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            # Only update with image data if we have valid images
            if len(image_paths) > 0:
                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                        cam_intrinsic=cam_intrinsics,
                        lidar2cam=lidar2cam_rts,
                    )
                )
            else:
                # Set empty lists to avoid KeyError in pipeline
                input_dict.update(
                    dict(
                        img_filename=[],
                        lidar2img=[],
                        cam_intrinsic=[],
                        lidar2cam=[],
                    )
                )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = input_dict["lidar2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = input_dict["ego2global_translation"]
        lidar2global = ego2global @ lidar2ego
        input_dict["lidar2global"] = lidar2global
        return input_dict

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)
        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def __len__(self):
        return 1

    def _format_gt(self):
        gt_annos = []
        print("Start to convert gt map format...")
        assert self.map_ann_file is not None
        if not os.path.exists(self.map_ann_file):
            dataset_length = len(self)
            prog_bar = ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]["token"]
                gt_anno = {}
                gt_anno["sample_token"] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                input_dict = self.get_data_info(sample_id)
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, input_dict)
                gt_labels = gt_sample_dict["gt_labels_3d"].data.numpy()
                gt_vecs = gt_sample_dict["gt_bboxes_3d"].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    pts = np.array(list(gt_vec.coords))
                    # Debug: Check GT coordinate centering
                    import os

                    debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
                    if debug_enabled and i == 0:
                        print(f"\n=== GT Coordinate Debug (first vector) ===")
                        print(f"GT pts shape: {pts.shape}")
                        print(
                            f"GT pts range: X[{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}], Y[{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]"
                        )
                        print(f"GT center: ({pts[:, 0].mean():.2f}, {pts[:, 1].mean():.2f})")
                        print(f"GT should be centered near (0, 0) in ego frame")
                    anno = dict(
                        pts=pts,
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno["vectors"] = gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {"GTs": gt_annos}
            print("\n GT anns writes to", self.map_ann_file)

            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                return obj

            with open(self.map_ann_file, "w") as f:
                json.dump(convert_to_serializable(nusc_submissions), f, indent=2)
        else:
            print(f"{self.map_ann_file} exist, not update")

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        # import pdb;pdb.set_trace()
        print("Start to convert map detection format...")
        from models.experimental.MapTR.dependency import track_iter_progress

        for sample_id, det in enumerate(track_iter_progress(results)):
            pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]["token"]
            pred_anno["sample_token"] = sample_token
            pred_vec_list = []
            for i, vec in enumerate(vecs):
                name = mapped_class_names[vec["label"]]
                anno = dict(
                    pts=vec["pts"],
                    pts_num=len(vec["pts"]),
                    cls_name=name,
                    type=vec["label"],
                    confidence_level=vec["score"],
                )
                pred_vec_list.append(anno)

            pred_anno["vectors"] = pred_vec_list
            pred_annos.append(pred_anno)

        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f"{self.map_ann_file} exist, not update")

        nusc_submissions = {
            "meta": self.modality,
            "results": pred_annos,
        }

        os.makedirs(jsonfile_prefix, exist_ok=True)
        res_path = osp.join(jsonfile_prefix, "nuscmap_results.json")
        print("Results writes to", res_path)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        with open(res_path, "w") as f:
            json.dump(convert_to_serializable(nusc_submissions), f, indent=2)
        return res_path, None

    def to_gt_vectors(self, gt_dict):
        # import pdb;pdb.set_trace()
        gt_labels = gt_dict["gt_labels_3d"].data
        gt_instances = gt_dict["gt_bboxes_3d"].data.instance_list

        gt_vectors = []

        for gt_instance, gt_label in zip(gt_instances, gt_labels):
            pts, pts_num = sample_pts_from_line(gt_instance, patch_size=self.patch_size)
            gt_vectors.append({"pts": pts, "pts_num": pts_num, "type": int(gt_label)})
        vector_num_list = {}
        for i in range(self.NUM_MAPCLASSES):
            vector_num_list[i] = []
        for vec in gt_vectors:
            if vector["pts_num"] >= 2:
                vector_num_list[vector["type"]].append(
                    (LineString(vector["pts"][: vector["pts_num"]]), vector.get("confidence_level", 1))
                )
        return gt_vectors

    def _evaluate_single(self, result_path, logger=None, metric="chamfer", result_name="pts_bbox"):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from models.experimental.MapTR.projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
        from models.experimental.MapTR.projects.mmdet3d_plugin.datasets.map_utils.mean_ap import (
            format_res_gt_by_classes,
        )

        result_path = osp.abspath(result_path)
        detail = dict()

        print("Formating results & gts by classes")
        with open(result_path, "r") as f:
            pred_results = json.load(f)
        gen_results = pred_results["results"]
        print(f"DEBUG: Loaded {len(gen_results)} prediction samples")
        with open(self.map_ann_file, "r") as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns["GTs"]
        print(f"DEBUG: Loaded {len(annotations)} ground truth samples")

        # Check for samples with no ground truth vectors
        total_gt_vectors = sum(len(ann.get("vectors", [])) for ann in annotations if isinstance(ann, dict))
        total_pred_vectors = sum(len(res.get("vectors", [])) for res in gen_results if isinstance(res, dict))

        print(f"DEBUG: Total GT vectors across all samples: {total_gt_vectors}")
        print(f"DEBUG: Total prediction vectors across all samples: {total_pred_vectors}")

        if total_gt_vectors == 0:
            print("WARNING: No ground truth vectors found in any sample!")
            print("This could mean:")
            print("  1. The test samples don't have ground truth annotations")
            print("  2. The annotation file is incomplete or incorrect")
            print("  3. You're evaluating on a test set (use validation set instead)")
            print("  4. The sample tokens don't match between predictions and ground truth")

        # Debug: Check first few samples
        if len(annotations) > 0:
            print(
                f"DEBUG: First GT sample keys: {list(annotations[0].keys()) if isinstance(annotations[0], dict) else 'not a dict'}"
            )
            if isinstance(annotations[0], dict):
                sample_token = annotations[0].get("sample_token", "N/A")
                print(f"DEBUG: First GT sample_token: {sample_token}")
                if "vectors" in annotations[0]:
                    print(f"DEBUG: First GT sample has {len(annotations[0]['vectors'])} vectors")
                    if len(annotations[0]["vectors"]) > 0:
                        print(
                            f"DEBUG: First vector type: {annotations[0]['vectors'][0].get('type', 'N/A')}, pts shape: {len(annotations[0]['vectors'][0].get('pts', []))}"
                        )
        if len(gen_results) > 0:
            print(
                f"DEBUG: First pred sample keys: {list(gen_results[0].keys()) if isinstance(gen_results[0], dict) else 'not a dict'}"
            )
            if isinstance(gen_results[0], dict):
                sample_token = gen_results[0].get("sample_token", "N/A")
                print(f"DEBUG: First pred sample_token: {sample_token}")
                if "vectors" in gen_results[0]:
                    print(f"DEBUG: First pred sample has {len(gen_results[0]['vectors'])} vectors")

        # Check if sample tokens match
        if len(annotations) > 0 and len(gen_results) > 0:
            gt_token = annotations[0].get("sample_token") if isinstance(annotations[0], dict) else None
            pred_token = gen_results[0].get("sample_token") if isinstance(gen_results[0], dict) else None
            if gt_token and pred_token:
                if gt_token != pred_token:
                    print(f"WARNING: Sample tokens don't match! GT: {gt_token}, Pred: {pred_token}")
                else:
                    print(f"DEBUG: Sample tokens match: {gt_token}")
        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path,
            gen_results,
            annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
        )

        metrics = metric if isinstance(metric, list) else [metric]
        # Map mAP to iou (mAP is computed using IoU metric)
        metrics = ["iou" if m == "mAP" or m.lower() == "map" else m for m in metrics]
        allowed_metrics = ["chamfer", "iou"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    f"metric {metric} is not supported. Allowed metrics: {allowed_metrics} (or use 'mAP' which maps to 'iou')"
                )

        for metric in metrics:
            print("-*" * 10 + f"use metric:{metric}" + "-*" * 10)

            if metric == "chamfer":
                thresholds = [0.5, 1.0, 1.5]
            elif metric == "iou":
                thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print("-*" * 10 + f"threshhold:{thr}" + "-*" * 10)
                mAP, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric,
                )
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]["ap"]

            for i, name in enumerate(self.MAPCLASSES):
                print("{}: {}".format(name, cls_aps.mean(0)[i]))
                detail["NuscMap_{}/{}_AP".format(metric, name)] = cls_aps.mean(0)[i]
            print("map: {}".format(cls_aps.mean(0).mean()))
            detail["NuscMap_{}/mAP".format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == "chamfer":
                        detail["NuscMap_{}/{}_AP_thr_{}".format(metric, name, thr)] = cls_aps[j][i]
                    elif metric == "iou":
                        if thr == 0.5 or thr == 0.75:
                            detail["NuscMap_{}/{}_AP_thr_{}".format(metric, name, thr)] = cls_aps[j][i]

        return detail

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


def output_to_vecs(detection):
    if "pts_bbox" in detection:
        detection = detection["pts_bbox"]
    box3d = detection["boxes_3d"].numpy()
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    pts = detection["pts_3d"].numpy()

    # Debug: Check coordinates at extraction point
    import os

    debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
    if debug_enabled and len(pts) > 0:
        print(f"\n=== output_to_vecs Debug ===")
        print(f"pts shape: {pts.shape}")
        print(f"pts range: [{pts.min():.2f}, {pts.max():.2f}]")
        print(f"First prediction pts sample: {pts[0, :3]}")
        print(f"boxes_3d range: [{box3d.min():.2f}, {box3d.max():.2f}]")
        print(f"First box3d: {box3d[0]}")

    vec_list = []
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox=box3d[i],  # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list


def sample_pts_from_line(
    line,
    fixed_num=-1,
    sample_dist=1,
    normalize=False,
    patch_size=None,
    padding=False,
    num_samples=250,
):
    if fixed_num < 0:
        distances = np.arange(0, line.length, sample_dist)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    else:
        # fixed number of points, so distance is line.length / fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

    if normalize:
        sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])

    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

        if normalize:
            sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])
            num_valid = len(sampled_points)

    return sampled_points, num_valid
