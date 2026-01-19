# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Based on PointPillars implementation from https://github.com/zhulf0804/PointPillars
# Original implementation by zhulf0804 under MIT license

import torch
import torch.nn as nn

import torch


def hard_voxelize_pytorch(points, voxel_size, coors_range, max_points=35, max_voxels=20000):
    """
    Pure PyTorch implementation of hard_voxelize

    Args:
        points: (N, C) tensor of points
        voxel_size: [x, y, z] voxel dimensions
        coors_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points: maximum points per voxel
        max_voxels: maximum number of voxels

    Returns:
        voxels: (M, max_points, C) tensor
        coordinates: (M, 3) tensor of voxel coordinates
        num_points_per_voxel: (M,) tensor
    """
    # Convert to tensors on the same device as points
    device = points.device
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
    coors_range = torch.tensor(coors_range, dtype=torch.float32, device=device)

    # Calculate grid size
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = torch.round(grid_size).long()

    # Filter points outside range
    mask = (
        (points[:, 0] >= coors_range[0])
        & (points[:, 0] < coors_range[3])
        & (points[:, 1] >= coors_range[1])
        & (points[:, 1] < coors_range[4])
        & (points[:, 2] >= coors_range[2])
        & (points[:, 2] < coors_range[5])
    )
    points = points[mask]

    # Compute voxel coordinates for each point
    voxel_coords = ((points[:, :3] - coors_range[:3]) / voxel_size).long()

    # Create unique voxel indices
    voxel_indices = (
        voxel_coords[:, 0] * grid_size[1] * grid_size[2] + voxel_coords[:, 1] * grid_size[2] + voxel_coords[:, 2]
    )

    # Get unique voxels and their inverse indices
    unique_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True)

    # Limit to max_voxels
    num_voxels = min(len(unique_indices), max_voxels)
    unique_indices = unique_indices[:num_voxels]

    # Initialize output tensors
    voxels = torch.zeros((num_voxels, max_points, points.shape[1]), dtype=points.dtype, device=device)
    coordinates = torch.zeros((num_voxels, 3), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.int32, device=device)

    # Fill voxels
    for i, voxel_idx in enumerate(unique_indices):
        point_mask = inverse_indices == i
        voxel_points = points[point_mask]

        # Limit points per voxel
        num_points = min(len(voxel_points), max_points)
        voxels[i, :num_points] = voxel_points[:num_points]
        num_points_per_voxel[i] = num_points

        # Store coordinates
        coordinates[i] = voxel_coords[point_mask][0]

    return voxels, coordinates, num_points_per_voxel


class _Voxelization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        voxels_out, coors_out, num_points_per_voxel_out = hard_voxelize_pytorch(
            points, voxel_size, coors_range, max_points, max_voxels
        )
        return voxels_out, coors_out, num_points_per_voxel_out


class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)

        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        input: shape=(N, c)
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return _Voxelization.apply(
            input, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
