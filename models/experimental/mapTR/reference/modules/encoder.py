# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


def gen_dx_bx(
    xbound: List[float],
    ybound: List[float],
    zbound: List[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate voxel grid parameters.

    Args:
        xbound: [min, max, step] for x-axis.
        ybound: [min, max, step] for y-axis.
        zbound: [min, max, step] for z-axis.

    Returns:
        Tuple of (dx, bx, nx) tensors.
    """
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseTransform(nn.Module):
    """Base class for image-to-BEV transformation (inference-only).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        feat_down_sample (int): Feature downsample factor.
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size (List[float]): Voxel size [x, y, z].
        dbound (List[float]): Depth bound [min, max, step].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feat_down_sample: int,
        pc_range: List[float],
        voxel_size: List[float],
        dbound: List[float],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_down_sample = feat_down_sample

        self.xbound = [pc_range[0], pc_range[3], voxel_size[0]]
        self.ybound = [pc_range[1], pc_range[4], voxel_size[1]]
        self.zbound = [pc_range[2], pc_range[5], voxel_size[2]]
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.register_buffer("dx", dx)
        self.register_buffer("bx", bx)
        self.register_buffer("nx", nx)

        self.C = out_channels
        self.frustum = None
        self.D = int((dbound[1] - dbound[0]) / dbound[2])

    def create_frustum(
        self,
        fH: int,
        fW: int,
        img_metas: List[Dict],
    ) -> torch.Tensor:
        """Create frustum grid for depth estimation.

        Args:
            fH: Feature height.
            fW: Feature width.
            img_metas: Image metadata.

        Returns:
            Frustum tensor with shape (D, fH, fW, 3).
        """
        iH = img_metas[0]["img_shape"][0][0]
        iW = img_metas[0]["img_shape"][0][1]
        assert iH // self.feat_down_sample == fH

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def get_geometry_v1(
        self,
        fH: int,
        fW: int,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
        lidar2ego_rots: torch.Tensor,
        lidar2ego_trans: torch.Tensor,
        img_metas: List[Dict],
        **kwargs,
    ) -> torch.Tensor:
        """Get 3D geometry from 2D features using camera parameters.

        Args:
            fH: Feature height.
            fW: Feature width.
            rots: Camera rotation matrices.
            trans: Camera translation vectors.
            intrins: Camera intrinsic matrices.
            post_rots: Post-processing rotation matrices.
            post_trans: Post-processing translation vectors.
            lidar2ego_rots: LiDAR to ego rotation matrices.
            lidar2ego_trans: LiDAR to ego translation vectors.
            img_metas: Image metadata.

        Returns:
            3D points with shape (B, N, D, H, W, 3).
        """
        B, N, _ = trans.shape
        device = trans.device

        if self.frustum is None:
            self.frustum = self.create_frustum(fH, fW, img_metas)
            self.frustum = self.frustum.to(device)

        # Undo post-transformation
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # Camera to ego transformation
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # Ego to lidar transformation
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = torch.inverse(lidar2ego_rots).view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_geometry(
        self,
        fH: int,
        fW: int,
        lidar2img: torch.Tensor,
        img_metas: List[Dict],
    ) -> torch.Tensor:
        """Get 3D geometry using lidar2img transformation matrix.

        Args:
            fH: Feature height.
            fW: Feature width.
            lidar2img: LiDAR to image transformation matrix.
            img_metas: Image metadata.

        Returns:
            3D points with shape (B, N, D, H, W, 3).
        """
        B, N, _, _ = lidar2img.shape
        device = lidar2img.device

        if self.frustum is None:
            self.frustum = self.create_frustum(fH, fW, img_metas)
            self.frustum = self.frustum.to(device)

        points = self.frustum.view(1, 1, self.D, fH, fW, 3).repeat(B, N, 1, 1, 1, 1)
        lidar2img = lidar2img.view(B, N, 1, 1, 1, 4, 4)

        points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        points = torch.linalg.solve(lidar2img.to(torch.float32), points.unsqueeze(-1).to(torch.float32)).squeeze(-1)

        eps = 1e-5
        points = points[..., 0:3] / torch.maximum(points[..., 3:4], torch.ones_like(points[..., 3:4]) * eps)

        return points

    def get_cam_feats(self, x: torch.Tensor) -> torch.Tensor:
        """Get camera features. To be implemented by subclasses."""
        raise NotImplementedError

    def bev_pool_simple(
        self,
        geom_feats: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Simple BEV pooling without CUDA ops (slower but portable).

        Args:
            geom_feats: Geometry features with shape (B, N, D, H, W, 3).
            x: Input features with shape (B, N, D, H, W, C).

        Returns:
            BEV features with shape (B, C, nx[0], nx[1]).
        """
        B, N, D, H, W, C = x.shape
        nx = self.nx.int()

        # Flatten
        x_flat = x.reshape(B, -1, C)  # (B, N*D*H*W, C)
        geom_flat = geom_feats.reshape(B, -1, 3)  # (B, N*D*H*W, 3)

        # Convert to voxel indices
        geom_idx = ((geom_flat - (self.bx - self.dx / 2.0)) / self.dx).long()

        # Create output
        output = torch.zeros(B, C, nx[2].int(), nx[0].int(), nx[1].int(), device=x.device, dtype=x.dtype)

        # Simple scatter (slow but works)
        for b in range(B):
            # Filter valid points
            valid = (
                (geom_idx[b, :, 0] >= 0)
                & (geom_idx[b, :, 0] < nx[0])
                & (geom_idx[b, :, 1] >= 0)
                & (geom_idx[b, :, 1] < nx[1])
                & (geom_idx[b, :, 2] >= 0)
                & (geom_idx[b, :, 2] < nx[2])
            )
            valid_idx = geom_idx[b, valid]
            valid_feats = x_flat[b, valid]

            # Scatter add
            for i in range(valid_idx.shape[0]):
                ix, iy, iz = valid_idx[i]
                output[b, :, iz, ix, iy] += valid_feats[i]

        # Collapse Z dimension
        final = output.sum(dim=2)  # (B, C, nx[0], nx[1])

        return final

    def forward(
        self,
        images: torch.Tensor,
        img_metas: List[Dict],
    ) -> torch.Tensor:
        """Forward function.

        Args:
            images: Input images with shape (B, N, C, H, W).
            img_metas: Image metadata.

        Returns:
            BEV features with shape (B, C, bev_h, bev_w).
        """
        B, N, C, fH, fW = images.shape

        # Extract camera parameters from metadata
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            camera2ego.append(img_meta["camera2ego"])
            camera_intrinsics.append(img_meta["camera_intrinsics"])
            img_aug_matrix.append(img_meta["img_aug_matrix"])
            lidar2ego.append(img_meta["lidar2ego"])

        camera2ego = images.new_tensor(np.asarray(camera2ego))
        camera_intrinsics = images.new_tensor(np.asarray(camera_intrinsics))
        img_aug_matrix = images.new_tensor(np.asarray(img_aug_matrix))
        lidar2ego = images.new_tensor(np.asarray(lidar2ego))

        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        geom = self.get_geometry_v1(
            fH, fW, rots, trans, intrins, post_rots, post_trans, lidar2ego_rots, lidar2ego_trans, img_metas
        )

        x = self.get_cam_feats(images)
        x = self.bev_pool_simple(geom, x)
        x = x.permute(0, 1, 3, 2).contiguous()

        return x


class LSSTransform(BaseTransform):
    """Lift-Splat-Shoot Transform for BEV feature generation (inference-only).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        feat_down_sample (int): Feature downsample factor.
        pc_range (List[float]): Point cloud range.
        voxel_size (List[float]): Voxel size.
        dbound (List[float]): Depth bound.
        downsample (int): Downsample factor. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feat_down_sample: int,
        pc_range: List[float],
        voxel_size: List[float],
        dbound: List[float],
        downsample: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
        )

        self.depthnet = nn.Conv2d(in_channels, int(self.D + self.C), 1)

        if downsample > 1:
            assert downsample == 2
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x: torch.Tensor) -> torch.Tensor:
        """Extract camera features with depth estimation.

        Args:
            x: Input features with shape (B, N, C, H, W).

        Returns:
            Features with shape (B, N, D, H, W, C).
        """
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def forward(
        self,
        images: torch.Tensor,
        img_metas: List[Dict],
    ) -> torch.Tensor:
        """Forward function.

        Args:
            images: Input images with shape (B, N, C, H, W).
            img_metas: Image metadata.

        Returns:
            BEV features with shape (B, C, bev_h, bev_w).
        """
        x = super().forward(images, img_metas)
        x = self.downsample(x)
        return x
