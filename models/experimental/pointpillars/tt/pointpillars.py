# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.pointpillars.tt.pillar_encoder import TtPillarEncoder
from models.experimental.pointpillars.tt.backbone import TtBackbone
from models.experimental.pointpillars.tt.neck import TtNeck
from models.experimental.pointpillars.tt.head import TtHead
from models.experimental.pointpillars.reference.pointpillars import PillarLayer


class PointPillarsPreprocessor:
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        parameters,
        device,
    ):
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        self.pillar_encoder = TtPillarEncoder(
            device=device,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64,
            parameters=parameters["pillar_encoder"],
        )

    def forward(self, batched_pts):
        """Process point cloud to pillar features"""
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        pillar_features = self.pillar_encoder.forward(pillars, coors_batch, npoints_per_pillar)
        return pillar_features


class TtPointPillars:
    def __init__(
        self,
        nclasses,
        parameters,
        device,
        dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.device = device
        self.nclasses = nclasses

        self.backbone = TtBackbone(
            in_channel=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            parameters=parameters["backbone"],
            device=device,
            batch_size=1,
            input_height=496,
            input_width=432,
            dtype=dtype,
        )

        self.neck = TtNeck(
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128],
            parameters=parameters["neck"],
            device=device,
            dtype=dtype,
        )

        self.head = TtHead(
            in_channel=384,
            n_anchors=2 * nclasses,
            n_classes=nclasses,
            parameters=parameters["head"],
            device=device,
            batch_size=1,
            input_height=248,
            input_width=216,
            dtype=dtype,
        )

    def forward(self, pillar_features, mode="test", batched_gt_bboxes=None, batched_gt_labels=None):
        """
        Forward pass through the TTNN PointPillars network starting from backbone.

        Args:
            pillar_features: ttnn tensor (bs, 64, 496, 432) - preprocessed pillar features

        Returns:
            bbox_cls_pred: ttnn tensor (bs, n_anchors*n_classes, 248, 216)
            bbox_pred: ttnn tensor (bs, n_anchors*7, 248, 216)
            bbox_dir_cls_pred: ttnn tensor (bs, n_anchors*2, 248, 216)
        """
        xs = self.backbone.forward(pillar_features)

        x = self.neck.forward(xs)

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head.forward(x)

        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
