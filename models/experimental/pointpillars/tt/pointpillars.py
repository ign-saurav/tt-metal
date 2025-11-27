import ttnn
from models.experimental.pointpillars.tt.pillar_encoder import TtPillarEncoder
from models.experimental.pointpillars.tt.backbone import TtBackbone
from models.experimental.pointpillars.tt.neck import TtNeck
from models.experimental.pointpillars.tt.head import TtHead
from models.experimental.pointpillars.reference.model.pointpillars import PillarLayer


class TtPointPillars:
    def __init__(
        self,
        nclasses,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        parameters,
        device,
        dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.device = device
        self.nclasses = nclasses
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        # Initialize all TTNN modules
        self.pillar_encoder = TtPillarEncoder(
            device=device,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64,
            parameters=parameters["pillar_encoder"],
        )

        self.backbone = TtBackbone(
            in_channel=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            parameters=parameters["backbone"],
            device=device,
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
            dtype=dtype,
        )

    def forward(self, batched_pts, mode="test", batched_gt_bboxes=None, batched_gt_labels=None):
        """
        Forward pass through the TTNN PointPillars network.

        Args:
            pillars: torch tensor (p1+p2+...+pb, num_points, 4)
            coors_batch: torch tensor (p1+p2+...+pb, 4)
            npoints_per_pillar: torch tensor (p1+p2+...+pb,)

        Returns:
            bbox_cls_pred: ttnn tensor (bs, n_anchors*n_classes, 248, 216)
            bbox_pred: ttnn tensor (bs, n_anchors*7, 248, 216)
            bbox_dir_cls_pred: ttnn tensor (bs, n_anchors*2, 248, 216)
        """
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # 1. Pillar encoding: (p1+p2+...+pb, num_points, 4) -> (bs, 64, 496, 432)
        pillar_features = self.pillar_encoder.forward(pillars, coors_batch, npoints_per_pillar)

        # 2. Backbone: (bs, 64, 496, 432) -> [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))
        xs = self.backbone.forward(pillar_features)

        # 3. Neck: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)] -> (bs, 384, 248, 216)
        x = self.neck.forward(xs)

        # 4. Head: (bs, 384, 248, 216) -> 3 detection outputs
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head.forward(x)

        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
