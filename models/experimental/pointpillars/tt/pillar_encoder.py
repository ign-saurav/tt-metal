# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from models.experimental.pointpillars.tt.utils import TtPointPillarsConv1D


class TtPillarEncoder:
    def __init__(
        self,
        device,
        voxel_size,
        point_cloud_range,
        in_channel,
        out_channel,
        parameters,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        self.shard_layout = shard_layout
        self.conv1d = TtPointPillarsConv1D(
            parameters["conv_args"]["conv"],
            parameters["conv"],
            device=device,
            activation=None,
            shard_layout=None,
            deallocate_activation=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: torch tensor (p1+p2+...+pb, num_points, c)
        coors_batch: torch tensor (p1+p2+...+pb, 4)
        npoints_per_pillar: torch tensor (p1+p2+...+pb,)
        """
        offset_pt_center_tt = ttnn.from_torch(
            pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x_offset_pi_center_tt = ttnn.from_torch(
            (pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        y_offset_pi_center_tt = ttnn.from_torch(
            (pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pillars_feature_tt = ttnn.from_torch(
            pillars[:, :, 2:],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        features_tt = ttnn.concat(
            [
                x_offset_pi_center_tt,
                y_offset_pi_center_tt,
                pillars_feature_tt,
                offset_pt_center_tt,
                x_offset_pi_center_tt,
                y_offset_pi_center_tt,
            ],
            dim=-1,
        )

        npoints_per_pillar_tt = ttnn.from_torch(
            npoints_per_pillar[None, :],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        voxel_ids = ttnn.arange(0, pillars.size(1), dtype=ttnn.bfloat16, device=self.device)
        mask = ttnn.unsqueeze(voxel_ids, dim=1) < npoints_per_pillar_tt
        mask = ttnn.permute(mask, (1, 0))
        mask = ttnn.unsqueeze(mask, dim=2)

        features_tt = features_tt * mask
        ttnn.deallocate(mask)

        num_pillars, num_points, in_ch = features_tt.shape

        features_tt = self.conv1d(features_tt)

        features_tt = ttnn.to_memory_config(features_tt, ttnn.DRAM_MEMORY_CONFIG)
        features_tt = ttnn.relu(features_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        features_tt = ttnn.reshape(features_tt, (num_pillars, num_points, features_tt.shape[-1]))
        features_tt = ttnn.permute(features_tt, (0, 2, 1))
        features_tt = ttnn.max(features_tt, dim=-1)

        # Handle multi-device: use mesh_composer to aggregate tensor from all devices
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        if num_devices > 1:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
            pooling_features = ttnn.to_torch(features_tt, mesh_composer=mesh_composer)
            pooling_features = pooling_features[:num_pillars]
        else:
            pooling_features = ttnn.to_torch(features_tt)
        ttnn.deallocate(features_tt)
        pooling_features = pooling_features.reshape(-1, 64)

        batched_canvas = []
        bs = int(coors_batch[-1, 0].item()) + 1

        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            flat_indices = cur_coors[:, 1].long() * self.y_l + cur_coors[:, 2].long()

            canvas_flat_tt = ttnn.zeros(
                (self.x_l * self.y_l, self.out_channel),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

            flat_indices_expanded = flat_indices.unsqueeze(1).expand(-1, self.out_channel)
            flat_indices_tt = ttnn.from_torch(
                flat_indices_expanded,
                dtype=ttnn.int32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            cur_features_tt = ttnn.from_torch(
                cur_features,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            canvas_flat_tt = ttnn.scatter(canvas_flat_tt, 0, flat_indices_tt, cur_features_tt)

            canvas = ttnn.view(canvas_flat_tt, (self.x_l, self.y_l, self.out_channel))
            ttnn.deallocate(canvas_flat_tt)
            canvas = ttnn.permute(canvas, (2, 1, 0))
            batched_canvas.append(canvas)

        batched_canvas = ttnn.stack(batched_canvas, dim=0)
        batched_canvas = ttnn.to_memory_config(
            batched_canvas, ttnn.DRAM_MEMORY_CONFIG
        )  # Output ~27MB, too large for L1
        # ttnn.deallocate(canvas)
        ttnn.deallocate(flat_indices_tt)
        ttnn.deallocate(cur_features_tt)
        ttnn.deallocate(offset_pt_center_tt)
        ttnn.deallocate(x_offset_pi_center_tt)
        ttnn.deallocate(y_offset_pi_center_tt)
        ttnn.deallocate(pillars_feature_tt)
        ttnn.deallocate(npoints_per_pillar_tt)
        return batched_canvas
