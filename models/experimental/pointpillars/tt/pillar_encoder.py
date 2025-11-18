import ttnn
import torch


class TtPillarEncoder:
    def __init__(self, device, voxel_size, point_cloud_range, in_channel, out_channel, parameters):
        self.device = device
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        # Access nested parameters from the helper function
        self.conv_weight = parameters["conv"]["weight"]
        self.bn_scale = parameters["conv"]["bn_scale"]
        self.bn_shift = parameters["conv"]["bn_shift"]

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: torch tensor (p1+p2+...+pb, num_points, c)
        coors_batch: torch tensor (p1+p2+...+pb, 4)
        npoints_per_pillar: torch tensor (p1+p2+...+pb,)
        """
        # Convert inputs to TTNN
        pillars_tt = ttnn.from_torch(
            pillars,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 1. Calculate offset to points center
        pillars_xyz = pillars[:, :, :3]
        points_sum = torch.sum(pillars_xyz, dim=1, keepdim=True)
        points_mean = points_sum / npoints_per_pillar[:, None, None]
        offset_pt_center = pillars_xyz - points_mean

        # 2. Calculate offset to pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        # 3. Concatenate features
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        # 4. Apply mask
        voxel_ids = torch.arange(0, pillars.size(1), device=pillars.device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        # Convert to TTNN for conv operation
        features = features.permute(0, 2, 1).contiguous()  # (num_pillars, 9, num_points)
        features_tt = ttnn.from_torch(
            features,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 5. Apply conv1d as linear (treating each pillar independently)
        # Reshape: (num_pillars, 9, num_points) -> (num_pillars * num_points, 9)
        num_pillars, in_ch, num_points = features.shape
        features_reshaped = features.permute(0, 2, 1).reshape(-1, in_ch)

        features_tt = ttnn.from_torch(
            features_reshaped,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Linear operation
        features_tt = ttnn.linear(
            features_tt,
            self.conv_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # Apply batch norm (fused as scale + shift)
        features_tt = ttnn.multiply(features_tt, self.bn_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        features_tt = ttnn.add(features_tt, self.bn_shift, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply ReLU
        features_tt = ttnn.relu(features_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to torch for max pooling and scatter
        features_out = ttnn.to_torch(features_tt)
        features_out = features_out.reshape(num_pillars, num_points, self.out_channel)
        features_out = features_out.permute(0, 2, 1)  # (num_pillars, out_channel, num_points)

        # 6. Max pooling
        pooling_features = torch.max(features_out, dim=-1)[0]

        # 7. Pillar scatter
        batched_canvas = []
        bs = int(coors_batch[-1, 0].item()) + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.bfloat16, device=pillars.device)
            canvas[cur_coors[:, 1].long(), cur_coors[:, 2].long()] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)

        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas
