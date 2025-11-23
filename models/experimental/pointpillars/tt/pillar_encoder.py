import ttnn


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
        pillars_x_tt = ttnn.from_torch(
            pillars[:, :, :1],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pillars_y_tt = ttnn.from_torch(
            pillars[:, :, 1:2],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pillars_features_tt = ttnn.from_torch(
            pillars[:, :, 2:],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pillars_xyz = ttnn.from_torch(
            pillars[:, :, :3],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Convert inputs to TTNN
        npoints_per_pillar_tt = ttnn.from_torch(
            npoints_per_pillar,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        coors_x_indices_tt = ttnn.from_torch(
            coors_batch[:, 1:2].unsqueeze(1),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        coors_y_indices_tt = ttnn.from_torch(
            coors_batch[:, 2:3].unsqueeze(1),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 1. Calculate offset to points center
        points_sum = ttnn.sum(pillars_xyz, dim=1, keepdim=True)
        npoints_per_pillar_tt_unsqueezed = ttnn.unsqueeze(npoints_per_pillar_tt, dim=1)
        npoints_per_pillar_tt_unsqueezed = ttnn.unsqueeze(npoints_per_pillar_tt_unsqueezed, dim=2)
        points_mean = ttnn.div(points_sum, npoints_per_pillar_tt_unsqueezed)
        offset_pt_center = pillars_xyz - points_mean

        # 2. Calculate offset to pillar center
        x_offset_pi_center = pillars_x_tt - (coors_x_indices_tt * self.vx + self.x_offset)
        y_offset_pi_center = pillars_y_tt - (coors_y_indices_tt * self.vy + self.y_offset)

        # 3. Concatenate features
        features_tt = ttnn.concat(
            [
                x_offset_pi_center,
                y_offset_pi_center,
                pillars_features_tt,
                offset_pt_center,
                x_offset_pi_center,
                y_offset_pi_center,
            ],
            dim=-1,
        )

        # 4. Apply mask
        voxel_ids = ttnn.arange(0, pillars.size(1), 1, device=self.device, dtype=ttnn.bfloat16)
        mask = ttnn.unsqueeze(voxel_ids, dim=1) < ttnn.unsqueeze(npoints_per_pillar_tt, dim=0)
        mask = ttnn.unsqueeze(ttnn.permute(mask, (1, 0)), dim=2)
        features_tt *= mask

        # Convert to TTNN for conv operation
        features_tt = ttnn.permute(features_tt, (0, 2, 1))

        # 5. Apply conv1d as linear (treating each pillar independently)
        # Reshape: (num_pillars, 9, num_points) -> (num_pillars * num_points, 9)
        num_pillars, in_ch, num_points = features_tt.shape
        features_reshaped = ttnn.reshape(ttnn.permute(features_tt, (0, 2, 1)), (num_pillars * num_points, in_ch))

        # Linear operation
        features_tt = ttnn.linear(
            features_reshaped,
            self.conv_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # Apply batch norm (fused as scale + shift)
        features_tt = ttnn.multiply(features_tt, self.bn_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        features_tt = ttnn.add(features_tt, self.bn_shift, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply ReLU
        features_tt = ttnn.relu(features_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # 6. Max pooling
        features_tt = ttnn.max_pool2d(  # Shape([1, 1, 6169, 64])
            input_tensor=features_tt,
            batch_size=int(coors_batch[-1, 0].item()) + 1,
            input_h=num_pillars,
            input_w=num_points,
            channels=self.out_channel,
            kernel_size=[1, 32],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        pooling_features = ttnn.to_torch(features_tt)
        ttnn.deallocate(features_tt)
        pooling_features = pooling_features.reshape(-1, 64)

        # 7. Pillar scatter
        batched_canvas = []
        bs = int(coors_batch[-1, 0].item()) + 1

        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            # Convert 2D coordinates to 1D indices
            flat_indices = cur_coors[:, 1].long() * self.y_l + cur_coors[:, 2].long()

            # Create flattened canvas
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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            cur_features_tt = ttnn.from_torch(
                cur_features,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Perform scatter (returns new tensor, not in-place)
            canvas_flat_tt = ttnn.scatter(canvas_flat_tt, 0, flat_indices_tt, cur_features_tt)

            # ttnn.deallocate(canvas_flat_tt)
            # Reshape back to 2D
            canvas = ttnn.view(canvas_flat_tt, (self.x_l, self.y_l, self.out_channel))
            canvas = ttnn.permute(canvas, (2, 1, 0))
            batched_canvas.append(canvas)

        batched_canvas = ttnn.stack(batched_canvas, dim=0)
        return batched_canvas
