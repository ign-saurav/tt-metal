# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
)


class TtTopDown:
    def __init__(
        self, device, parameters, perception_output_features=512, bev_features_channels=64, bev_upsample_factor=2
    ):
        self.device = device
        self.perception_output_features = perception_output_features
        self.bev_features_channels = bev_features_channels
        self.bev_upsample_factor = bev_upsample_factor
        self.dtype = ttnn.bfloat16

        def get_conv_params(name):
            if isinstance(parameters, dict):
                return parameters[name]
            return getattr(parameters, name)

        # Assume a fixed input resolution for P5 (e.g., 8x8) and batch size 1
        base_height = 8
        base_width = 8
        batch_size = 1

        self.c5_conv = self._create_conv_layer(
            conv_params=get_conv_params("c5_conv"),
            batch_size=batch_size,
            input_height=base_height,
            input_width=base_width,
            in_channels=self.perception_output_features,
            out_channels=self.bev_features_channels,
        )

        up_height = base_height * self.bev_upsample_factor
        up_width = base_width * self.bev_upsample_factor
        self.up_conv5 = self._create_conv_layer(
            conv_params=get_conv_params("up_conv5"),
            batch_size=batch_size,
            input_height=up_height,
            input_width=up_width,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
        )

        up_height2 = up_height * self.bev_upsample_factor
        up_width2 = up_width * self.bev_upsample_factor
        self.up_conv4 = self._create_conv_layer(
            conv_params=get_conv_params("up_conv4"),
            batch_size=batch_size,
            input_height=up_height2,
            input_width=up_width2,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
        )

        up_height3 = up_height2 * self.bev_upsample_factor
        up_width3 = up_width2 * self.bev_upsample_factor
        self.up_conv3 = self._create_conv_layer(
            conv_params=get_conv_params("up_conv3"),
            batch_size=batch_size,
            input_height=up_height3,
            input_width=up_width3,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
        )

    def _create_conv_layer(
        self,
        conv_params,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
    ):
        weight = conv_params["weight"]
        bias = conv_params["bias"]
        if bias is not None and len(bias.shape) == 1:
            bias = ttnn.reshape(bias, (1, 1, 1, bias.shape[0]))

        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            weight=weight,
            bias=bias,
            activation=None,
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=False,
            packer_l1_acc=False,
            config_tensors_in_dram=True,
        )

        return TtConv2d(config, device=self.device)

    def _convert_interleaved_to_sharded_for_conv(self, tensor, height, width, channels):
        """Convert interleaved tensor to sharded layout suitable for conv2d.

        Creates a height-sharded memory config which is commonly used for conv2d operations.
        """
        # Get tensor shape and flatten if needed
        if len(tensor.shape) == 4:
            batch, h, w, c = tensor.shape
            # Flatten tensor for conv2d: (batch, height, width, channels) -> (1, 1, batch*height*width, channels)
            tensor_flat = ttnn.reshape(tensor, (1, 1, batch * h * w, c))
            total_height = batch * h * w
            shard_width = c
        else:
            # Already flattened
            tensor_flat = tensor
            total_height = tensor.shape[2]  # batch * height * width
            shard_width = tensor.shape[3]  # channels

        # Ensure tensor is interleaved before converting to sharded
        if tensor_flat.is_sharded():
            tensor_flat = ttnn.sharded_to_interleaved(tensor_flat, ttnn.L1_MEMORY_CONFIG)

        # Ensure tensor is in TILE_LAYOUT for sharding (interleaved_to_sharded may require this)
        if tensor_flat.layout != ttnn.TILE_LAYOUT:
            tensor_flat = ttnn.to_layout(tensor_flat, ttnn.TILE_LAYOUT)

        # Calculate shard shape for height sharding with tile alignment
        # For TILE layout, shard dimensions must be multiples of 32
        TILE_HEIGHT = 32
        TILE_WIDTH = 32

        compute_grid_size = self.device.compute_with_storage_grid_size()
        num_cores = compute_grid_size.x * compute_grid_size.y

        # For height sharding with TILE layout, pad total_height to be a multiple of (num_cores * TILE_HEIGHT)
        # This ensures each shard height is a multiple of TILE_HEIGHT
        total_height_padded = ((total_height + num_cores * TILE_HEIGHT - 1) // (num_cores * TILE_HEIGHT)) * (
            num_cores * TILE_HEIGHT
        )
        shard_height = total_height_padded // num_cores

        # Ensure shard_width is a multiple of TILE_WIDTH
        shard_width_padded = ((shard_width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH

        # Pad tensor if needed to match padded dimensions
        total_height_needed = shard_height * num_cores
        if total_height_needed > total_height or shard_width_padded > shard_width:
            # Calculate padding needed
            pad_height = total_height_needed - total_height
            pad_width = shard_width_padded - shard_width
            if pad_height > 0 or pad_width > 0:
                # Ensure padding is tile-aligned for TILE_LAYOUT
                # Pad format: tuple of (start, end) for each dimension
                # For shape (1, 1, height, width), we pad dims 2 and 3
                # Since we're in TILE_LAYOUT, padding must be multiples of 32
                pad_height_aligned = ((pad_height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
                pad_width_aligned = ((pad_width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH

                # Update shard dimensions to match aligned padding
                if pad_height_aligned != pad_height:
                    total_height_needed = total_height + pad_height_aligned
                    shard_height = total_height_needed // num_cores
                if pad_width_aligned != pad_width:
                    shard_width_padded = shard_width + pad_width_aligned

                # Pad the tensor: padding format is ((start_dim0, end_dim0), (start_dim1, end_dim1), ...)
                # For shape (1, 1, height, width), we pad dims 2 and 3
                tensor_flat = ttnn.pad(
                    tensor_flat, ((0, 0), (0, 0), (0, pad_height_aligned), (0, pad_width_aligned)), 0.0
                )

        # Create core grid (use all available cores)
        core_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1),
                )
            }
        )

        shard_spec = ttnn.ShardSpec(
            core_grid,
            (shard_height, shard_width_padded),
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        # Convert to sharded using to_memory_config which handles the conversion properly
        return ttnn.to_memory_config(tensor_flat, memory_config)

    def __call__(self, x):
        batch_size, input_height, input_width, _ = x.shape

        # Lateral conv: 512 -> 64 channels (c5_conv)
        p5_flat, (p5_h, p5_w) = self.c5_conv(x, return_output_dim=True)
        p5_flat = ttnn.sharded_to_interleaved(p5_flat, ttnn.DRAM_MEMORY_CONFIG)
        p5_flat = ttnn.relu(p5_flat)
        p5_flat = ttnn.to_layout(p5_flat, ttnn.ROW_MAJOR_LAYOUT)
        p5 = ttnn.reshape(p5_flat, (batch_size, p5_h, p5_w, self.bev_features_channels))

        # Ensure DRAM + ROW_MAJOR before upsampling
        p5 = ttnn.to_memory_config(p5, ttnn.DRAM_MEMORY_CONFIG)
        if p5.layout == ttnn.TILE_LAYOUT:
            p5 = ttnn.to_layout(p5, ttnn.ROW_MAJOR_LAYOUT)

        # First upsample stage
        p5_upsampled = ttnn.upsample(p5, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height = input_height * self.bev_upsample_factor
        up_width = input_width * self.bev_upsample_factor

        # Convert upsampled tensor to sharded before conv2d
        p5_upsampled_sharded = self._convert_interleaved_to_sharded_for_conv(
            p5_upsampled, up_height, up_width, self.bev_features_channels
        )

        # Second conv on upsampled P5
        p4_flat, (p4_h, p4_w) = self.up_conv5(p5_upsampled_sharded, return_output_dim=True)
        p4_flat = ttnn.sharded_to_interleaved(p4_flat, ttnn.DRAM_MEMORY_CONFIG)
        p4_flat = ttnn.relu(p4_flat)
        p4_flat = ttnn.to_layout(p4_flat, ttnn.ROW_MAJOR_LAYOUT)

        # Slice to remove any padding before reshaping
        expected_flat_size = batch_size * p4_h * p4_w
        if p4_flat.shape[2] > expected_flat_size:
            p4_flat = ttnn.slice(p4_flat, [0, 0, 0, 0], [1, 1, expected_flat_size, self.bev_features_channels])

        p4 = ttnn.reshape(p4_flat, (batch_size, p4_h, p4_w, self.bev_features_channels))

        p4 = ttnn.to_memory_config(p4, ttnn.DRAM_MEMORY_CONFIG)
        if p4.layout == ttnn.TILE_LAYOUT:
            p4 = ttnn.to_layout(p4, ttnn.ROW_MAJOR_LAYOUT)

        # Second upsample stage
        p4_upsampled = ttnn.upsample(p4, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height2 = up_height * self.bev_upsample_factor
        up_width2 = up_width * self.bev_upsample_factor

        # Convert upsampled tensor to sharded before conv2d
        p4_upsampled_sharded = self._convert_interleaved_to_sharded_for_conv(
            p4_upsampled, up_height2, up_width2, self.bev_features_channels
        )

        # Third conv on upsampled P4
        p3_flat, (p3_h, p3_w) = self.up_conv4(p4_upsampled_sharded, return_output_dim=True)
        p3_flat = ttnn.sharded_to_interleaved(p3_flat, ttnn.DRAM_MEMORY_CONFIG)
        p3_flat = ttnn.relu(p3_flat)
        p3_flat = ttnn.to_layout(p3_flat, ttnn.ROW_MAJOR_LAYOUT)

        # Slice to remove any padding before reshaping
        expected_flat_size = batch_size * p3_h * p3_w
        if p3_flat.shape[2] > expected_flat_size:
            p3_flat = ttnn.slice(p3_flat, [0, 0, 0, 0], [1, 1, expected_flat_size, self.bev_features_channels])

        p3 = ttnn.reshape(p3_flat, (batch_size, p3_h, p3_w, self.bev_features_channels))

        p3 = ttnn.to_memory_config(p3, ttnn.DRAM_MEMORY_CONFIG)
        if p3.layout == ttnn.TILE_LAYOUT:
            p3 = ttnn.to_layout(p3, ttnn.ROW_MAJOR_LAYOUT)

        # Third upsample stage
        p3_upsampled = ttnn.upsample(p3, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height3 = up_height2 * self.bev_upsample_factor
        up_width3 = up_width2 * self.bev_upsample_factor

        # Convert upsampled tensor to sharded before conv2d
        p3_upsampled_sharded = self._convert_interleaved_to_sharded_for_conv(
            p3_upsampled, up_height3, up_width3, self.bev_features_channels
        )

        # Fourth conv on upsampled P3
        p2_flat, (p2_h, p2_w) = self.up_conv3(p3_upsampled_sharded, return_output_dim=True)
        p2_flat = ttnn.sharded_to_interleaved(p2_flat, ttnn.DRAM_MEMORY_CONFIG)
        p2_flat = ttnn.relu(p2_flat)
        p2_flat = ttnn.to_layout(p2_flat, ttnn.ROW_MAJOR_LAYOUT)

        # Slice to remove any padding before reshaping
        expected_flat_size = batch_size * p2_h * p2_w
        if p2_flat.shape[2] > expected_flat_size:
            p2_flat = ttnn.slice(p2_flat, [0, 0, 0, 0], [1, 1, expected_flat_size, self.bev_features_channels])

        p2 = ttnn.reshape(p2_flat, (batch_size, p2_h, p2_w, self.bev_features_channels))

        p2 = ttnn.to_memory_config(p2, ttnn.DRAM_MEMORY_CONFIG)
        p3 = ttnn.to_memory_config(p3, ttnn.DRAM_MEMORY_CONFIG)
        p4 = ttnn.to_memory_config(p4, ttnn.DRAM_MEMORY_CONFIG)
        p5 = ttnn.to_memory_config(p5, ttnn.DRAM_MEMORY_CONFIG)

        return p2, p3, p4, p5
