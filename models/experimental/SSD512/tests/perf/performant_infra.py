# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.SSD512.reference.ssd import build_ssd

_TILE_HEIGHT = 32  # 32x32 tile size
_L1_ALIGNMENT = 32  # L1 memory alignment
_DTYPE_SIZE_BYTES = 2  # bfloat16
_MAX_CB_PAGES = 60000  # Circular Buffer page limit, max is 65535


class SSD512PerformantTestInfra:
    def __init__(self, device, ttnn_model, dtype=ttnn.bfloat16):
        self.device = device
        self.ttnn_model = ttnn_model
        self.dtype = dtype
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

    def __call__(self, l1_input_tensor):
        """Run pipeline model forward pass."""
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE
        assert l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1

        input_for_model = ttnn.to_memory_config(l1_input_tensor, self.memory_config)
        if input_for_model.layout != ttnn.TILE_LAYOUT:
            input_for_model = ttnn.to_layout(input_for_model, ttnn.TILE_LAYOUT)

        tt_loc_preds, tt_conf_preds = self.ttnn_model(self.device, input_for_model)

        loc = self.reshape_prediction_tensors(tt_loc_preds, self.memory_config)
        conf = self.reshape_prediction_tensors(tt_conf_preds, self.memory_config)

        if loc.layout != ttnn.ROW_MAJOR_LAYOUT:
            loc = ttnn.to_layout(loc, ttnn.ROW_MAJOR_LAYOUT)
        if conf.layout != ttnn.ROW_MAJOR_LAYOUT:
            conf = ttnn.to_layout(conf, ttnn.ROW_MAJOR_LAYOUT)

        loc = ttnn.to_memory_config(loc, self.memory_config)
        conf = ttnn.to_memory_config(conf, self.memory_config)

        return (loc, conf)

    def reshape_prediction_tensors(self, pred_tensors, memory_config):
        """Reshape and concatenate prediction tensors from multiple feature scales."""
        reshaped_tensors = []
        for pred in pred_tensors:
            if pred.is_sharded():
                pred = ttnn.sharded_to_interleaved(pred, memory_config)
            pred = ttnn.to_memory_config(pred, memory_config)
            if pred.layout != ttnn.ROW_MAJOR_LAYOUT:
                pred = ttnn.to_layout(pred, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
            batch_size = pred.shape[0]
            total_elements = pred.shape[1] * pred.shape[2] * pred.shape[3]
            pred_reshaped = ttnn.experimental.view(pred, (batch_size, total_elements))
            pred_reshaped = ttnn.to_memory_config(pred_reshaped, memory_config)
            reshaped_tensors.append(pred_reshaped)

        if len(reshaped_tensors) > 1:
            return ttnn.concat(reshaped_tensors, dim=1, memory_config=memory_config)
        return reshaped_tensors[0]

    def create_pipeline_memory_configs(self, torch_input):
        """Create L1 and DRAM memory configs for pipeline executor from input tensor."""
        input_permuted = torch_input.permute(0, 2, 3, 1)
        batch_size, height, width, channels = input_permuted.shape
        total_height = batch_size * height * width

        shard_width_bytes = channels * _DTYPE_SIZE_BYTES
        padded_shard_width_bytes = self._align_to(shard_width_bytes, _L1_ALIGNMENT)
        padded_shard_width_channels = padded_shard_width_bytes // _DTYPE_SIZE_BYTES

        if padded_shard_width_channels > channels:
            padding_size = padded_shard_width_channels - channels
            input_padded = torch.nn.functional.pad(input_permuted, (0, padding_size), mode="constant", value=0)
        else:
            input_padded = input_permuted

        ttnn_input_tensor = ttnn.from_torch(
            input_padded, device=None, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        l1_input_memory_config = self._create_l1_memory_config(self.device, total_height, padded_shard_width_channels)
        dram_input_memory_config = self._create_dram_memory_config(
            self.device, total_height, padded_shard_width_channels
        )

        return ttnn_input_tensor, l1_input_memory_config, dram_input_memory_config

    def _align_to(self, value, alignment):
        """Round up value to nearest multiple of alignment."""
        return ((value + alignment - 1) // alignment) * alignment

    def _round_up_to_tile(self, value):
        """Round up value to nearest multiple of tile height."""
        return max(_TILE_HEIGHT, ((value + _TILE_HEIGHT - 1) // _TILE_HEIGHT) * _TILE_HEIGHT)

    def _create_height_shard_spec(self, core_range, shard_height, shard_width):
        """Create a height-sharded ShardSpec."""
        return ttnn.ShardSpec(
            ttnn.CoreRangeSet({core_range}),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    def _create_l1_memory_config(self, device, total_height, padded_shard_width_channels):
        """Create L1 memory config for pipeline executor."""
        core_grid = device.core_grid
        num_l1_cores = core_grid.x * core_grid.y

        max_shard_height = min(total_height // num_l1_cores, _MAX_CB_PAGES * _TILE_HEIGHT)
        l1_shard_height = self._round_up_to_tile(max_shard_height)

        l1_core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))
        l1_shard_spec = self._create_height_shard_spec(l1_core_range, l1_shard_height, padded_shard_width_channels)

        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    def _create_dram_memory_config(self, device, total_height, padded_shard_width_channels):
        """Create DRAM memory config for pipeline executor."""
        dram_grid_size = device.dram_grid_size()
        num_dram_cores = dram_grid_size.x

        min_shard_height = (total_height + num_dram_cores - 1) // num_dram_cores
        dram_shard_height = self._round_up_to_tile(min_shard_height)

        num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height
        while num_shards_needed > num_dram_cores:
            dram_shard_height += self._TILE_HEIGHT
            num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height

        actual_num_shards = min(num_shards_needed, num_dram_cores)

        dram_core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(actual_num_shards - 1, 0))
        dram_shard_spec = self._create_height_shard_spec(
            dram_core_range, dram_shard_height, padded_shard_width_channels
        )

        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

    @staticmethod
    def load_torch_model(phase="test", size=512, num_classes=21):
        """Load the SSD512 torch model."""
        torch_model = build_ssd(phase, size=size, num_classes=num_classes)
        torch_model.eval()
        return torch_model
