# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSD512.common import SSD512_L1_SMALL_SIZE, SSD512_NUM_CLASSES, load_torch_model
from models.experimental.SSD512.tt.tt_ssd import TtSSD
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.common.utility_functions import run_for_wormhole_b0


# Creates pipeline-compatible wrapper for SSD512 model
def create_ssd512_pipeline_model(ttnn_model, dtype=ttnn.bfloat16):
    device_ref = ttnn_model.device

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE
        assert l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1

        # Convert from L1 to DRAM memory config for model execution
        input_for_model = ttnn.to_memory_config(l1_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        if input_for_model.layout != ttnn.TILE_LAYOUT:
            input_for_model = ttnn.to_layout(input_for_model, ttnn.TILE_LAYOUT)

        tt_loc_preds, tt_conf_preds = ttnn_model(device_ref, input_for_model)

        loc_tensors = []
        conf_tensors = []
        memory_config = ttnn.DRAM_MEMORY_CONFIG

        for loc_pred in tt_loc_preds:
            if loc_pred.is_sharded():
                loc_pred = ttnn.sharded_to_interleaved(loc_pred, memory_config)
            loc_pred = ttnn.to_memory_config(loc_pred, memory_config)
            if loc_pred.layout != ttnn.ROW_MAJOR_LAYOUT:
                loc_pred = ttnn.to_layout(loc_pred, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
            batch_size = loc_pred.shape[0]
            total_elements = loc_pred.shape[1] * loc_pred.shape[2] * loc_pred.shape[3]
            loc_reshaped = ttnn.experimental.view(loc_pred, (batch_size, total_elements))
            loc_reshaped = ttnn.to_memory_config(loc_reshaped, memory_config)
            loc_tensors.append(loc_reshaped)

        for conf_pred in tt_conf_preds:
            if conf_pred.is_sharded():
                conf_pred = ttnn.sharded_to_interleaved(conf_pred, memory_config)
            conf_pred = ttnn.to_memory_config(conf_pred, memory_config)
            if conf_pred.layout != ttnn.ROW_MAJOR_LAYOUT:
                conf_pred = ttnn.to_layout(conf_pred, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
            batch_size = conf_pred.shape[0]
            total_elements = conf_pred.shape[1] * conf_pred.shape[2] * conf_pred.shape[3]
            conf_reshaped = ttnn.experimental.view(conf_pred, (batch_size, total_elements))
            conf_reshaped = ttnn.to_memory_config(conf_reshaped, memory_config)
            conf_tensors.append(conf_reshaped)

        if len(loc_tensors) > 1:
            loc = ttnn.concat(loc_tensors, dim=1, memory_config=memory_config)
        else:
            loc = loc_tensors[0]

        if len(conf_tensors) > 1:
            conf = ttnn.concat(conf_tensors, dim=1, memory_config=memory_config)
        else:
            conf = conf_tensors[0]

        if loc.layout != ttnn.ROW_MAJOR_LAYOUT:
            loc = ttnn.to_layout(loc, ttnn.ROW_MAJOR_LAYOUT)
        if conf.layout != ttnn.ROW_MAJOR_LAYOUT:
            conf = ttnn.to_layout(conf, ttnn.ROW_MAJOR_LAYOUT)

        loc = ttnn.to_memory_config(loc, ttnn.DRAM_MEMORY_CONFIG)
        conf = ttnn.to_memory_config(conf, ttnn.DRAM_MEMORY_CONFIG)

        return (loc, conf)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SSD512_L1_SMALL_SIZE, "trace_region_size": 10000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize("batch_size, size, expected_compile_time, expected_throughput_fps", [(1, 512, 25.4, 39.3)])
@pytest.mark.models_performance_bare_metal
def test_ssd512_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    dtype = ttnn.bfloat16

    torch_model = load_torch_model(phase="test", size=size, num_classes=SSD512_NUM_CLASSES)

    input_shape = (batch_size, 3, size, size)
    sample_input = torch.randn(input_shape, dtype=torch.float32)
    torch_input = sample_input

    ttnn_model = TtSSD(torch_model, torch_input, device, batch_size)

    ttnn.synchronize_device(device)

    pipeline_model = create_ssd512_pipeline_model(ttnn_model, dtype=dtype)

    sample_input_permuted = sample_input.permute(0, 2, 3, 1)
    sample_input_shape = sample_input_permuted.shape
    batch_size, height, width, channels = sample_input_shape
    total_height = batch_size * height * width

    core_grid = device.core_grid
    num_l1_cores = core_grid.x * core_grid.y

    max_cb_pages = 60000
    tile_height = 32
    l1_alignment = 32
    dtype_size = 2
    max_shard_height = min(total_height // num_l1_cores, max_cb_pages * tile_height)
    max_shard_height = max(tile_height, (max_shard_height // tile_height) * tile_height)

    shard_width_bytes = channels * dtype_size
    padded_shard_width = ((shard_width_bytes + l1_alignment - 1) // l1_alignment) * l1_alignment
    padded_shard_width_channels = padded_shard_width // dtype_size

    if padded_shard_width_channels > channels:
        padding_size = padded_shard_width_channels - channels
        sample_input_padded = torch.nn.functional.pad(
            sample_input_permuted, (0, padding_size), mode="constant", value=0
        )
    else:
        sample_input_padded = sample_input_permuted

    ttnn_input_tensor = ttnn.from_torch(sample_input_padded, device=None, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    l1_shard_height = max_shard_height
    l1_core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))
    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({l1_core_range}),
        [l1_shard_height, padded_shard_width_channels],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec
    )

    dram_grid_size = device.dram_grid_size()
    num_dram_cores = dram_grid_size.x

    min_shard_height = (total_height + num_dram_cores - 1) // num_dram_cores
    dram_shard_height = max(tile_height, ((min_shard_height + tile_height - 1) // tile_height) * tile_height)

    num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height
    while num_shards_needed > num_dram_cores:
        dram_shard_height += tile_height
        num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height

    actual_num_shards = min(num_shards_needed, num_dram_cores)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(actual_num_shards - 1, 0))}),
        [dram_shard_height, padded_shard_width_channels],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    prep_perf_report(
        model_name="ssd512-trace-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )
