# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.BevDepth.common import (
    create_reference_inputs,
)
from models.experimental.BevDepth.tt.custom_preprocessing import (
    prepare_all_parameters_from_reference,
)
from models.experimental.BevDepth.tt.ttnn_bevdepth import TtBEVDepth
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from models.common.utility_functions import run_for_wormhole_b0


def create_bevdepth_pipeline_model(ttnn_model, mats_dict, img_h, img_w):
    """
    Create a pipeline model function for BevDepth.
    Processes real input from pipeline, converting it to the expected format.
    """

    def run(input_imgs_ttnn):
        input_imgs_torch = ttnn.to_torch(input_imgs_ttnn)
        batch_size, num_sweeps, num_cameras = mats_dict["sensor2ego_mats"].shape[:3]

        input_imgs_torch = input_imgs_torch.squeeze(0).squeeze(0)
        total_pixels, padded_channels = input_imgs_torch.shape
        num_images = batch_size * num_sweeps * num_cameras

        input_imgs_torch = input_imgs_torch.reshape(num_images, img_h, img_w, padded_channels)
        input_imgs_torch = input_imgs_torch.permute(0, 3, 1, 2)
        input_imgs_torch = input_imgs_torch[:, :3, :, :]
        input_imgs_torch = input_imgs_torch.reshape(batch_size, num_sweeps, num_cameras, 3, img_h, img_w)

        ttnn_output = ttnn_model(input_imgs_torch, mats_dict)

        output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
        output_tensors = []

        for task_idx in range(len(ttnn_output)):
            for key in output_keys:
                ttnn_tensor, _ = ttnn_output[task_idx][key]
                if ttnn_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
                ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, ttnn.DRAM_MEMORY_CONFIG)
                output_tensors.append(ttnn_tensor)

        return tuple(output_tensors)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 98304,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, num_sweeps, num_cameras, img_h, img_w, expected_compile_time, expected_throughput_fps",
    [(1, 2, 1, 256, 704, 208, 0.55)],
)
@pytest.mark.models_performance_bare_metal
def test_bevdepth_e2e_performant(
    device,
    device_params,
    num_iterations,
    batch_size,
    num_sweeps,
    num_cameras,
    img_h,
    img_w,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    """
    Test BevDepth end-to-end performance with Pipeline API (2CQ, no trace).
    Uses 1 camera to avoid L1 memory limitations when processing through pipeline.
    """
    torch.manual_seed(0)

    logger.info("Building BevDepth model...")
    params, _ = prepare_all_parameters_from_reference(device)

    lss_conf = {
        "x_bound": [-51.2, 51.2, 0.8],
        "y_bound": [-51.2, 51.2, 0.8],
        "z_bound": [-5.0, 3.0, 0.2],
        "d_bound": [2.0, 58.0, 0.5],
        "final_dim": [img_h, img_w],
        "downsample_factor": 16,
        "output_channels": 80,
    }

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "batch_size": batch_size,
        "neck_in_channels": [256, 512, 1024, 2048],
        "neck_out_channels": [128, 128, 128, 128],
        "neck_upsample_strides": [0.25, 0.5, 1, 2],
        "depthnet_in_channels": 512,
        "depthnet_mid_channels": 512,
        "depthnet_context_channels": 80,
        "depthnet_depth_channels": 112,
    }

    ttnn_model = TtBEVDepth(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        head_parameters=params["head"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    torch_input_imgs, mats_dict = create_reference_inputs(
        batch_size=batch_size, num_sweeps=num_sweeps, num_cameras=num_cameras, img_h=img_h, img_w=img_w
    )

    logger.info("Creating pipeline model...")
    pipeline_model = create_bevdepth_pipeline_model(ttnn_model, mats_dict, img_h, img_w)

    logger.info("Preparing input tensor...")
    batch_size, num_sweeps, num_cameras, channels, height, width = torch_input_imgs.shape

    imgs_flat = torch_input_imgs.flatten(0, 2)
    imgs_nhwc = imgs_flat.permute(0, 2, 3, 1)

    page_size = 16
    padded_channels = ((channels + page_size - 1) // page_size) * page_size

    if padded_channels > channels:
        padding = torch.zeros(
            imgs_nhwc.shape[0],
            imgs_nhwc.shape[1],
            imgs_nhwc.shape[2],
            padded_channels - channels,
            dtype=imgs_nhwc.dtype,
        )
        imgs_nhwc = torch.cat([imgs_nhwc, padding], dim=3)

    imgs_nhwc_2d = imgs_nhwc.reshape(1, 1, -1, padded_channels)

    ttnn_input_tensor = ttnn.from_torch(
        imgs_nhwc_2d,
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    input_shape = imgs_nhwc_2d.shape
    core_grid = device.core_grid
    num_l1_cores = core_grid.x * core_grid.y

    dram_grid_size = device.dram_grid_size()
    num_dram_cores = dram_grid_size.x
    assert dram_grid_size.y == 1, "Only 1D DRAM grid is supported"

    tile_height = 32
    dram_shard_height = (input_shape[-2] + num_dram_cores - 1) // num_dram_cores
    dram_shard_height = max(tile_height, ((dram_shard_height + tile_height - 1) // tile_height) * tile_height)
    dram_shard_width = input_shape[-1]

    actual_num_dram_shards = (input_shape[-2] + dram_shard_height - 1) // dram_shard_height
    actual_num_dram_shards = min(actual_num_dram_shards, num_dram_cores)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(actual_num_dram_shards - 1, 0))}),
        [dram_shard_height, dram_shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_shard_height = (input_shape[-2] + num_l1_cores - 1) // num_l1_cores
    l1_shard_height = max(tile_height, ((l1_shard_height + tile_height - 1) // tile_height) * tile_height)
    l1_shard_width = input_shape[-1]

    actual_num_l1_shards = (input_shape[-2] + l1_shard_height - 1) // l1_shard_height
    actual_num_l1_shards = min(actual_num_l1_shards, num_l1_cores)

    l1_core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))
    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({l1_core_range}),
        [l1_shard_height, l1_shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec
    )

    # TODO: 2CQ with trace as currently the deformconv2d is not supported in TTNN. Issue #34509 and PR #34940 are still open.
    logger.info("Configuring pipeline (2CQ without trace)...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start
    logger.info(f"Compilation time: {compile_time:.2f}s")

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    _ = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    throughput_fps = num_iterations * batch_size / (end - start)

    logger.info(f"Average model time: {1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance: {throughput_fps:.2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="bevdepth-notrace-2cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}-sweeps_{num_sweeps}-cameras_{num_cameras}-img_{img_h}x{img_w}",
    )

    logger.info("Performance test completed!")
