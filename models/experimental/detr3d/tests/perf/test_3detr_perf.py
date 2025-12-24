# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.detr3d.runner.performant_runner_infra import Detr3dPerformanceRunnerInfra
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)


def run_model_pipeline(device, test_infra, num_measurement_iterations):
    """Run DETR3D model with trace+2CQ pipeline"""
    # Setup input for DETR3D - handle dictionary format
    ttnn_dict = test_infra.setup_device_input()

    # Use point_clouds as the main input tensor for pipeline
    tt_inputs_host = ttnn_dict["point_clouds"]

    # Create proper sharded DRAM memory config for point cloud input
    dram_input_mem_config = get_memory_config_for_persistent_dram_tensor(
        tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )

    # Get device grid size and create appropriate core grid
    device_grid_size = device.compute_with_storage_grid_size()
    import pdb

    pdb.set_trace()
    height_dim = tt_inputs_host.shape[-2]

    # Find optimal core grid that fits device and divides height evenly
    max_cores = device_grid_size.x * device_grid_size.y

    # Start with full grid and adjust if needed
    num_cores = max_cores
    while num_cores > 1 and height_dim % num_cores != 0:
        num_cores -= 1

    # Create core grid from num_cores
    if num_cores <= device_grid_size.x:
        core_grid_x = num_cores
        core_grid_y = 1
    else:
        core_grid_x = device_grid_size.x
        core_grid_y = num_cores // core_grid_x
        # Ensure we don't exceed device grid
        if core_grid_y > device_grid_size.y:
            core_grid_y = device_grid_size.y
            core_grid_x = num_cores // core_grid_y

    input_l1_core_grid = ttnn.CoreGrid(x=core_grid_x, y=core_grid_y)

    l1_input_mem_config = ttnn.create_sharded_memory_config(
        shape=(height_dim // input_l1_core_grid.num_cores, tt_inputs_host.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def model_wrapper(input_tensor):
        # Reconstruct input dictionary with the tensor from pipeline
        input_dict = ttnn_dict.copy()
        input_dict["point_clouds"] = input_tensor
        test_infra.run(input_dict)
        return test_infra.cls_logits  # Return primary output for pipeline

    pipeline = create_pipeline_from_config(
        device=device,
        model=model_wrapper,
        config=PipelineConfig(
            use_trace=True,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=dram_input_mem_config,
        l1_input_memory_config=l1_input_mem_config,
    )

    logger.info(f"Running DETR3D warmup with input shape {list(tt_inputs_host.shape)}")
    logger.info(f"Using core grid: {input_l1_core_grid.x}x{input_l1_core_grid.y}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(
        f"Starting DETR3D performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )
    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")

    # Validate all outputs for each iteration
    for i, output in enumerate(outputs):
        # Set the output in test_infra for validation
        test_infra.cls_logits = output
        test_infra.validate()
        logger.info(f"Output {i} validation passed")

    pipeline.cleanup()
    return outputs


def run_perf_e2e_detr3d(
    device,
    batch_size_per_device,
    model_location_generator,
    input_shape,
    expected_inference_throughput,
    encoder_only=False,
):
    """Run end-to-end performance test for DETR3D"""
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = Detr3dPerformanceRunnerInfra(
        device=device,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
        input_shape=input_shape,
        encoder_only=encoder_only,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
        load_real_input=True,
    )

    num_measurement_iterations = 32
    run_model_pipeline(device, test_infra, num_measurement_iterations)

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_detr3d_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=240,
        expected_inference_time=expected_inference_time,
        comments=f"input_shape_{input_shape}_batchsize{batch_size}_encoder_only_{encoder_only}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"DETR3D input_shape={input_shape} batch_size: {batch_size}, inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"DETR3D compile time: {compile_time} s")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("encoder_only", (False,))
@pytest.mark.parametrize(
    "input_shape, expected_inference_throughput",
    [((1, 20000, 3), 50)],
)
def test_detr3d_perf_single_device(
    device,
    batch_size_per_device,
    model_location_generator,
    input_shape,
    expected_inference_throughput,
    encoder_only,
):
    run_perf_e2e_detr3d(
        device,
        batch_size_per_device,
        model_location_generator,
        input_shape,
        expected_inference_throughput,
        encoder_only,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("encoder_only", (False,))
@pytest.mark.parametrize(
    "input_shape, expected_inference_throughput",
    [((1, 20000, 3), 90)],
)
def test_detr3d_perf_multi_device(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
    input_shape,
    expected_inference_throughput,
    encoder_only,
):
    run_perf_e2e_detr3d(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        input_shape,
        expected_inference_throughput,
        encoder_only,
    )
