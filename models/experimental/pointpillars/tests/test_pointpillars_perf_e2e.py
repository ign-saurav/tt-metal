# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, profiler, run_for_wormhole_b0
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.model.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)


def multi_device_to_torch(tt_tensor, device):
    """Convert ttnn tensor to torch, handling multi-device case."""
    num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    if num_devices > 1:
        original_batch = tt_output.shape[0]
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        result = tt_output.to_torch(mesh_composer=mesh_composer)
        return result[:original_batch]
    return tt_output.to_torch()


def run_pointpillars_pipeline(device, test_infra, num_measurement_iterations):
    """Run the PointPillars model through the pipeline and measure performance."""
    tt_inputs_host = test_infra["host_input_tensor"]
    input_dram_mem_config = test_infra["input_dram_mem_config"]
    input_l1_mem_config = test_infra["input_l1_mem_config"]
    tt_model = test_infra["tt_model"]

    def pointpillars_model_wrapper(l1_input_tensor):
        return tt_model.forward(l1_input_tensor)

    pipeline = create_pipeline_from_config(
        device=device,
        model=pointpillars_model_wrapper,
        config=PipelineConfig(
            use_trace=True,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(
        f"Starting performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra['batch_size']} and num_devices={test_infra['num_devices']}"
    )
    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")

    # Validate last output
    tt_cls, tt_reg, tt_dir = outputs[-1]
    torch_cls = test_infra["torch_cls"]
    torch_reg = test_infra["torch_reg"]
    torch_dir = test_infra["torch_dir"]

    tt_cls_torch = multi_device_to_torch(tt_cls, device).permute(0, 3, 1, 2)
    passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.97)
    logger.info(f"Classification PCC: {pcc_cls}")

    tt_reg_torch = multi_device_to_torch(tt_reg, device).permute(0, 3, 1, 2)
    passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
    logger.info(f"Regression PCC: {pcc_reg}")

    tt_dir_torch = multi_device_to_torch(tt_dir, device).permute(0, 3, 1, 2)
    passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
    logger.info(f"Direction PCC: {pcc_dir}")

    pipeline.cleanup()

    return outputs, passing_cls and passing_reg and passing_dir


def setup_pointpillars_test_infra(device, batch_size_per_device):
    """Setup test infrastructure for PointPillars."""
    torch.manual_seed(0)
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    # Model parameters
    voxel_size = [0.16, 0.16, 4]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    max_num_points = 32
    max_voxels = (16000, 40000)
    nclasses = 3

    # Initialize torch model
    torch_model = PointPillars(
        nclasses=nclasses,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
    )

    # Load pretrained weights
    try:
        checkpoint = torch.load("epoch_160.pth", map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        torch_model.load_state_dict(state_dict)
        logger.info("Successfully loaded pretrained weights")
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, using random weights")

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create test input (point cloud) and get reference outputs
    batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
    torch_cls, torch_reg, torch_dir = torch_model(batched_pts)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Create preprocessor
    preprocessor = PointPillarsPreprocessor(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
        parameters=parameters,
        device=device,
    )

    # Create TTNN model
    tt_model = TtPointPillars(
        nclasses=nclasses,
        parameters=parameters,
        device=device,
    )

    # Preprocess point cloud to pillar features
    pillar_features = preprocessor.forward(batched_pts)
    pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))  # NHWC to NCHW
    pillar_features = ttnn.reshape(
        pillar_features,
        (pillar_features.shape[0], 1, pillar_features.shape[1] * pillar_features.shape[2], pillar_features.shape[3]),
    )

    # Convert to host tensor for pipeline input
    # Handle multi-device: use mesh_composer to aggregate tensor from all devices
    if num_devices > 1:
        original_batch = pillar_features.shape[0]
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        host_input_tensor = ttnn.to_torch(pillar_features, mesh_composer=mesh_composer)
        # Data is replicated, take only first device's result
        host_input_tensor = host_input_tensor[:original_batch]
    else:
        host_input_tensor = ttnn.to_torch(pillar_features)
    host_input_tensor = ttnn.from_torch(
        host_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Set up memory configurations
    input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
        host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )

    input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
    assert host_input_tensor.shape[-2] % input_l1_core_grid.num_cores == 0, "Expecting even sharding on L1 input tensor"
    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(host_input_tensor.shape[2] // input_l1_core_grid.num_cores, host_input_tensor.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Warmup pass to prepare conv weights on device BEFORE trace capture
    warmup_input = ttnn.to_device(host_input_tensor, device, memory_config=input_l1_mem_config)
    _ = tt_model.forward(warmup_input)
    ttnn.deallocate(warmup_input)
    logger.info("Warmup pass complete - weights prepared on device")

    return {
        "host_input_tensor": host_input_tensor,
        "input_dram_mem_config": input_dram_mem_config,
        "input_l1_mem_config": input_l1_mem_config,
        "tt_model": tt_model,
        "torch_cls": torch_cls,
        "torch_reg": torch_reg,
        "torch_dir": torch_dir,
        "batch_size": batch_size,
        "num_devices": num_devices,
    }


def run_perf_e2e_pointpillars(
    device,
    batch_size_per_device,
    expected_inference_throughput,
):
    """Run end-to-end performance test for PointPillars."""
    profiler.clear()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = setup_pointpillars_test_infra(device, batch_size_per_device)

    num_measurement_iterations = 32
    outputs, validation_passed = run_pointpillars_pipeline(device, test_infra, num_measurement_iterations)

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_pointpillars_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=300,
        expected_inference_time=expected_inference_time,
        comments=f"pointpillars_batchsize{batch_size}_devices{num_devices}",
        inference_time_cpu=0.0,
    )

    fps = batch_size / inference_time_avg
    logger.info(
        f"PointPillars batch_size: {batch_size}, inference time (avg): {inference_time_avg:.4f}s, FPS: {fps:.2f}"
    )
    logger.info(f"PointPillars compile time: {compile_time:.2f}s")
    logger.info(f"Expected throughput: {expected_inference_throughput} FPS, Actual: {fps:.2f} FPS")

    assert validation_passed, "Output validation failed"


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "expected_inference_throughput",
    [30],  # Expected FPS for single device (N150)
)
def test_pointpillars_perf_single_device(
    device,
    batch_size_per_device,
    expected_inference_throughput,
):
    run_perf_e2e_pointpillars(
        device,
        batch_size_per_device,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "expected_inference_throughput",
    [60],  # Expected FPS for multi-device N300 (scales ~linearly)
)
def test_pointpillars_perf_multi_device(
    mesh_device,
    batch_size_per_device,
    expected_inference_throughput,
):
    run_perf_e2e_pointpillars(
        mesh_device,
        batch_size_per_device,
        expected_inference_throughput,
    )
