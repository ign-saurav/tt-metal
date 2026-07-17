# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, profiler, run_for_wormhole_b0
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from models.experimental.pointpillars.common import (
    VOXEL_SIZE,
    POINT_CLOUD_RANGE,
    MAX_NUM_POINTS,
    MAX_VOXELS,
    NCLASSES,
    load_checkpoint,
    download_checkpoint,
    multi_device_to_torch,
)


def run_pointpillars_pipeline(device, test_infra, num_iterations):
    """Run PointPillars through the pipeline and validate outputs against PyTorch reference."""
    tt_inputs_host = test_infra["host_input_tensor"]
    tt_model = test_infra["tt_model"]

    pipeline = create_pipeline_from_config(
        device=device,
        model=lambda x: tt_model.forward(x),
        config=PipelineConfig(
            use_trace=True,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=test_infra["input_dram_mem_config"],
        l1_input_memory_config=test_infra["input_l1_mem_config"],
    )

    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue([tt_inputs_host] * num_iterations).pop_all()
    profiler.end("run_model_pipeline_2cqs")

    tt_cls, tt_reg, tt_dir = outputs[-1]
    tt_cls_torch = multi_device_to_torch(tt_cls, device).permute(0, 3, 1, 2)
    tt_reg_torch = multi_device_to_torch(tt_reg, device).permute(0, 3, 1, 2)
    tt_dir_torch = multi_device_to_torch(tt_dir, device).permute(0, 3, 1, 2)

    passing_cls, pcc_cls = comp_pcc(test_infra["torch_cls"], tt_cls_torch, 0.97)
    passing_reg, pcc_reg = comp_pcc(test_infra["torch_reg"], tt_reg_torch, 0.99)
    passing_dir, pcc_dir = comp_pcc(test_infra["torch_dir"], tt_dir_torch, 0.99)

    logger.info(f"PCC - Classification: {pcc_cls}, Regression: {pcc_reg}, Direction: {pcc_dir}")

    pipeline.cleanup()
    return outputs, passing_cls and passing_reg and passing_dir


def setup_pointpillars_test_infra(device, batch_size_per_device):
    """Initialize PyTorch and TTNN models, preprocess inputs, and configure memory layouts."""
    torch.manual_seed(0)
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    torch_model = PointPillars(
        nclasses=NCLASSES,
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=MAX_NUM_POINTS,
        max_voxels=MAX_VOXELS,
    )
    checkpoint_dir = "models/experimental/pointpillars/resources/checkpoint"
    checkpoint_path = download_checkpoint(checkpoint_dir)
    state_dict = load_checkpoint(checkpoint_path)
    if state_dict is not None:
        torch_model.load_state_dict(state_dict)

    torch_model = torch_model.to(dtype=torch.bfloat16).eval()

    batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
    torch_cls, torch_reg, torch_dir = torch_model(batched_pts)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    preprocessor = PointPillarsPreprocessor(
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=MAX_NUM_POINTS,
        max_voxels=MAX_VOXELS,
        parameters=parameters,
        device=device,
    )

    tt_model = TtPointPillars(
        nclasses=NCLASSES,
        parameters=parameters,
        device=device,
    )

    pillar_features = preprocessor.forward(batched_pts)
    pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))
    pillar_features = ttnn.reshape(
        pillar_features,
        (pillar_features.shape[0], 1, pillar_features.shape[1] * pillar_features.shape[2], pillar_features.shape[3]),
    )

    if num_devices > 1:
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        host_input_tensor = ttnn.to_torch(pillar_features, mesh_composer=mesh_composer)
        host_input_tensor = host_input_tensor[: pillar_features.shape[0]]
    else:
        host_input_tensor = ttnn.to_torch(pillar_features)

    host_input_tensor = ttnn.from_torch(host_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
        host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )

    input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
    assert host_input_tensor.shape[-2] % input_l1_core_grid.num_cores == 0
    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(host_input_tensor.shape[2] // input_l1_core_grid.num_cores, host_input_tensor.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    warmup_input = ttnn.to_device(host_input_tensor, device, memory_config=input_l1_mem_config)
    _ = tt_model.forward(warmup_input)
    ttnn.deallocate(warmup_input)

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


def run_perf_e2e_pointpillars(device, batch_size_per_device, expected_throughput):
    """Execute end-to-end performance test and generate perf report."""
    profiler.clear()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    num_iterations = 32

    test_infra = setup_pointpillars_test_infra(device, batch_size_per_device)
    outputs, validation_passed = run_pointpillars_pipeline(device, test_infra, num_iterations)

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / num_iterations
    fps = batch_size / inference_time_avg

    prep_perf_report(
        model_name=f"ttnn_pointpillars_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=300,
        expected_inference_time=batch_size / expected_throughput,
        comments=f"pointpillars_batchsize{batch_size}_devices{num_devices}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"PointPillars BS={batch_size}, Device= {num_devices}: Inference time= {inference_time_avg:.4f}s/frame, Achieved FPS= {fps:.2f} (expected: {expected_throughput})"
    )
    assert validation_passed, "Output validation failed"


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("expected_inference_throughput", [19.7])
def test_pointpillars_perf_single_device(device, batch_size_per_device, expected_inference_throughput):
    """Test PointPillars performance on single device (N150)."""
    run_perf_e2e_pointpillars(device, batch_size_per_device, expected_inference_throughput)


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("expected_inference_throughput", [40])
def test_pointpillars_perf_multi_device(mesh_device, batch_size_per_device, expected_inference_throughput):
    """Test PointPillars performance on multi-device (N300)."""
    run_perf_e2e_pointpillars(mesh_device, batch_size_per_device, expected_inference_throughput)
