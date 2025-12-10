# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.model.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)


def run_pointpillars_e2e(
    device,
    batch_size_per_device,
    model_location_generator=None,
):
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
        logger.info("Successfully loaded pretrained weights from epoch_160.pth")
    except FileNotFoundError:
        logger.warning("Checkpoint file 'epoch_160.pth' not found, using random weights")

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create test input (point cloud)
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

    # === PREPROCESSING (done once, outside pipeline) ===
    # Preprocess point cloud to pillar features
    pillar_features = preprocessor.forward(batched_pts)
    pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))  # NHWC to NCHW
    pillar_features = ttnn.reshape(
        pillar_features,
        (pillar_features.shape[0], 1, pillar_features.shape[1] * pillar_features.shape[2], pillar_features.shape[3]),
    )

    # Convert to host tensor for pipeline input
    host_input_tensor = ttnn.to_torch(pillar_features)
    host_input_tensor = ttnn.from_torch(
        host_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # === PIPELINE SETUP ===
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

    # Create model wrapper for pipeline
    def pointpillars_model_wrapper(l1_input_tensor):
        # The pipeline handles transfers, so we just run the model
        return tt_model.forward(l1_input_tensor)

    # Warmup pass to prepare conv_transpose2d weights on device BEFORE trace capture
    warmup_input = ttnn.to_device(host_input_tensor, device, memory_config=input_l1_mem_config)
    _ = tt_model.forward(warmup_input)
    ttnn.deallocate(warmup_input)
    logger.info("Warmup pass complete - weights prepared on device")

    # Configure pipeline with trace enabled (weights are now prepared, no writes during trace)
    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)

    pipe = create_pipeline_from_config(
        config,
        pointpillars_model_wrapper,
        device,
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    # === PIPELINE EXECUTION ===
    iterations = 10
    host_inputs = [host_input_tensor] * iterations

    pipe.compile(host_input_tensor)
    pipe.preallocate_output_tensors_on_host(len(host_inputs))

    start = time.time()
    outputs = pipe.enqueue(host_inputs).pop_all()
    end = time.time()

    pipe.cleanup()

    # Compare outputs (using last iteration)
    tt_cls, tt_reg, tt_dir = outputs[-1]

    # Compare classification output
    tt_cls_torch = tt2torch_tensor(tt_cls)
    tt_cls_torch = tt_cls_torch.permute(0, 3, 1, 2)
    passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.97)
    logger.info(f"Classification PCC: {pcc_cls}")
    assert passing_cls, f"Classification PCC check failed: {pcc_cls}"

    # Compare regression output
    tt_reg_torch = tt2torch_tensor(tt_reg)
    tt_reg_torch = tt_reg_torch.permute(0, 3, 1, 2)
    passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
    logger.info(f"Regression PCC: {pcc_reg}")
    assert passing_reg, f"Regression PCC check failed: {pcc_reg}"

    # Compare direction output
    tt_dir_torch = tt2torch_tensor(tt_dir)
    tt_dir_torch = tt_dir_torch.permute(0, 3, 1, 2)
    passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
    logger.info(f"Direction PCC: {pcc_dir}")
    assert passing_dir, f"Direction PCC check failed: {pcc_dir}"

    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * batch_size / (end-start) : .2f} fps")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", [1])
def test_pointpillars_e2e_pipeline(batch_size_per_device, device, model_location_generator):
    run_pointpillars_e2e(device, batch_size_per_device, model_location_generator)
