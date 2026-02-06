# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import (
    ParameterDict,
    infer_ttnn_module_args as infer_ttnn_module_args_torch,
    preprocess_model_parameters,
)

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet, process_input
from models.experimental.transfuser.resources.transfuser_checkpoint import ensure_transfuser_checkpoint_2022
from models.experimental.transfuser.resources.transfuser_dataset import ensure_scenario3_town01_curved_route0
from models.experimental.transfuser.tests.pcc.test_gpt import create_gpt_preprocessor
from models.experimental.transfuser.tests.pcc.test_lidar_center_net import (
    delete_incompatible_keys,
    get_mesh_mappers,
    load_trained_weights,
)
from models.experimental.transfuser.tests.pcc.test_transfuser_backbone import regroup_model_args
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def load_clean_checkpoint(checkpoint_path):
    """Load and clean checkpoint by removing 'module.' prefix from keys."""
    state_dict = load_trained_weights(checkpoint_path)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v
    return delete_incompatible_keys(cleaned, [])


def dummy_head_preprocessor(device, dtype):
    """Dummy head preprocessor for performance testing (returns empty params)."""

    def _preprocess(model, name, *, ttnn_module_args=None, convert_to_ttnn=None):
        return ParameterDict({})

    return _preprocess


def create_transfuser_pipeline_model(tt_layer, tt_lidar_bev, tt_velocity, target_point):
    """Create a pipeline model function for Transfuser."""

    def run(l1_input_tensor):
        # Convert to DRAM interleaved before slicing to avoid page size alignment issues with sharded tensors
        l1_input_dram = ttnn.to_memory_config(l1_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        tt_image_unpadded = l1_input_dram[:, :, :, :3]
        return tt_layer.forward_ego(tt_image_unpadded, tt_lidar_bev, tt_velocity, target_point)

    return run


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("num_warmup", [2])
@pytest.mark.parametrize("batch_size", [1])
def test_perf_transfuser_e2e(device, num_iterations, num_warmup, batch_size):
    """Test Transfuser end-to-end performance with Pipeline API (2CQ with trace)."""
    torch.manual_seed(42)

    # Load data and checkpoint
    data_root = ensure_scenario3_town01_curved_route0()
    checkpoint_path = ensure_transfuser_checkpoint_2022()

    config = GlobalConfig(setting="eval")
    config.n_layer = 4
    config.use_target_point_image = True

    inputs = process_input(data_root, "0120", config=config, normalize_image=False)
    image, lidar_bev, velocity, target_point = (
        inputs["image"],
        inputs["lidar"],
        inputs["velocity"],
        inputs["target_point"],
    )

    inputs_mesh_mapper, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Load reference model
    logger.info("Loading model...")
    ref_layer = LidarCenterNet(
        config,
        backbone="transFuser",
        image_architecture="regnety_032",
        lidar_architecture="regnety_032",
        use_velocity=False,
    ).eval()
    ref_layer.load_state_dict(load_clean_checkpoint(checkpoint_path), strict=False)
    torch_model = ref_layer._model

    # Preprocess parameters
    logger.info("Preprocessing parameters...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )

    model_args = regroup_model_args(
        infer_ttnn_module_args_torch(
            model=torch_model,
            run_model=lambda m: m(image, lidar_bev, velocity),
            device=None,
            absolute_name=True,
        )
    )

    # Transformers
    for name in ["transformer1", "transformer2", "transformer3", "transformer4"]:
        parameters[name] = preprocess_model_parameters(
            initialize_model=lambda n=name: getattr(torch_model, n),
            custom_preprocessor=create_gpt_preprocessor(device, config.n_layer, ttnn.bfloat16, False),
            device=device,
        )

    # Head (perf-only dummy preprocessing)
    parameters["head"] = preprocess_model_parameters(
        initialize_model=lambda: ref_layer.head,
        custom_preprocessor=dummy_head_preprocessor(device, ttnn.bfloat16),
        device=device,
    )

    # Create TTNN model
    logger.info("Creating TTNN model...")
    tt_layer = TtLidarCenterNet(
        device,
        parameters,
        config,
        backbone="transFuser",
        torch_model=torch_model,
        model_args=model_args,
    )

    # Prepare input tensors
    logger.info("Preparing input tensor...")
    image_nhwc = image.permute(0, 2, 3, 1)
    image_padded = torch.nn.functional.pad(image_nhwc, (0, 29))
    ttnn_input_tensor = ttnn.from_torch(
        image_padded,
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_lidar_bev = ttnn.from_torch(
        lidar_bev.permute(0, 2, 3, 1),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    tt_velocity = ttnn.from_torch(
        velocity,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Create pipeline model
    logger.info("Creating pipeline model...")
    pipeline_model = create_transfuser_pipeline_model(tt_layer, tt_lidar_bev, tt_velocity, target_point)

    # Create sharded memory configs for 2CQ
    logger.info("Creating memory configs...")
    batch_size, height, width, channels = image_padded.shape
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

    # Create DRAM input memory config
    dram_grid_size = device.dram_grid_size()
    num_dram_cores = dram_grid_size.x
    assert dram_grid_size.y == 1, "Only 1D DRAM grid is supported"

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

    # Create pipeline
    logger.info("Configuring pipeline (2CQ with trace)...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    # Compile
    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    ttnn.synchronize_device(device)
    end = time.time()
    compile_time = end - start

    # Warmup
    warmup_inputs = [ttnn_input_tensor] * num_warmup
    pipeline.enqueue(warmup_inputs).pop_all()

    # Benchmark
    logger.info(f"Running {num_iterations} inference iterations...")
    ttnn.synchronize_device(device)
    start = time.time()
    pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    prep_perf_report(
        model_name="transfuser-e2e-2cq-trace",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=100.0,
        expected_inference_time=1.0,
        comments="2cq_trace",
    )

    logger.info("Performance test completed!")
