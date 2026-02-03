# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn

from loguru import logger
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0

from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import SunrgbdDatasetConfig

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.model_config import Tt3DetrArgs
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.ttnn.utils import infer_ttnn_module_args, NO_FALLBACK
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


def create_3detr_pipeline_model(ttnn_model, input_dict):
    """Placeholder wrapper to adapt 3detr model inputs for pipeline interface."""

    def run(pipeline_input):
        output = ttnn_model(inputs=input_dict, encoder_only=False)
        return output

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, size, expected_compile_time, expected_throughput_fps",
    [(1, 20000, 3.32, 0.40)],
)
@pytest.mark.models_performance_bare_metal
def test_3detr_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    model_location_generator,
):
    """
    Test 3detr end-to-end performance with Pipeline API.
    """
    torch.manual_seed(0)

    # Configuration
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()
    dtype = ttnn.bfloat16

    logger.info("Building 3detr model...")
    torch_model, _ = build_3detr(args, dataset_config)
    load_torch_model_state(torch_model, model_location_generator=model_location_generator)

    # Sample input
    min_val = -1.8827
    max_val = 8.3542
    pc = (max_val - min_val) * torch.rand((batch_size, size, 3)) + min_val
    sample_input = {
        "point_clouds": pc,
        "point_cloud_dims_min": torch.min(pc, 1)[0],
        "point_cloud_dims_max": torch.max(pc, 1)[0],
    }

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=torch_model,
        run_model=lambda model: torch_model(inputs=sample_input, encoder_only=False),
        device=device,
    )

    ttnn_args = Tt3DetrArgs()
    ttnn_args.parameters = parameters
    ttnn_args.device = device

    ttnn_model, _ = build_ttnn_3detr(ttnn_args, dataset_config)

    ttnn.synchronize_device(device)

    logger.info("Creating pipeline model...")
    pipeline_model = create_3detr_pipeline_model(ttnn_model, sample_input)

    logger.info("Preparing input tensor...")
    if NO_FALLBACK:
        ttnn_dict = {}
        for key, value in sample_input.items():
            if isinstance(value, torch.Tensor):
                ttnn_dict[key] = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            else:
                ttnn_dict[key] = value
        sample_input = ttnn_dict
        ttnn_input_tensor = ttnn_dict["point_clouds"]
    else:
        ttnn_input_tensor = ttnn.from_torch(
            sample_input["point_clouds"],
            device=None,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    logger.info(f"Configuring pipeline...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensors = [ttnn_input_tensor] * num_iterations

    # Pipeline API accepts single tensor; wrapper uses actual 3detr inputs from closure
    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="3detr-notrace-1cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )

    logger.info("Performance test completed!")
