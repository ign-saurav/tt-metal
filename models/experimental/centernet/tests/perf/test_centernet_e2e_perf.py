# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.centernet.tests.perf.performant_infra import CenterNetPerformantTestInfra
from models.experimental.centernet.reference.dlav0 import DLASeg
from models.experimental.centernet.tt.dla_seg import TtDLASeg
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.common.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize("batch_size, size, expected_compile_time, expected_throughput_fps", [(1, 512, 30.0, 91.0)])
@pytest.mark.models_performance_bare_metal
def test_centernet_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    """End-to-end performance test for CenterNet with DLA-34 backbone."""
    torch.manual_seed(42)

    # CenterNet configuration
    heads = {"hm": 80, "wh": 2, "reg": 2}
    down_ratio = 4
    head_conv = 256

    # Create PyTorch model
    torch_model = DLASeg(
        base_name="dla34",
        heads=heads,
        pretrained=False,
        down_ratio=down_ratio,
        head_conv=head_conv,
    )
    torch_model.eval()

    # Create input tensor
    torch_input = torch.randn(batch_size, 3, size, size, dtype=torch.float32)

    # Get mesh mappers for TTNN
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess model parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.layer_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input), device=device
    )

    # Create TTNN model
    ttnn_model = TtDLASeg(
        heads=heads,
        down_ratio=down_ratio,
        head_conv=head_conv,
        parameters=parameters.dla_seg,
        device=device,
        layer_args=parameters.layer_args,
    )

    ttnn.synchronize_device(device)

    # Initialize performant infrastructure
    infra = CenterNetPerformantTestInfra(device, ttnn_model, dtype=ttnn.bfloat16)
    pipeline_model = infra

    # Create pipeline memory configs
    ttnn_input_tensor, l1_input_memory_config, dram_input_memory_config = infra.create_pipeline_memory_configs(
        torch_input
    )

    # Create pipeline
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    # Prepare input tensors for multiple iterations
    input_tensors = [ttnn_input_tensor] * num_iterations

    # Compile pipeline
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()
    compile_time = end - start

    # Preallocate output tensors
    pipeline.preallocate_output_tensors_on_host(num_iterations)

    # Run inference
    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    # Cleanup
    pipeline.cleanup()

    # Calculate performance metrics
    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    # Prepare performance report
    prep_perf_report(
        model_name="centernet-dlav0-trace-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )
