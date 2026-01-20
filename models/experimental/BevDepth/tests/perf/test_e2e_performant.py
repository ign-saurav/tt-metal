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


def create_bevdepth_pipeline_model(ttnn_model, mats_dict, torch_input_imgs):
    """
    Create a pipeline model function for BevDepth..
    """

    def run(dummy_input):
        ttnn_output = ttnn_model(torch_input_imgs, mats_dict)

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
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, num_sweeps, num_cameras, img_h, img_w, expected_compile_time, expected_throughput_fps",
    [(1, 2, 6, 256, 704, 9.9, 0.17)],
)
@pytest.mark.models_performance_bare_metal
def test_bevdepth_e2e_performant(
    device,
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
    Test BevDepth end-to-end performance with Pipeline API (1CQ, no trace).
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
    pipeline_model = create_bevdepth_pipeline_model(ttnn_model, mats_dict, torch_input_imgs)

    logger.info("Preparing dummy input tensor...")
    dummy_input = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32),
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TODO: 2CQ with trace as currently the deformconv2d is not supported in TTNN. Issue #34509 and PR #34940 are still open.
    logger.info(f"Configuring pipeline (1CQ without trace)...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensors = [dummy_input] * num_iterations

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(dummy_input)
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
        model_name="bevdepth-notrace-1cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}-sweeps_{num_sweeps}-cameras_{num_cameras}-img_{img_h}x{img_w}",
    )

    logger.info("Performance test completed!")
