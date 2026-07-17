# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import time
import torch

import pytest
import ttnn
from loguru import logger
from tqdm import tqdm

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.pointpillars.runner.performant_runner_infra import PointPillarsPerformanceRunnerInfra


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("test_duration", [5])
@pytest.mark.parametrize("pcc_check_interval", [5])
def test_pointpillars_stability(
    device,
    batch_size,
    model_location_generator,
    test_duration,
    pcc_check_interval,
):
    # Initialize infrastructure directly
    runner_infra = PointPillarsPerformanceRunnerInfra(
        device=device,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
    )

    # Setup default input and get torch reference
    default_batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
    runner_infra.get_torch_reference(default_batched_pts)
    tt_inputs_host, input_mem_config = runner_infra.setup_dram_interleaved_input(default_batched_pts)

    # Allocate device tensor and capture trace
    input_dram_tensor = ttnn.allocate_tensor_on_device(
        tt_inputs_host.shape,
        tt_inputs_host.dtype,
        tt_inputs_host.layout,
        device,
        input_mem_config,
    )

    # Capture trace
    op_event = ttnn.record_event(device, 0)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, input_dram_tensor, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    runner_infra.input_tensor = input_dram_tensor
    op_event = ttnn.record_event(device, 0)
    runner_infra.run()
    runner_infra.validate()
    runner_infra.dealloc_output()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, input_dram_tensor, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    runner_infra.input_tensor = input_dram_tensor
    op_event = ttnn.record_event(device, 0)
    runner_infra.run()
    runner_infra.validate()

    # Capture trace
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, input_dram_tensor, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    runner_infra.input_tensor = input_dram_tensor
    op_event = ttnn.record_event(device, 0)
    runner_infra.dealloc_output()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    runner_infra.run()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    logger.info(f"Running stability test for PointPillars with batch_size={batch_size}")

    pcc_iter = 0
    check_pcc = False
    start_time = time.time()

    with tqdm(total=test_duration, desc="Executing on device", unit="sec", mininterval=1) as pbar:
        while True:
            elapsed_time = round(time.time() - start_time, 1)
            pbar.update(min(elapsed_time, test_duration) - pbar.n)

            if elapsed_time >= test_duration:
                break

            if elapsed_time >= pcc_iter * pcc_check_interval:
                check_pcc = True
                pcc_iter += 1

            # Execute trace
            ttnn.wait_for_event(1, op_event)
            ttnn.copy_host_to_device_tensor(tt_inputs_host, input_dram_tensor, 1)
            write_event = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, write_event)
            op_event = ttnn.record_event(device, 0)
            tt_output = ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

            if check_pcc:
                runner_infra.validate(tt_output)
            check_pcc = False

    # Release trace
    ttnn.release_trace(device, tid)
