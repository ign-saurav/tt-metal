# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger
import tracy
from models.experimental.panoptic_deeplab.runner.performant_runner import PanopticDeepLabPerformantRunner
from models.experimental.panoptic_deeplab.runner.performant_runner_infra import PanopticDeepLabPerformanceRunnerInfra


MATH_FIDELITY = ttnn.MathFidelity.LoFi
WEIGHTS_DTYPE = ttnn.bfloat8_b
ACTIVATIONS_DTYPE = ttnn.bfloat8_b
INPUT_PATH = "./models/experimental/panoptic_deeplab/resources/input.png"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "resoultion",
    [
        (512, 1024),
    ],
)
def test_panoptic_deeplab_infra(
    device,
    resoultion,
):
    panoptic_runner_infra = PanopticDeepLabPerformanceRunnerInfra(
        device,
        1,
        ACTIVATIONS_DTYPE,
        WEIGHTS_DTYPE,
        MATH_FIDELITY,
        None,
        resolution=resoultion,
        input_path=INPUT_PATH,
    )
    (
        tt_inputs_host,
        input_mem_config,
    ) = panoptic_runner_infra.setup_dram_interleaved_input()

    # First run configures JIT, second run is optimized
    for phase in ("JIT configuration", "optimized"):
        tracy.signpost("start")
        logger.info(f"Running TTNN model pass ({phase})...")
        panoptic_runner_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        panoptic_runner_infra.run()
        panoptic_runner_infra.validate()
        panoptic_runner_infra.dealloc_output()
        tracy.signpost("stop")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "resoultion",
    [
        (512, 1024),
    ],
)
def test_panoptic_deeplab_traced_run(
    device,
    resoultion,
):
    panoptic_runner = PanopticDeepLabPerformantRunner(
        device,
        1,
        ACTIVATIONS_DTYPE,
        WEIGHTS_DTYPE,
        MATH_FIDELITY,
        None,
        resolution=resoultion,
        input_path=INPUT_PATH,
    )
    tracy.signpost("Traced run start")
    panoptic_runner.run(check_pcc=True)
    panoptic_runner.release()
    tracy.signpost("Traced run end")
