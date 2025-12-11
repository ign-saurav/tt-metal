# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger
from tqdm import tqdm

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.pointpillars.runner.performant_runner import PointPillarsPerformantRunner


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
    performant_runner = PointPillarsPerformantRunner(
        device=device,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
    )

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

            _ = performant_runner.run(check_pcc=check_pcc)
            check_pcc = False

    performant_runner.release()
