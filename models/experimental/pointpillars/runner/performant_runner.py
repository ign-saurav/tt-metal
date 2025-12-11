# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.experimental.pointpillars.runner.performant_runner_infra import PointPillarsPerformanceRunnerInfra


class PointPillarsPerformantRunner:
    def __init__(
        self,
        device,
        batch_size=1,
        model_location_generator=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        checkpoint_path="epoch_160.pth",
    ):
        self.device = device
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        self.runner_infra = PointPillarsPerformanceRunnerInfra(
            device=device,
            batch_size=batch_size,
            model_location_generator=model_location_generator,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            outputs_mesh_composer=outputs_mesh_composer,
            checkpoint_path=checkpoint_path,
        )

        self.default_batched_pts = [torch.randn(18221, 4, dtype=torch.bfloat16)]
        self.runner_infra.get_torch_reference(self.default_batched_pts)

        (
            self.tt_inputs_host,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_interleaved_input(self.default_batched_pts)

        self._capture_pointpillars_trace_2cqs()

    def _capture_pointpillars_trace_2cqs(self):
        self.input_dram_tensor = ttnn.allocate_tensor_on_device(
            self.tt_inputs_host.shape,
            self.tt_inputs_host.dtype,
            self.tt_inputs_host.layout,
            self.device,
            self.input_mem_config,
        )

        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

    def _execute_pointpillars_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.tt_output

    def _validate(self):
        self.runner_infra.validate(self.tt_output)

    def run(self, batched_pts=None, check_pcc=False):
        if batched_pts is not None:
            self.runner_infra.get_torch_reference(batched_pts)
            (
                self.tt_inputs_host,
                _,
            ) = self.runner_infra.setup_dram_interleaved_input(batched_pts)

        self.tt_output = None
        self.tt_output = self._execute_pointpillars_trace_2cqs_inference(tt_inputs_host=self.tt_inputs_host)

        if check_pcc:
            self._validate()
        return self.tt_output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
