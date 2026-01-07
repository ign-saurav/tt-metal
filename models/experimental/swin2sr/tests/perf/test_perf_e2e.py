import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)

from models.experimental.swin2sr.reference.swin2sr import Swin2SR as TorchSwin2SR
from models.experimental.swin2sr.tt.tt_swin2sr import TtSwin2SR
from models.experimental.swin2sr.tests.pcc.test_ttnn_swin2sr import create_swin2sr_preprocessor


class Swin2SRPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        img_size=64,
        upscale=2,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        resi_connection="1conv",
        inputs_mesh_mapper=None,
        outputs_mesh_composer=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.img_size = img_size
        self.upscale = upscale
        self.batch_size = batch_size
        self.device = device
        self.model_location_generator = model_location_generator

        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        # Reference PyTorch model
        self.torch_model = TorchSwin2SR(
            img_size=img_size,
            patch_size=1,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upscale=upscale,
            img_range=1.0,
            upsampler="pixelshuffle",
            resi_connection=resi_connection,
        )
        self.torch_model.eval()

        # Random input in NCHW format
        self.torch_input_tensor = torch.randn(batch_size, 3, img_size, img_size)

        with torch.no_grad():
            self.torch_output = self.torch_model(self.torch_input_tensor)

        # Keep input in NCHW format - Swin2SR model expects NCHW

        # Preprocess and create TT model
        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_swin2sr_preprocessor(device),
            device=device,
        )

        self.ttnn_model = TtSwin2SR(
            device=device,
            parameters=parameters,
            img_size=img_size,
            patch_size=1,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upscale=upscale,
            img_range=1.0,
            upsampler="pixelshuffle",
            resi_connection=resi_connection,
        )

    def run(self):
        self.tt_output = self.ttnn_model.forward(self.input_tensor)

    def validate(self, tt_output=None):
        # For pipeline performance tests, we rely on separate PCC tests for correctness validation.
        # Here we only verify that the model produces an output tensor without runtime errors.
        # The output tensor shape validation is handled by the model itself.
        tt_output = self.tt_output if tt_output is None else tt_output
        # Just verify the output exists and is a valid tensor
        assert tt_output is not None, "Swin2SR output tensor is None"
        # Note: Full PCC validation is done in test_ttnn_swin2sr.py


def run_model_pipeline(device, test_infra, num_measurement_iterations, use_trace=False, num_command_queues=1):
    # Swin2SR doesn't work with sharded input due to 3-channel requirement and buffer alignment issues
    # This means it can't use 2CQ+trace configuration (which requires sharding)
    # Use 1CQ without trace for Swin2SR's transformer architecture
    # Note: device=None keeps tensor on host (required for pipeline.compile)
    # For Swin2SR, we don't use mesh_mapper on input to avoid sharding requirements
    # Multi-device support is handled at the model level, not input level
    tt_inputs_host = ttnn.from_torch(
        test_infra.torch_input_tensor,
        device=None,  # Keep on host for pipeline.compile()
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        # Note: Not using mesh_mapper here to avoid sharding requirements
        # Swin2SR's transformer architecture handles multi-device differently than CNNs
    )

    def model_wrapper(input_tensor):
        # Input tensor is already on device in NCHW format
        test_infra.input_tensor = input_tensor
        test_infra.run()
        return test_infra.tt_output

    # Swin2SR uses 1CQ (not 2CQ) because 2CQ requires sharded input which Swin2SR doesn't support
    # device_params may specify 2CQ for device capability, but Swin2SR's architecture limits us to 1CQ
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=use_trace,
            num_command_queues=num_command_queues,  # 1CQ for Swin2SR (2CQ requires sharding)
            all_transfers_on_separate_command_queue=False,
        ),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    logger.info(f"Running Swin2SR pipeline warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(
        f"Starting Swin2SR performance pipeline for {num_measurement_iterations} iterations "
        f"with batch_size={test_infra.batch_size} and num_devices={test_infra.device.get_num_devices()}"
    )

    # Use profiler label that matches actual configuration
    profiler_label = f"run_swin2sr_pipeline_{num_command_queues}cq"
    if use_trace:
        profiler_label += "_trace"
    else:
        profiler_label += "_notrace"

    profiler.start(profiler_label)
    try:
        outputs = pipeline.enqueue(host_inputs).pop_all()
    except Exception as e:
        logger.error(f"Swin2SR pipeline execution failed: {e}")
        pipeline.cleanup()
        raise
    finally:
        profiler.end(profiler_label)

    # Validate outputs
    for i, output in enumerate(outputs):
        try:
            test_infra.validate(output)
            logger.info(f"Swin2SR pipeline output {i} validation passed")
        except Exception as e:
            logger.error(f"Swin2SR pipeline output {i} validation failed: {e}")
            raise

    pipeline.cleanup()

    return outputs, profiler_label


def run_perf_e2e_swin2sr(
    device,
    batch_size_per_device,
    model_location_generator,
    img_size,
    expected_inference_throughput,
    use_trace=False,
    num_command_queues=1,  # Swin2SR uses 1CQ (2CQ requires sharding which Swin2SR doesn't support)
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = Swin2SRPerformanceRunnerInfra(
        device,
        batch_size,
        model_location_generator=model_location_generator,
        img_size=img_size,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
    )

    num_measurement_iterations = 32
    outputs, profiler_label = run_model_pipeline(
        device, test_infra, num_measurement_iterations, use_trace=use_trace, num_command_queues=num_command_queues
    )

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get(profiler_label) / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    # Update model name to match actual configuration
    trace_suffix = "trace" if use_trace else "notrace"
    model_name = f"ttnn_swin2sr_{trace_suffix}_{num_command_queues}cq_batch_size{batch_size}"

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=300,  # Conservative default, can be tuned
        expected_inference_time=expected_inference_time,
        comments=f"{img_size}x{img_size}_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"Swin2SR {img_size}x{img_size} batch_size: {batch_size}, "
        f"inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"Swin2SR compile time: {compile_time} s")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    # Note: Swin2SR uses 1CQ (not 2CQ) because 2CQ requires sharded input which Swin2SR doesn't support
    # due to its 3-channel requirement and transformer architecture
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 1702912, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "img_size, expected_inference_throughput",
    [
        (64, 40),
    ],
)
def test_swin2sr_perf_single_device(
    device,
    batch_size_per_device,
    model_location_generator,
    img_size,
    expected_inference_throughput,
):
    run_perf_e2e_swin2sr(
        device,
        batch_size_per_device,
        model_location_generator,
        img_size,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    # Note: Swin2SR uses 1CQ (not 2CQ) because 2CQ requires sharded input which Swin2SR doesn't support
    # due to its 3-channel requirement and transformer architecture
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 1702912, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "img_size, expected_inference_throughput",
    [
        (64, 80),
    ],
)
def test_swin2sr_perf_multi_device(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
    img_size,
    expected_inference_throughput,
):
    run_perf_e2e_swin2sr(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        img_size,
        expected_inference_throughput,
    )
