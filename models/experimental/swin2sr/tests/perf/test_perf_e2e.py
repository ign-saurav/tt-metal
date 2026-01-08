import os
import pytest
import torch
import cv2
import numpy as np
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


def load_test_image(image_path: str, target_size: int, batch_size: int) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1))

    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0)

    if batch_size > 1:
        img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)

    return img_tensor


def get_default_test_image_path(img_size: int = 64) -> str:
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    test_image_dir = os.path.join(workspace_root, "models", "experimental", "swin2sr", "resources", "test_images")

    default_image = os.path.join(test_image_dir, "Set5", "LR_bicubic", "X2", "babyx2.png")

    if not os.path.exists(default_image):
        for root, dirs, files in os.walk(test_image_dir):
            for file in files:
                if file.endswith(".png"):
                    default_image = os.path.join(root, file)
                    break
            if os.path.exists(default_image):
                break

    if not os.path.exists(default_image):
        raise FileNotFoundError(
            f"No test image found in {test_image_dir}. "
            "Please provide a test_image_path or ensure test images are available."
        )

    return default_image


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
        test_image_path=None,
        use_test_image=True,
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

        if test_image_path is not None:
            logger.info(f"Loading test image from: {test_image_path}")
            self.torch_input_tensor = load_test_image(test_image_path, img_size, batch_size)
        elif use_test_image:
            try:
                default_image_path = get_default_test_image_path(img_size)
                logger.info(f"Using default test image: {default_image_path}")
                self.torch_input_tensor = load_test_image(default_image_path, img_size, batch_size)
            except FileNotFoundError as e:
                logger.warning(f"Could not find test image, falling back to random input: {e}")
                self.torch_input_tensor = torch.randn(batch_size, 3, img_size, img_size)
        else:
            logger.info("Using random input tensor for performance testing")
            self.torch_input_tensor = torch.randn(batch_size, 3, img_size, img_size)

        with torch.no_grad():
            self.torch_output = self.torch_model(self.torch_input_tensor)
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

        self.input_tensor = None
        self.tt_output = None

    def run(self):
        assert self.input_tensor is not None, "input_tensor must be set before calling run()"
        assert self.input_tensor.shape == (
            self.batch_size,
            3,
            self.img_size,
            self.img_size,
        ), f"Expected input shape {(self.batch_size, 3, self.img_size, self.img_size)}, got {self.input_tensor.shape}"
        self.tt_output = self.ttnn_model.forward(self.input_tensor)

    def validate(self, tt_output=None):
        tt_output = self.tt_output if tt_output is None else tt_output
        assert tt_output is not None, "Swin2SR output tensor is None"


def run_model_pipeline(device, test_infra, num_measurement_iterations, use_trace=False, num_command_queues=1):
    tt_inputs_host = ttnn.from_torch(
        test_infra.torch_input_tensor,
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def model_wrapper(input_tensor):
        assert input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Input tensor must be on device"
        assert len(input_tensor.shape) == 4, f"Expected 4D input tensor (NCHW), got shape {input_tensor.shape}"

        test_infra.input_tensor = input_tensor
        test_infra.run()
        return test_infra.tt_output

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=use_trace,
            num_command_queues=num_command_queues,
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
    num_command_queues=1,
    upscale=2,
    test_image_path=None,
    use_test_image=True,
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
        upscale=upscale,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
        test_image_path=test_image_path,
        use_test_image=use_test_image,
    )

    num_measurement_iterations = 32
    outputs, profiler_label = run_model_pipeline(
        device, test_infra, num_measurement_iterations, use_trace=use_trace, num_command_queues=num_command_queues
    )

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get(profiler_label) / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    trace_suffix = "trace" if use_trace else "notrace"
    model_name = f"ttnn_swin2sr_{trace_suffix}_{num_command_queues}cq_batch_size{batch_size}"

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=300,
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
