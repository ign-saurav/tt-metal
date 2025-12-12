# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from ....pipelines.flux1_kontext.pipeline_flux1_kontext import Flux1KontextPipeline
from diffusers.utils import load_image


@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, true_cfg_scale, num_inference_steps",
    [
        (1024, 1024, 2.5, 1.6, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
    ],
    ids=[
        "2x4cfg1sp2tp4",
        "2x4cfg2sp1tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 34000000}],
    indirect=True,
)
def test_flux1_kontext_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w,
    image_h,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    cfg,
    sp,
    tp,
    topology,
    num_links,
    model_location_generator,
    is_ci_env,
) -> None:
    """Performance test for Flux.1 Kontext pipeline with detailed timing analysis. We use the dev variant"""

    benchmark_profiler = BenchmarkProfiler()

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Guidance scale: {guidance_scale}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    pipeline = Flux1KontextPipeline.create_pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-Kontext-dev"),
        mesh_device=mesh_device,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        topology=topology,
        num_links=num_links,
    )

    # Test prompts
    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ).convert("RGB")
    prompts = [
        "Make Pikachu hold a sign that says 'TTNN is awesome', yarn art style, detailed, vibrant colors"
    ]
    negative_prompts = [""] * len(prompts)

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        images = pipeline.run_single_prompt(
            image=input_image,
            width=image_w,
            height=image_h,
            prompt=prompts[0],
            negative_prompt=negative_prompts[0],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
            cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
        )
    images[0].save(f"flux1_kontext_dev_{image_w}_{image_h}_warmup.png")

    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    num_perf_runs = 1

    # Optional Tracy profiling (if available)
    profiler = None
    try:
        from tracy import Profiler

        profiler = Profiler()
        profiler.enable()
        logger.info("Tracy profiling enabled")
    except ImportError:
        logger.info("Tracy profiler not available, continuing without profiling")

    try:
        for i in range(num_perf_runs):
            logger.info(f"Performance run {i+1}/{num_perf_runs}...")

            # Run pipeline with different prompt
            prompt_idx = (i + 1) % len(prompts)
            with benchmark_profiler("run", iteration=i):
                images = pipeline.run_single_prompt(
                    image=input_image,
                    prompt=prompts[prompt_idx],
                    negative_prompt=negative_prompts[prompt_idx],
                    cfg_scale=true_cfg_scale,
                    height=image_h,
                    width=image_w,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=0,
                    traced=True,
                    timer=benchmark_profiler,
                    timer_iteration=i,
                )
            images[0].save(f"flux1_kontext_dev_{image_w}_{image_h}_perf_run{i}.png")

            logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # Calculate statistics
    clip_times = [benchmark_profiler.get_duration("clip_encoding", i) for i in range(num_perf_runs)]
    t5_times = [benchmark_profiler.get_duration("t5_encoding", i) for i in range(num_perf_runs)]
    total_encoding_times = [benchmark_profiler.get_duration("total_encoding", i) for i in range(num_perf_runs)]
    vae_encoding_times = [benchmark_profiler.get_duration("vae_encoding", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae_decoding", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    # Calculate per-step denoising times
    all_denoising_steps = []
    for i in range(num_perf_runs):
        for j in range(num_inference_steps):
            assert benchmark_profiler.contains_step(
                f"denoising_step_{j}", i
            ), f"All runs should have {num_inference_steps} denoising steps"
            all_denoising_steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", i))

    # Report results
    sp_factor = sp[0]  # pipeline.dit_parallel_config.sequence_parallel.factor
    tp_factor = tp[0]  # pipeline.dit_parallel_config.tensor_parallel.factor

    print("\n" + "=" * 80)
    print("FLUX.1 KONTEXT DEV PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: FLUX.1-Kontext-dev")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {topology}")
    print("-" * 80)

    def print_stats(name, times):
        if not times:
            print(f"{name:25} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    print_stats("CLIP Encoding", clip_times)
    print_stats("T5 Encoding", t5_times)
    print_stats("Total Encoding", total_encoding_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("VAE Decoding", vae_times)
    print_stats("VAE Encoding", vae_encoding_times)
    print_stats("Total Pipeline", total_times)

    print("-" * 80)

    # Additional metrics
    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * num_inference_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {num_inference_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} images/second")

        # Breakdown percentages
        avg_encoding_time = statistics.mean(total_encoding_times)
        avg_vae_time = statistics.mean(vae_times)
        avg_vae_encoding_time = statistics.mean(vae_encoding_times)

        print(f"\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE Encoding: {avg_vae_encoding_time/avg_total_time*100:.1f}%")
        print(f"  VAE Decoding: {avg_vae_time/avg_total_time*100:.1f}%")

    print("=" * 80)

    # Validate performance
    measurements = {
        "clip_encoding_time": statistics.mean(clip_times),
        "t5_encoding_time": statistics.mean(t5_times),
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_encoding_time": statistics.mean(vae_encoding_times),
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 4):
        expected_metrics = {
            "clip_encoding_time": 0.1,
            "t5_encoding_time": 0.27,
            "total_encoding_time": 0.6,
            "denoising_steps_time": 2.5 * num_inference_steps,
            "vae_encoding_time": 1.8,
            "vae_decoding_time": 1.8,
            "total_time": 70,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
        benchmark_data = BenchmarkData()
        for iteration in range(num_perf_runs):
            for step_name, target in zip(
                ["clip_encoder", "t5_encoder", "vae_encoder", "denoising", "vae_decoder", "run"],
                [
                    expected_metrics["clip_encoding_time"],
                    expected_metrics["t5_encoding_time"],
                    expected_metrics["vae_encoding_time"],
                    expected_metrics["denoising_steps_time"],
                    expected_metrics["vae_decoding_time"],
                    expected_metrics["total_time"],
                ],
            ):
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=target,
                )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="WH_T3K",
            ml_model_name="Flux1KontextDev",
            batch_size=1,
            config_params={
                "width": image_w,
                "height": image_h,
                "num_frames": 1,
                "num_steps": num_inference_steps,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
                "num_links": num_links,
                "fsdp": False,
            },
        )

    pass_perf_check = True
    assert_msgs = []
    for k in expected_metrics.keys():
        if measurements[k] > expected_metrics[k]:
            assert_msgs.append(
                f"Warning: {k} is outside of the tolerance range. Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False

    assert pass_perf_check, "\n".join(assert_msgs)

    # Synchronize all devices
    pipeline.synchronize_devices()

    logger.info("Performance test completed successfully!")
