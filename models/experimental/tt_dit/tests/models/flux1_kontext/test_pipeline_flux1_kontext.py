# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger
from diffusers.utils import load_image

from ....pipelines.flux1_kontext.pipeline_flux1_kontext import Flux1KontextPipeline
from models.perf.benchmarking_utils import BenchmarkProfiler


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 34000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("width", "height", "guidance_scale", "true_cfg_scale", "num_inference_steps"),
    [
        (1024, 1024, 2.5, 3.5, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(1, 4), (1, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        # [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1], # TODO: support sub-mesh (2, 2)
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
    ],
    ids=[
        "1x4cfg1sp1tp4",
        "2x4cfg1sp2tp4",
        # "2x4cfg2sp1tp4",
        "2x4cfg2sp1tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("use_torch_t5_text_encoder", "use_torch_clip_text_encoder"),
    [
        pytest.param(True, True, id="encoder_cpu"),
        pytest.param(False, False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "use_torch_vae",
    [
        pytest.param(True, id="vae_cpu"),
        pytest.param(False, id="vae_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        pytest.param(False, id="not_traced"),
    ],
)
@pytest.mark.parametrize(
    "use_cache",
    [
        pytest.param(True, id="yes_use_cache"),
        pytest.param(False, id="no_use_cache"),
    ],
)
def test_flux1_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    guidance_scale: float,
    true_cfg_scale: float,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    use_torch_t5_text_encoder: bool,
    use_torch_clip_text_encoder: bool,
    use_torch_vae: bool,
    model_location_generator,
    traced: bool,
    use_cache: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setup CI environment
    if is_ci_env:
        if use_cache:
            monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
        else:
            pytest.skip("Skipping. No use cache is implicitly tested with the configured non persistent cache path.")
        if traced:
            pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

    pipeline = Flux1KontextPipeline.create_pipeline(
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.1-Kontext-dev"),
        mesh_device=mesh_device,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        use_torch_t5_text_encoder=use_torch_t5_text_encoder,
        use_torch_clip_text_encoder=use_torch_clip_text_encoder,
        use_torch_vae=use_torch_vae,
        num_links=num_links,
        topology=topology,
    )

    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ).convert("RGB")
    prompts = [
        "Make Pikachu hold a sign that says 'TTNN is awesome', yarn art style, detailed, vibrant colors",
    ]
    negative_prompts = [""] * len(prompts)

    filename_prefix = f"flux_1_kontext_dev_{width}_{height}"
    if use_torch_t5_text_encoder:
        filename_prefix += "_t5cpu"
    if use_torch_clip_text_encoder:
        filename_prefix += "_clipcpu"
    if use_torch_vae:
        filename_prefix += "_vaecpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, negative_prompt: str, number: int, seed: int) -> None:
        benchmark_profiler = BenchmarkProfiler()
        images = pipeline.run_single_prompt(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            traced=traced,
            timer=benchmark_profiler,
            timer_iteration=0,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        logger.info(f"CLIP encoding time: {benchmark_profiler.get_duration('clip_encoding', 0):.2f}s")
        logger.info(f"T5 encoding time: {benchmark_profiler.get_duration('t5_encoding', 0):.2f}s")
        logger.info(f"Total encoding time: {benchmark_profiler.get_duration('total_encoding', 0):.2f}s")
        logger.info(f"VAE encoding time: {benchmark_profiler.get_duration('vae_encoding', 0):.2f}s")
        logger.info(f"VAE decoding time: {benchmark_profiler.get_duration('vae_decoding', 0):.2f}s")
        logger.info(f"Total pipeline time: {benchmark_profiler.get_duration('total', 0):.2f}s")
        avg_step_time = benchmark_profiler.get_duration("denoising", 0) / num_inference_steps
        logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

    if no_prompt:
        for i in range(len(negative_prompts)):
            run(prompt=prompts[i], negative_prompt=negative_prompts[i], number=i, seed=0)
    else:
        prompt = prompts[0]
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, negative_prompt="", number=i, seed=i)
