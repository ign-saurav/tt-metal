# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger
from diffusers.utils import load_image

from ....pipelines.flux1_kontext.pipeline_flux1_kontext import Flux1KontextPipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 34000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("model_variant", "width", "height", "guidance_scale", "true_cfg_scale", "num_inference_steps"),
    [
        ("dev", 1024, 1024, 2.5, 1.6, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(1, 4), (1, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],  # Fully functional
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],  # Fully functional
    ],
    ids=[
        "1x4sp0tp1",
        "2x4sp0tp1",
        "2x4cfg1sp0tp1",
        "2x4cfg0sp0tp1",
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
    model_variant: str,
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

    # Create timing collector
    timing_collector = TimingCollector()

    pipeline = Flux1KontextPipeline.create_pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-Kontext-{model_variant}"),
        mesh_device=mesh_device,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        use_torch_t5_text_encoder=use_torch_t5_text_encoder,
        use_torch_clip_text_encoder=use_torch_clip_text_encoder,
        use_torch_vae_encoder=False,
        num_links=num_links,
        topology=topology,
    )

    # Set timing collector
    pipeline.timing_collector = timing_collector

    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    )
    prompts = [
        "Add a hat to the cat",
    ]
    negative_prompts = [""] * len(prompts)

    filename_prefix = f"flux_{model_variant}_{width}_{height}"
    if use_torch_t5_text_encoder:
        filename_prefix += "_t5cpu"
    if use_torch_clip_text_encoder:
        filename_prefix += "_clipcpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, negative_prompt: str, number: int, seed: int) -> None:
        images = pipeline.run_single_prompt(
            image=input_image,
            width=width,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=traced,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        timing_data = timing_collector.get_timing_data()
        logger.info(f"CLIP encoding time: {timing_data.clip_encoding_time:.2f}s")
        logger.info(f"T5 encoding time: {timing_data.t5_encoding_time:.2f}s")
        logger.info(f"Total encoding time: {timing_data.total_encoding_time:.2f}s")
        logger.info(f"VAE decoding time: {timing_data.vae_decoding_time:.2f}s")
        logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")
        logger.info(f"Total pipeline FPS: {(1 / timing_data.total_time):.2f}")
        if timing_data.denoising_step_times:
            avg_step_time = sum(timing_data.denoising_step_times) / len(timing_data.denoising_step_times)
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
