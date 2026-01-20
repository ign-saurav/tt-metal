# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger

from models.experimental.tt_dit.pipelines.stable_diffusion_35_medium.pipeline_stable_diffusion_35_medium import (
    StableDiffusion3MediumPipeline as TTSD35MediumPipeline,
)
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(1, 2), 0, 1, 1],  # N300 configuration - 2 devices with CFG parallel
    ],
    ids=["1x2_n300"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
def test_sd35_medium_pipeline_functional(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
) -> None:
    """Functional test for SD3.5 Medium pipeline on N300 with CFG enabled."""
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=2, mesh_axis=1),  # CFG parallel on axis 1 for N300
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=sp_axis),
    )

    # Create pipeline with N300 CFG configuration
    tt_pipe = TTSD35MediumPipeline(
        mesh_device=mesh_device,
        enable_t5_text_encoder=False,
        guidance_cond=2,  # CFG enabled: positive + negative prompt
        parallel_config=parallel_config,
        num_links=num_links,
        height=512,  # Smaller image for faster test
        width=512,
        model_checkpoint_path="stabilityai/stable-diffusion-3.5-medium",
        use_cache=False,
    )

    # Prepare with guidance_scale=4.5 for CFG on N300
    tt_pipe.prepare(
        batch_size=1,
        num_images_per_prompt=1,
        width=512,
        height=512,
        guidance_scale=7,  # CFG enabled with guidance scale 4.5
        max_t5_sequence_length=256,
        prompt_sequence_length=333,
        spatial_sequence_length=1024,
    )

    # Test with a simple prompt
    prompt = "A capybara wearing a suit holding a sign that reads hello world"
    seed = 23
    num_steps = 40

    images = tt_pipe.run_single_prompt(
        prompt=prompt,
        negative_prompt=" ",
        # negative_prompt="blurry, low quality, low contrast",
        num_inference_steps=num_steps,
        seed=seed,
    )

    # Basic validation
    assert len(images) == 1, "Should generate exactly one image"
    assert images[0].size == (512, 512), f"Image size should be 512x512, got {images[0].size}"

    # Save TT test image for visual inspection
    images[0].save("test_sd35_medium_tt_output.png")
    logger.info("TT image saved to test_sd35_medium_tt_output.png")
    logger.info("TT image generation complete. Check test_sd35_medium_tt_output.png")
