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
@pytest.mark.parametrize(
    "image_size, spatial_sequence_length",
    [
        (512, 1024),
        (1024, 4096),
    ],
    ids=["512x512", "1024x1024"],
)
def test_sd35_medium_pipeline_functional(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    image_size: int,
    spatial_sequence_length: int,
) -> None:
    """Functional test for SD3.5 Medium pipeline on N300 with CFG enabled."""
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=2, mesh_axis=1),  # CFG parallel on axis 1 for N300
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=sp_axis),
    )

    # Test with a simple prompt
    prompt = "A capybara wearing a suit holding a sign that reads hello world"
    seed = 23
    num_steps = 40

    tt_pipe = TTSD35MediumPipeline(
        mesh_device=mesh_device,
        enable_t5_text_encoder=False,
        guidance_cond=2,  # CFG enabled: positive + negative prompt
        parallel_config=parallel_config,
        num_links=num_links,
        height=image_size,
        width=image_size,
        model_checkpoint_path="stabilityai/stable-diffusion-3.5-medium",
        use_cache=False,
    )

    tt_pipe.prepare(
        batch_size=1,
        num_images_per_prompt=1,
        width=image_size,
        height=image_size,
        guidance_scale=7,
        max_t5_sequence_length=256,
        prompt_sequence_length=333,
        spatial_sequence_length=spatial_sequence_length,
    )

    images = tt_pipe.run_single_prompt(
        prompt=prompt,
        negative_prompt="blurry image",
        num_inference_steps=num_steps,
        seed=seed,
    )

    assert len(images) == 1, "Should generate exactly one image"
    assert images[0].size == (
        image_size,
        image_size,
    ), f"Image size should be {image_size}x{image_size}, got {images[0].size}"
    images[0].save(f"test_sd35_medium_tt_output_{image_size}.png")
    logger.info(f"TT {image_size}x{image_size} image saved to test_sd35_medium_tt_output_{image_size}.png")
