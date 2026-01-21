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
    "mesh_device, sp_axis, tp_axis, num_links, cfg_factor, device_params",
    [
        [
            (1, 1),
            0,
            0,
            1,
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED, "l1_small_size": 32768},
        ],  # N150 configuration - 1 device, no CFG parallel, no fabric needed
        [
            (1, 2),
            0,
            1,
            1,
            2,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768},
        ],  # N300 configuration - 2 devices with CFG parallel
    ],
    ids=["1x1_n150", "1x2_n300"],
    indirect=["mesh_device", "device_params"],
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
    cfg_factor: int,
    image_size: int,
    spatial_sequence_length: int,
) -> None:
    """Functional test for SD3.5 Medium pipeline on N150 and N300."""
    # CFG parallel: factor=2 for N300 (2 devices), factor=1 for N150 (1 device)
    cfg_mesh_axis = 1 if cfg_factor > 1 else 0
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_mesh_axis),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=sp_axis),
    )

    # Test with a simple prompt
    prompt = "A capybara wearing a suit holding a sign that reads hello world"
    seed = 23
    num_steps = 40

    # guidance_cond: 2 for CFG (positive + negative prompts), 1 for no CFG
    # For N150 (cfg_factor=1): guidance_cond=2 (process both prompts on same device)
    # For N300 (cfg_factor=2): guidance_cond=2 (CFG parallel across devices)
    guidance_cond = 2 if cfg_factor == 1 else cfg_factor

    tt_pipe = TTSD35MediumPipeline(
        mesh_device=mesh_device,
        enable_t5_text_encoder=False,
        guidance_cond=guidance_cond,
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
