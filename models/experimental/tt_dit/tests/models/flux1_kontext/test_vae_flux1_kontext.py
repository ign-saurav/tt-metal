# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ....utils.check import assert_quality
from ....models.vae import vae_flux1_kontext as vae_flux1
from ....parallel.manager import CCLManager
from ....parallel.config import vae_all_gather, VAEParallelConfig, ParallelFactor
from time import time
from loguru import logger
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


# Custom pytest mark for shared VAE device configuration
def vae_device_config(func):
    """Decorator to apply standard VAE device configuration to tests"""
    func = pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], ids=["t3k", "tg"], indirect=True)(func)
    func = pytest.mark.parametrize("submesh_shape", [(1, 4)])(func)
    func = pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
        indirect=True,
    )(func)
    return func


def skip_invalid_submesh_shape(mesh_device: ttnn.Device, submesh_shape: tuple[int, int]):
    mesh_device_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > mesh_device_shape[0] or submesh_shape[1] > mesh_device_shape[1]:
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")


@vae_device_config
@pytest.mark.parametrize(
    (
        "batch",
        "in_channels",
        "height",
        "width",
    ),
    [
        (1, 3, 1024, 1024),
    ],
)
def test_flux1kontext_vae_encoder(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: tuple[int, int],
    batch: int,
    in_channels: int,
    height: int,
    width: int,
):
    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="vae").encoder
    torch_model.eval()

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_flux1.VAEEncoder.from_torch(
        torch_ref=torch_model, mesh_device=submesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=submesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.99_000)

    start = time()
    tt_out = tt_model(tt_input_tensor)
    ttnn.synchronize_device(submesh_device)
    logger.info(f"VAE Encoder Time taken: {time() - start}")


@vae_device_config
@pytest.mark.parametrize(
    (
        "batch",
        "in_channels",
        "height",
        "width",
    ),
    [
        (1, 16, 128, 128),
    ],
)
def test_flux1kontext_vae_decoder(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: tuple[int, int],
    batch: int,
    in_channels: int,
    height: int,
    width: int,
):
    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="vae").decoder
    torch_model.eval()

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_flux1.VAEDecoder.from_torch(
        torch_ref=torch_model, mesh_device=submesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=submesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.99_000)

    start = time()
    tt_out = tt_model(tt_input_tensor)
    ttnn.synchronize_device(submesh_device)
    logger.info(f"VAE Decoder Time taken: {time() - start}")
