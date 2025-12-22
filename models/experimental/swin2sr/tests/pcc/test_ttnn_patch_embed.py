# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from torch import nn
import pytest

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.swin2sr.reference.patch_embed import PatchEmbed as TorchSwin2SRPatchEmbed
from models.experimental.swin2sr.tt.tt_patch_embed import TtSwin2SRPatchEmbed


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchSwin2SRPatchEmbed):
            proj_weight = torch_model.proj.weight
            proj_bias = (
                torch_model.proj.bias
                if torch_model.proj.bias is not None
                else torch.zeros(torch_model.proj.out_channels)
            )

            parameters["proj"] = {
                "weight": ttnn.from_torch(proj_weight, dtype=ttnn.bfloat16, device=device),
                "bias": ttnn.from_torch(torch.reshape(proj_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, device=device),
            }

            if torch_model.norm is not None:
                parameters["norm"] = {
                    "weight": ttnn.from_torch(
                        torch_model.norm.weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                    ),
                    "bias": ttnn.from_torch(
                        torch_model.norm.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                    ),
                }
        return parameters

    return custom_preprocessor


def load_patch_embed_weights_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = "patch_embed"

    weights = {
        "proj_weight": params[f"{prefix}.proj.weight"],
        "proj_bias": params[f"{prefix}.proj.bias"],
    }

    if f"{prefix}.norm.weight" in params:
        weights["norm_weight"] = params[f"{prefix}.norm.weight"]
        weights["norm_bias"] = params[f"{prefix}.norm.bias"]
    else:
        weights["norm_weight"] = None
        weights["norm_bias"] = None

    return weights


@pytest.mark.parametrize(
    "img_size,patch_size,in_chans,embed_dim,use_norm",
    [
        (64, 4, 3, 96, False),
        (64, 4, 3, 96, True),
        (128, 4, 3, 180, False),
        (128, 4, 3, 180, True),
    ],
)
def test_swin2sr_patch_embed_ttnn_vs_torch(device, img_size, patch_size, in_chans, embed_dim, use_norm, reset_seeds):
    norm_layer = nn.LayerNorm if use_norm else None

    torch_model = TorchSwin2SRPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, in_chans, img_size, img_size)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    ttnn_model = TtSwin2SRPatchEmbed(
        device=device,
        parameters=parameters,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def test_swin2sr_patch_embed_ttnn_vs_torch_with_checkpoint(device, reset_seeds):
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "swin2sr",
        "model_zoo",
        "swin2sr",
        "Swin2SR_ClassicalSR_X2_64.pth",
    )

    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    weights = load_patch_embed_weights_from_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint

    proj_weight_shape = params["patch_embed.proj.weight"].shape
    out_channels, in_channels, kernel_h, kernel_w = proj_weight_shape

    img_size = 64
    patch_size = kernel_h
    in_chans = in_channels
    embed_dim = out_channels

    norm_layer = None
    if weights["norm_weight"] is not None:
        norm_layer = nn.LayerNorm

    torch_model = TorchSwin2SRPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )

    torch_model.proj.weight.data.copy_(weights["proj_weight"])
    torch_model.proj.bias.data.copy_(weights["proj_bias"])
    if norm_layer is not None:
        torch_model.norm.weight.data.copy_(weights["norm_weight"])
        torch_model.norm.bias.data.copy_(weights["norm_bias"])

    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, in_chans, img_size, img_size)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    ttnn_model = TtSwin2SRPatchEmbed(
        device=device,
        parameters=parameters,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
