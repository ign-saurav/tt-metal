# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import pytest

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.swin2sr.reference.swin2sr import Swin2SR as TorchSwin2SR
from models.experimental.swin2sr.tt.tt_swin2sr import TtSwin2SR
from models.experimental.swin2sr.tt.utils import get_checkpoint_path
from models.experimental.swin2sr.tests.pcc.test_ttnn_rstb import (
    create_custom_preprocessor as create_rstb_preprocessor,
)
from models.experimental.swin2sr.tests.pcc.test_ttnn_patch_embed import (
    create_custom_preprocessor as create_patch_embed_preprocessor,
)
from models.experimental.swin2sr.tests.pcc.test_ttnn_upsample import (
    create_upsample_parameters,
)


def create_swin2sr_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchSwin2SR):
            conv_first_weight = torch_model.conv_first.weight
            conv_first_bias = (
                torch_model.conv_first.bias
                if torch_model.conv_first.bias is not None
                else torch.zeros(torch_model.conv_first.out_channels)
            )
            parameters["conv_first"] = {
                "weight": ttnn.to_device(
                    ttnn.from_torch(conv_first_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                    device=device,
                ),
                "bias": ttnn.to_device(
                    ttnn.from_torch(
                        torch.reshape(conv_first_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                    device=device,
                ),
            }

            patch_embed_preprocessor = create_patch_embed_preprocessor(device)
            parameters["patch_embed"] = patch_embed_preprocessor(torch_model.patch_embed, None, None)

            parameters["layers"] = []
            rstb_preprocessor = create_rstb_preprocessor(device)
            for layer in torch_model.layers:
                layer_params = rstb_preprocessor(layer, None, None)
                parameters["layers"].append(layer_params)

            if torch_model.norm is not None:
                parameters["norm"] = {
                    "weight": ttnn.to_device(
                        ttnn.from_torch(
                            torch_model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                        ),
                        device=device,
                    ),
                    "bias": ttnn.to_device(
                        ttnn.from_torch(
                            torch_model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                        ),
                        device=device,
                    ),
                }

            if isinstance(torch_model.conv_after_body, nn.Conv2d):
                conv_after_weight = torch_model.conv_after_body.weight
                conv_after_bias = (
                    torch_model.conv_after_body.bias
                    if torch_model.conv_after_body.bias is not None
                    else torch.zeros(torch_model.conv_after_body.out_channels)
                )
                parameters["conv_after_body"] = {
                    "weight": ttnn.to_device(
                        ttnn.from_torch(conv_after_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                        device=device,
                    ),
                    "bias": ttnn.to_device(
                        ttnn.from_torch(
                            torch.reshape(conv_after_bias, (1, 1, 1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        device=device,
                    ),
                }
            else:  # Sequential (3conv)
                parameters["conv_after_body"] = []
                for i, layer in enumerate(torch_model.conv_after_body):
                    if isinstance(layer, nn.Conv2d):
                        conv_weight = layer.weight
                        conv_bias = layer.bias if layer.bias is not None else torch.zeros(layer.out_channels)
                        parameters["conv_after_body"].append(
                            {
                                "weight": ttnn.to_device(
                                    ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                                    device=device,
                                ),
                                "bias": ttnn.to_device(
                                    ttnn.from_torch(
                                        torch.reshape(conv_bias, (1, 1, 1, -1)),
                                        dtype=ttnn.bfloat16,
                                        layout=ttnn.ROW_MAJOR_LAYOUT,
                                    ),
                                    device=device,
                                ),
                            }
                        )

            conv_before_weight = torch_model.conv_before_upsample[0].weight
            conv_before_bias = (
                torch_model.conv_before_upsample[0].bias
                if torch_model.conv_before_upsample[0].bias is not None
                else torch.zeros(torch_model.conv_before_upsample[0].out_channels)
            )
            parameters["conv_before_upsample"] = [
                {
                    "weight": ttnn.to_device(
                        ttnn.from_torch(conv_before_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                        device=device,
                    ),
                    "bias": ttnn.to_device(
                        ttnn.from_torch(
                            torch.reshape(conv_before_bias, (1, 1, 1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        device=device,
                    ),
                }
            ]

            parameters["upsample"] = create_upsample_parameters(torch_model.upsample, device)

            conv_last_weight = torch_model.conv_last.weight
            conv_last_bias = (
                torch_model.conv_last.bias
                if torch_model.conv_last.bias is not None
                else torch.zeros(torch_model.conv_last.out_channels)
            )
            parameters["conv_last"] = {
                "weight": ttnn.to_device(
                    ttnn.from_torch(conv_last_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                    device=device,
                ),
                "bias": ttnn.to_device(
                    ttnn.from_torch(
                        torch.reshape(conv_last_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                    device=device,
                ),
            }

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "img_size,embed_dim,depths,num_heads,window_size,upscale,resi_connection",
    [
        (64, 180, (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6), 8, 2, "1conv"),
        (64, 180, (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6), 8, 2, "3conv"),
    ],
)
def test_swin2sr_ttnn_vs_torch(
    device, img_size, embed_dim, depths, num_heads, window_size, upscale, resi_connection, reset_seeds
):
    torch_model = TorchSwin2SR(
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.0,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection=resi_connection,
    )
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_swin2sr_preprocessor(device),
        device=device,
    )

    tt_model = TtSwin2SR(
        device=device,
        parameters=parameters,
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.0,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection=resi_connection,
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output_tensor = tt_model.forward(tt_input_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    if torch_output_tensor.shape != tt_output_tensor.shape:
        if tt_output_tensor.numel() == torch_output_tensor.numel():
            tt_output_tensor = tt_output_tensor.reshape(torch_output_tensor.shape)
        else:
            pytest.fail(
                f"Output size mismatch: torch shape={torch_output_tensor.shape} (size={torch_output_tensor.numel()}), "
                f"tt shape={tt_output_tensor.shape} (size={tt_output_tensor.numel()})"
            )

    # PCC threshold of 0.99 for full model with 36 transformer blocks (6 blocks × 6 layers)
    # Individual components achieve >0.99 PCC; accumulated precision loss is expected in deep bfloat16 models
    pcc_required = 0.99
    passed, pcc = comp_pcc(torch_output_tensor, tt_output_tensor, pcc_required)
    assert passed, f"PCC value {pcc} is lower than required {pcc_required}"


def test_swin2sr_checkpoint(device, reset_seeds):
    """Test with real Swin2SR checkpoint."""
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint

    # Parameters matching the Swin2SR_ClassicalSR_X2_64 checkpoint
    img_size = 64
    embed_dim = 180
    depths = (6, 6, 6, 6, 6, 6)
    num_heads = (6, 6, 6, 6, 6, 6)
    window_size = 8
    mlp_ratio = 2.0  # Checkpoint uses mlp_ratio=2.0
    upscale = 2
    resi_connection = "1conv"

    torch_model = TorchSwin2SR(
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

    torch_model.load_state_dict(params, strict=False)
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_swin2sr_preprocessor(device),
        device=device,
    )

    tt_model = TtSwin2SR(
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

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output_tensor = tt_model.forward(tt_input_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    # PCC threshold of 0.99 for full model with 36 transformer blocks (6 layers × 6 blocks)
    pcc_required = 0.99
    passed, pcc = comp_pcc(torch_output_tensor, tt_output_tensor, pcc_required)
    assert passed, f"PCC value {pcc} is lower than required {pcc_required}"
