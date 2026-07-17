# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.experimental.swin2sr.reference.rstb import RSTB as TorchRSTB
from models.experimental.swin2sr.tt.tt_rstb import TtRSTB
from models.experimental.swin2sr.tests.pcc.test_ttnn_basic_layer import (
    create_custom_preprocessor as create_basic_layer_preprocessor,
    load_basic_layer_weights_from_checkpoint,
)
from models.experimental.swin2sr.tests.pcc.test_ttnn_patch_embed import (
    create_custom_preprocessor as create_patch_embed_preprocessor,
)
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchRSTB):
            # Residual group (BasicLayer)
            basic_layer_preprocessor = create_basic_layer_preprocessor(device)
            parameters["residual_group"] = basic_layer_preprocessor(torch_model.residual_group, None, None)

            # Convolution
            if isinstance(torch_model.conv, torch.nn.Conv2d):
                # 1conv case
                conv_weight = torch_model.conv.weight
                conv_bias = (
                    torch_model.conv.bias
                    if torch_model.conv.bias is not None
                    else torch.zeros(torch_model.conv.out_channels)
                )
                parameters["conv"] = {
                    "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, device=device),
                    "bias": ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, device=device
                    ),
                }
            else:
                # 3conv case (Sequential)
                parameters["conv"] = []
                for i, layer in enumerate(torch_model.conv):
                    if isinstance(layer, torch.nn.Conv2d):
                        conv_weight = layer.weight
                        conv_bias = layer.bias if layer.bias is not None else torch.zeros(layer.out_channels)
                        parameters["conv"].append(
                            {
                                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, device=device),
                                "bias": ttnn.from_torch(
                                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, device=device
                                ),
                            }
                        )

            # Patch embed
            patch_embed_preprocessor = create_patch_embed_preprocessor(device)
            parameters["patch_embed"] = patch_embed_preprocessor(torch_model.patch_embed, None, None)

            # Patch unembed doesn't need parameters (it's just reshape/permute)

        return parameters

    return custom_preprocessor


def load_rstb_weights_from_checkpoint(checkpoint_path, layer_idx=0):
    """Load RSTB weights from Swin2SR checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        layer_idx: Layer index (0-based).

    Returns:
        Dictionary containing weights and model configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = f"layers.{layer_idx}"

    # Load residual_group (BasicLayer) weights
    basic_layer_weights = load_basic_layer_weights_from_checkpoint(checkpoint_path, layer_idx=layer_idx)

    # Load conv weights
    conv_weight = params[f"{prefix}.conv.weight"]
    conv_bias = params.get(f"{prefix}.conv.bias", None)
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.shape[0])

    # Check if it's 3conv by checking for conv.0.weight
    is_3conv = f"{prefix}.conv.0.weight" in params

    if is_3conv:
        conv_weights = []
        for i in [0, 2, 4]:  # Conv layers at indices 0, 2, 4 (LeakyReLU at 1, 3)
            conv_w = params[f"{prefix}.conv.{i}.weight"]
            conv_b = params.get(f"{prefix}.conv.{i}.bias", None)
            if conv_b is None:
                conv_b = torch.zeros(conv_w.shape[0])
            conv_weights.append({"weight": conv_w, "bias": conv_b})
    else:
        conv_weights = {"weight": conv_weight, "bias": conv_bias}

    # Load patch_embed weights
    patch_embed_proj_weight = params[f"{prefix}.patch_embed.proj.weight"]
    patch_embed_proj_bias = params.get(f"{prefix}.patch_embed.proj.bias", None)
    if patch_embed_proj_bias is None:
        patch_embed_proj_bias = torch.zeros(patch_embed_proj_weight.shape[0])

    weights = {
        "dim": basic_layer_weights["dim"],
        "num_heads": basic_layer_weights["num_heads"],
        "mlp_ratio": basic_layer_weights["mlp_ratio"],
        "window_size": basic_layer_weights["window_size"],
        "depth": basic_layer_weights["depth"],
        "residual_group": basic_layer_weights,
        "conv": conv_weights,
        "patch_embed": {
            "proj_weight": patch_embed_proj_weight,
            "proj_bias": patch_embed_proj_bias,
        },
        "is_3conv": is_3conv,
    }

    return weights


@pytest.mark.parametrize(
    "layer_idx",
    [
        0,
        1,
    ],
)
def test_rstb_ttnn_vs_torch_with_checkpoint(device, layer_idx, reset_seeds):
    """Test RSTB with weights from Swin2SR checkpoint."""
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights = load_rstb_weights_from_checkpoint(checkpoint_path, layer_idx=layer_idx)

    dim = weights["dim"]
    num_heads = weights["num_heads"]
    mlp_ratio = weights["mlp_ratio"]
    window_size = weights["window_size"]
    depth = weights["depth"]
    img_size = 64
    patch_size = 1
    # input_resolution should match patches_resolution = (img_size // patch_size, img_size // patch_size)
    patches_resolution = (img_size // patch_size, img_size // patch_size)
    input_resolution = patches_resolution
    resi_connection = "3conv" if weights["is_3conv"] else "1conv"

    # Create PyTorch model
    torch_model = TorchRSTB(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        img_size=img_size,
        patch_size=patch_size,
        resi_connection=resi_connection,
    )

    # Load residual_group weights

    for block_idx, block in enumerate(torch_model.residual_group.blocks):
        block_weights = weights["residual_group"]["blocks"][block_idx]

        block.norm1.weight.data.copy_(block_weights["norm1_weight"])
        block.norm1.bias.data.copy_(block_weights["norm1_bias"])
        block.norm2.weight.data.copy_(block_weights["norm2_weight"])
        block.norm2.bias.data.copy_(block_weights["norm2_bias"])

        block.attn.qkv.weight.data.copy_(block_weights["attn_qkv_weight"])
        if block_weights["attn_q_bias"] is not None:
            block.attn.q_bias.data.copy_(block_weights["attn_q_bias"])
        if block_weights["attn_v_bias"] is not None:
            block.attn.v_bias.data.copy_(block_weights["attn_v_bias"])
        block.attn.proj.weight.data.copy_(block_weights["attn_proj_weight"])
        if block_weights["attn_proj_bias"] is not None:
            block.attn.proj.bias.data.copy_(block_weights["attn_proj_bias"])
        if block_weights["attn_logit_scale"] is not None:
            block.attn.logit_scale.data.copy_(block_weights["attn_logit_scale"])

        block.attn.cpb_mlp[0].weight.data.copy_(block_weights["attn_cpb_mlp_fc1_weight"])
        if block_weights["attn_cpb_mlp_fc1_bias"] is not None:
            block.attn.cpb_mlp[0].bias.data.copy_(block_weights["attn_cpb_mlp_fc1_bias"])
        block.attn.cpb_mlp[2].weight.data.copy_(block_weights["attn_cpb_mlp_fc2_weight"])

        block.mlp.fc1.weight.data.copy_(block_weights["mlp_fc1_weight"])
        block.mlp.fc1.bias.data.copy_(block_weights["mlp_fc1_bias"])
        block.mlp.fc2.weight.data.copy_(block_weights["mlp_fc2_weight"])
        block.mlp.fc2.bias.data.copy_(block_weights["mlp_fc2_bias"])

    # Load conv weights
    if resi_connection == "1conv":
        torch_model.conv.weight.data.copy_(weights["conv"]["weight"])
        torch_model.conv.bias.data.copy_(weights["conv"]["bias"])
    else:  # 3conv
        for i, conv_idx in enumerate([0, 2, 4]):
            torch_model.conv[conv_idx].weight.data.copy_(weights["conv"][i]["weight"])
            torch_model.conv[conv_idx].bias.data.copy_(weights["conv"][i]["bias"])

    # Load patch_embed weights
    torch_model.patch_embed.proj.weight.data.copy_(weights["patch_embed"]["proj_weight"])
    torch_model.patch_embed.proj.bias.data.copy_(weights["patch_embed"]["proj_bias"])

    torch_model.eval()

    # Create input tensor
    H, W = input_resolution
    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, H * W, dim)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor, x_size=input_resolution)

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    ttnn_model = TtRSTB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        img_size=img_size,
        patch_size=patch_size,
        resi_connection=resi_connection,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create TTNN input tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN model
    output_tensor = ttnn_model(input_tensor, x_size=input_resolution)

    # Convert output to torch
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs
    pcc = comp_pcc(torch_output_tensor, output_tensor)
    logger.info(
        f"[CHECKPOINT - RSTB layer_idx={layer_idx}, depth={depth}, resi_connection={resi_connection}] PCC: {pcc}"
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "dim,num_heads,window_size,depth,img_size,patch_size,resi_connection",
    [
        (180, 6, 8, 6, 64, 1, "1conv"),
        (192, 4, 8, 4, 64, 1, "1conv"),
        (96, 6, 4, 6, 64, 1, "1conv"),
        (180, 6, 8, 6, 64, 1, "3conv"),
        (96, 6, 4, 6, 64, 1, "3conv"),
    ],
)
def test_rstb_ttnn_vs_torch(
    device, dim, num_heads, window_size, depth, img_size, patch_size, resi_connection, reset_seeds
):
    """Test RSTB TTNN vs PyTorch."""
    mlp_ratio = 4.0
    # input_resolution should match patches_resolution = (img_size // patch_size, img_size // patch_size)
    patches_resolution = (img_size // patch_size, img_size // patch_size)
    input_resolution = patches_resolution

    torch_model = TorchRSTB(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        img_size=img_size,
        patch_size=patch_size,
        resi_connection=resi_connection,
    )
    torch_model.eval()

    # Create input tensor
    H, W = input_resolution
    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, H * W, dim)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor, x_size=input_resolution)

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    ttnn_model = TtRSTB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        img_size=img_size,
        patch_size=patch_size,
        resi_connection=resi_connection,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create TTNN input tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN model
    output_tensor = ttnn_model(input_tensor, x_size=input_resolution)

    # Convert output to torch
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs
    pcc = comp_pcc(torch_output_tensor, output_tensor)
    logger.info(
        f"[SYNTHETIC - RSTB dim={dim}, num_heads={num_heads}, window_size={window_size}, depth={depth}, resi_connection={resi_connection}] PCC: {pcc}"
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
