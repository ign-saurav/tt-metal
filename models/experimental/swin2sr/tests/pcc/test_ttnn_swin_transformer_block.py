# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.experimental.swin2sr.reference.swin_transformer_block import (
    SwinTransformerBlock as TorchSwinTransformerBlock,
)
from models.experimental.swin2sr.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from models.experimental.swin2sr.tests.pcc.test_ttnn_window_attention import create_window_attention_preprocessor
from models.experimental.swin2sr.tests.pcc.test_ttnn_mlp import create_custom_preprocessor as create_mlp_preprocessor
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchSwinTransformerBlock):
            # Norm1 - use ttnn.from_torch with device to ensure tensor is on device
            parameters["norm1"] = {}
            parameters["norm1"]["weight"] = ttnn.from_torch(
                torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm1"]["bias"] = ttnn.from_torch(
                torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Attention
            attn_preprocessor = create_window_attention_preprocessor(device)
            parameters["attn"] = attn_preprocessor(torch_model.attn, None, None)

            # Norm2 - use ttnn.from_torch with device to ensure tensor is on device
            parameters["norm2"] = {}
            parameters["norm2"]["weight"] = ttnn.from_torch(
                torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            parameters["norm2"]["bias"] = ttnn.from_torch(
                torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # MLP
            mlp_preprocessor = create_mlp_preprocessor(device)
            parameters["mlp"] = mlp_preprocessor(torch_model.mlp, None, None)

        return parameters

    return custom_preprocessor


def load_swin_transformer_block_weights_from_checkpoint(checkpoint_path, layer_idx=0, block_idx=0):
    """Load SwinTransformerBlock weights from Swin2SR checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        layer_idx: Layer index (0-based).
        block_idx: Block index within the layer (0-based).

    Returns:
        Dictionary containing weights and model configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = f"layers.{layer_idx}.residual_group.blocks.{block_idx}"

    # Extract weights
    weights = {
        "norm1_weight": params[f"{prefix}.norm1.weight"],
        "norm1_bias": params[f"{prefix}.norm1.bias"],
        "norm2_weight": params[f"{prefix}.norm2.weight"],
        "norm2_bias": params[f"{prefix}.norm2.bias"],
        # Attention weights
        "attn_qkv_weight": params[f"{prefix}.attn.qkv.weight"],
        "attn_q_bias": params.get(f"{prefix}.attn.q_bias", None),
        "attn_v_bias": params.get(f"{prefix}.attn.v_bias", None),
        "attn_proj_weight": params[f"{prefix}.attn.proj.weight"],
        "attn_proj_bias": params.get(f"{prefix}.attn.proj.bias", None),
        "attn_logit_scale": params.get(f"{prefix}.attn.logit_scale", None),
        "attn_cpb_mlp_fc1_weight": params[f"{prefix}.attn.cpb_mlp.0.weight"],
        "attn_cpb_mlp_fc1_bias": params.get(f"{prefix}.attn.cpb_mlp.0.bias", None),
        "attn_cpb_mlp_fc2_weight": params[f"{prefix}.attn.cpb_mlp.2.weight"],
        # MLP weights
        "mlp_fc1_weight": params[f"{prefix}.mlp.fc1.weight"],
        "mlp_fc1_bias": params[f"{prefix}.mlp.fc1.bias"],
        "mlp_fc2_weight": params[f"{prefix}.mlp.fc2.weight"],
        "mlp_fc2_bias": params[f"{prefix}.mlp.fc2.bias"],
    }

    # Extract model configuration from weights
    dim = weights["norm1_weight"].shape[0]
    if weights["attn_logit_scale"] is not None:
        num_heads = weights["attn_logit_scale"].shape[0]
    else:
        num_heads = 3  # Default fallback

    # Calculate mlp_ratio from fc1 weight shape: (hidden_dim, dim)
    mlp_hidden_dim = weights["mlp_fc1_weight"].shape[0]
    mlp_ratio = mlp_hidden_dim / dim

    # Calculate window_size from relative_position_index shape: (window_size^2, window_size^2)
    rel_pos_idx_key = f"{prefix}.attn.relative_position_index"
    if rel_pos_idx_key in params:
        rel_pos_shape = params[rel_pos_idx_key].shape[0]
        window_size = int(rel_pos_shape**0.5)
    else:
        window_size = 8  # Default for Swin2SR

    weights["dim"] = dim
    weights["num_heads"] = num_heads
    weights["mlp_ratio"] = mlp_ratio
    weights["window_size"] = window_size

    return weights


@pytest.mark.parametrize(
    "layer_idx,block_idx",
    [
        (0, 0),
        (0, 1),
        (1, 0),
    ],
)
def test_swin_transformer_block_ttnn_vs_torch_with_checkpoint(device, layer_idx, block_idx, reset_seeds):
    """Test SwinTransformerBlock with weights from Swin2SR checkpoint."""
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights = load_swin_transformer_block_weights_from_checkpoint(
        checkpoint_path, layer_idx=layer_idx, block_idx=block_idx
    )

    dim = weights["dim"]
    num_heads = weights["num_heads"]
    mlp_ratio = weights["mlp_ratio"]  # Extract from checkpoint instead of hardcoding
    window_size = weights["window_size"]  # Extract from checkpoint (8 for Swin2SR)
    shift_size = 0 if block_idx % 2 == 0 else window_size // 2
    # Use resolution divisible by window_size
    input_resolution = (64, 64)  # 64 is divisible by 8

    # Create PyTorch model
    torch_model = TorchSwinTransformerBlock(
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )

    # Load weights
    torch_model.norm1.weight.data.copy_(weights["norm1_weight"])
    torch_model.norm1.bias.data.copy_(weights["norm1_bias"])
    torch_model.norm2.weight.data.copy_(weights["norm2_weight"])
    torch_model.norm2.bias.data.copy_(weights["norm2_bias"])

    # Load attention weights
    torch_model.attn.qkv.weight.data.copy_(weights["attn_qkv_weight"])
    if weights["attn_q_bias"] is not None:
        torch_model.attn.q_bias.data.copy_(weights["attn_q_bias"])
    if weights["attn_v_bias"] is not None:
        torch_model.attn.v_bias.data.copy_(weights["attn_v_bias"])
    torch_model.attn.proj.weight.data.copy_(weights["attn_proj_weight"])
    if weights["attn_proj_bias"] is not None:
        torch_model.attn.proj.bias.data.copy_(weights["attn_proj_bias"])
    if weights["attn_logit_scale"] is not None:
        torch_model.attn.logit_scale.data.copy_(weights["attn_logit_scale"])

    # Load CPB MLP weights
    torch_model.attn.cpb_mlp[0].weight.data.copy_(weights["attn_cpb_mlp_fc1_weight"])
    if weights["attn_cpb_mlp_fc1_bias"] is not None:
        torch_model.attn.cpb_mlp[0].bias.data.copy_(weights["attn_cpb_mlp_fc1_bias"])
    torch_model.attn.cpb_mlp[2].weight.data.copy_(weights["attn_cpb_mlp_fc2_weight"])

    # Load MLP weights
    torch_model.mlp.fc1.weight.data.copy_(weights["mlp_fc1_weight"])
    torch_model.mlp.fc1.bias.data.copy_(weights["mlp_fc1_bias"])
    torch_model.mlp.fc2.weight.data.copy_(weights["mlp_fc2_weight"])
    torch_model.mlp.fc2.bias.data.copy_(weights["mlp_fc2_bias"])

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
    ttnn_model = TtSwinTransformerBlock(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Convert input to TTNN
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    output_tensor = ttnn_model(input_tensor, x_size=input_resolution)
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs and log PCC
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"\n[CHECKPOINT - SwinTransformerBlock Layer {layer_idx}, Block {block_idx}] PCC: {pcc_message}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "dim,num_heads,window_size,shift_size,input_resolution",
    [
        (96, 3, 7, 0, (28, 28)),  # 28 is divisible by 7
        (96, 3, 7, 3, (28, 28)),
        (192, 6, 7, 0, (14, 14)),  # 14 is divisible by 7
        (192, 6, 7, 3, (14, 14)),
    ],
)
def test_swin_transformer_block_ttnn_vs_torch(
    device, dim, num_heads, window_size, shift_size, input_resolution, reset_seeds
):
    """Test SwinTransformerBlock TTNN vs PyTorch."""
    mlp_ratio = 4.0

    torch_model = TorchSwinTransformerBlock(
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )
    torch_model.eval()

    # Create input tensor
    H, W = input_resolution
    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, H * W, dim)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor, x_size=input_resolution)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    ttnn_model = TtSwinTransformerBlock(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Convert input to TTNN
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    output_tensor = ttnn_model(input_tensor, x_size=input_resolution)
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs and log PCC
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(
        f"\n[SYNTHETIC - SwinTransformerBlock dim={dim}, num_heads={num_heads}, window_size={window_size}, shift_size={shift_size}] PCC: {pcc_message}"
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
