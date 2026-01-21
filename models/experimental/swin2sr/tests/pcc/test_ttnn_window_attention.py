# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.experimental.swin2sr.reference.window_attention import WindowAttention as TorchWindowAttention
from models.experimental.swin2sr.tt.tt_window_attention import TtSwin2SRWindowAttention
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def to_2tuple(x):
    """Convert input to a tuple of 2 elements."""
    if isinstance(x, (int, float)):
        return (x, x)
    return x


def create_window_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchWindowAttention):
            # QKV projection
            qkv_weight = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16)
            parameters["qkv"] = {
                "weight": ttnn.to_device(qkv_weight, device),
            }

            # Q and V biases (no K bias in SwinV2)
            if torch_model.q_bias is not None:
                parameters["q_bias"] = ttnn.from_torch(
                    torch_model.q_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                )
            if torch_model.v_bias is not None:
                parameters["v_bias"] = ttnn.from_torch(
                    torch_model.v_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                )

            # Logit scale
            if torch_model.logit_scale is not None:
                parameters["logit_scale"] = ttnn.from_torch(
                    torch_model.logit_scale, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                )

            # Pre-compute relative position bias (like SwinV2)
            parameters["relative_position_bias"] = ttnn.from_torch(
                torch_model.get_relative_position_bias(), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )

            # Output projection
            proj_weight = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"] = {
                "weight": ttnn.to_device(proj_weight, device),
            }
            if torch_model.proj.bias is not None:
                proj_bias = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)
                parameters["proj"]["bias"] = ttnn.to_device(proj_bias, device)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "dim,window_size,num_heads,batch_windows,seq_len",
    [
        (96, (7, 7), 3, 16, 49),  # dim=96, num_heads=3 → head_dim=32 (multiple of 32)
        (192, (7, 7), 6, 16, 49),  # dim=192, num_heads=6 → head_dim=32 (multiple of 32)
        (128, (7, 7), 4, 16, 49),  # dim=128, num_heads=4 → head_dim=32 (multiple of 32)
    ],
)
def test_swin2sr_window_attention_ttnn_vs_torch(
    device, dim, window_size, num_heads, batch_windows, seq_len, reset_seeds
):
    """Test Swin2SR window attention TTNN vs PyTorch."""
    window_size = to_2tuple(window_size)

    torch_model = TorchWindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    torch_model.eval()

    # Create input tensor: (num_windows*B, N, C)
    torch_input_tensor = torch.randn(batch_windows, seq_len, dim)

    # Create mask if needed (for shifted windows)
    mask = None

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor, mask=mask)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    ttnn_model = TtSwin2SRWindowAttention(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Convert input to TTNN
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Convert mask if provided
    tt_mask = None
    if mask is not None:
        tt_mask = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    # Run TTNN model
    output_tensor = ttnn_model(input_tensor, mask=tt_mask)
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs and log PCC
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"\n[SYNTHETIC - dim={dim}, num_heads={num_heads}, head_dim={dim//num_heads}] PCC: {pcc_message}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def test_swin2sr_window_attention_with_mask(device, reset_seeds):
    """Test window attention with mask (for shifted windows)."""
    dim = 96
    window_size = (7, 7)
    num_heads = 3  # Changed to 3 so head_dim = 32 (multiple of 32)
    batch_windows = 16
    seq_len = 49

    torch_model = TorchWindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    torch_model.eval()

    torch_input_tensor = torch.randn(batch_windows, seq_len, dim)

    # Create a simple mask for testing
    nW = batch_windows
    mask = torch.zeros((nW, seq_len, seq_len))

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor, mask=mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
    )

    ttnn_model = TtSwin2SRWindowAttention(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_model(input_tensor, mask=tt_mask)
    output_tensor = ttnn.to_torch(output_tensor)

    # Log PCC value
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"\n[SYNTHETIC - With Mask] PCC: {pcc_message}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def load_window_attention_weights_from_checkpoint(checkpoint_path, layer_idx=0, block_idx=0):
    """Load window attention weights from Swin2SR checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        layer_idx: Layer index (0-based).
        block_idx: Block index within the layer (0-based).

    Returns:
        Dictionary containing weights and model configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = f"layers.{layer_idx}.residual_group.blocks.{block_idx}.attn"

    # Extract weights
    weights = {
        "qkv_weight": params[f"{prefix}.qkv.weight"],
        "q_bias": params.get(f"{prefix}.q_bias", None),
        "v_bias": params.get(f"{prefix}.v_bias", None),
        "proj_weight": params[f"{prefix}.proj.weight"],
        "proj_bias": params.get(f"{prefix}.proj.bias", None),
        "logit_scale": params.get(f"{prefix}.logit_scale", None),
        "cpb_mlp_fc1_weight": params[f"{prefix}.cpb_mlp.0.weight"],
        "cpb_mlp_fc1_bias": params.get(f"{prefix}.cpb_mlp.0.bias", None),
        "cpb_mlp_fc2_weight": params[f"{prefix}.cpb_mlp.2.weight"],
    }

    # Extract model configuration from weights
    dim = weights["qkv_weight"].shape[1]  # Input dimension
    qkv_out_dim = weights["qkv_weight"].shape[0]  # Should be 3 * dim

    # Infer num_heads from logit_scale shape or qkv weight
    if weights["logit_scale"] is not None:
        num_heads = weights["logit_scale"].shape[0]
    else:
        # Fallback: assume standard head_dim calculation
        head_dim = dim // 6  # Common default
        num_heads = dim // head_dim

    weights["dim"] = dim
    weights["num_heads"] = num_heads

    return weights


@pytest.mark.parametrize(
    "layer_idx,block_idx",
    [
        (0, 0),
        (0, 1),
        (1, 0),
    ],
)
def test_swin2sr_window_attention_ttnn_vs_torch_with_checkpoint(device, layer_idx, block_idx, reset_seeds):
    """Test window attention with weights from Swin2SR checkpoint."""
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights = load_window_attention_weights_from_checkpoint(checkpoint_path, layer_idx=layer_idx, block_idx=block_idx)

    dim = weights["dim"]
    num_heads = weights["num_heads"]
    window_size = (7, 7)  # Standard Swin2SR window size
    head_dim = dim // num_heads

    # Note: We now support any head_dim (not just multiples of 32) by using manual QKV splitting
    # The implementation uses torch operations for splitting, then converts back to TTNN

    # Create PyTorch model
    torch_model = TorchWindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # Load weights
    torch_model.qkv.weight.data.copy_(weights["qkv_weight"])
    if weights["q_bias"] is not None:
        torch_model.q_bias.data.copy_(weights["q_bias"])
    if weights["v_bias"] is not None:
        torch_model.v_bias.data.copy_(weights["v_bias"])
    torch_model.proj.weight.data.copy_(weights["proj_weight"])
    if weights["proj_bias"] is not None:
        torch_model.proj.bias.data.copy_(weights["proj_bias"])
    if weights["logit_scale"] is not None:
        torch_model.logit_scale.data.copy_(weights["logit_scale"])

    # Load CPB MLP weights
    torch_model.cpb_mlp[0].weight.data.copy_(weights["cpb_mlp_fc1_weight"])
    if weights["cpb_mlp_fc1_bias"] is not None:
        torch_model.cpb_mlp[0].bias.data.copy_(weights["cpb_mlp_fc1_bias"])
    torch_model.cpb_mlp[2].weight.data.copy_(weights["cpb_mlp_fc2_weight"])

    torch_model.eval()

    # Create input tensor
    batch_windows = 4  # Smaller batch for checkpoint test
    seq_len = window_size[0] * window_size[1]  # 49
    batch_size = batch_windows
    torch_input_tensor = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
    )

    # Create TTNN model
    ttnn_model = TtSwin2SRWindowAttention(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
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
    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Compare outputs and log PCC
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"\n[CHECKPOINT - Layer {layer_idx}, Block {block_idx}] PCC: {pcc_message}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
