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
from models.experimental.swin2sr.reference.basic_layer import BasicLayer as TorchBasicLayer
from models.experimental.swin2sr.tt.tt_basic_layer import TtBasicLayer
from models.experimental.swin2sr.tests.pcc.test_ttnn_swin_transformer_block import (
    create_custom_preprocessor as create_block_preprocessor,
    load_swin_transformer_block_weights_from_checkpoint,
)
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchBasicLayer):
            parameters["blocks"] = []
            for i, block in enumerate(torch_model.blocks):
                block_preprocessor = create_block_preprocessor(device)
                block_params = block_preprocessor(block, None, None)
                parameters["blocks"].append(block_params)

        return parameters

    return custom_preprocessor


def load_basic_layer_weights_from_checkpoint(checkpoint_path, layer_idx=0):
    """Load BasicLayer weights from Swin2SR checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        layer_idx: Layer index (0-based).

    Returns:
        Dictionary containing weights and model configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = f"layers.{layer_idx}.residual_group"

    # Load first block to extract configuration
    first_block_weights = load_swin_transformer_block_weights_from_checkpoint(
        checkpoint_path, layer_idx=layer_idx, block_idx=0
    )

    # Count number of blocks by checking how many blocks exist
    depth = 0
    while f"{prefix}.blocks.{depth}.norm1.weight" in params:
        depth += 1

    if depth == 0:
        raise ValueError(f"No blocks found for layer {layer_idx}")

    weights = {
        "dim": first_block_weights["dim"],
        "num_heads": first_block_weights["num_heads"],
        "mlp_ratio": first_block_weights["mlp_ratio"],
        "window_size": first_block_weights["window_size"],
        "depth": depth,
    }

    # Load all block weights
    weights["blocks"] = []
    for block_idx in range(depth):
        block_weights = load_swin_transformer_block_weights_from_checkpoint(
            checkpoint_path, layer_idx=layer_idx, block_idx=block_idx
        )
        weights["blocks"].append(block_weights)

    return weights


@pytest.mark.parametrize(
    "layer_idx",
    [
        0,
        1,
    ],
)
def test_basic_layer_ttnn_vs_torch_with_checkpoint(device, layer_idx, reset_seeds):
    """Test BasicLayer with weights from Swin2SR checkpoint."""
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights = load_basic_layer_weights_from_checkpoint(checkpoint_path, layer_idx=layer_idx)

    dim = weights["dim"]
    num_heads = weights["num_heads"]
    mlp_ratio = weights["mlp_ratio"]
    window_size = weights["window_size"]
    depth = weights["depth"]
    input_resolution = (64, 64)

    # Create PyTorch model
    torch_model = TorchBasicLayer(
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
    )

    # Load weights for all blocks
    for block_idx, block in enumerate(torch_model.blocks):
        block_weights = weights["blocks"][block_idx]

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
    ttnn_model = TtBasicLayer(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
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
    logger.info(f"[CHECKPOINT - BasicLayer layer_idx={layer_idx}, depth={depth}] PCC: {pcc}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "dim,num_heads,window_size,depth,input_resolution",
    [
        (96, 3, 7, 2, (28, 28)),
        (96, 3, 7, 4, (28, 28)),
        (192, 6, 7, 2, (14, 14)),
        (192, 6, 7, 4, (14, 14)),
    ],
)
def test_basic_layer_ttnn_vs_torch(device, dim, num_heads, window_size, depth, input_resolution, reset_seeds):
    """Test BasicLayer TTNN vs PyTorch."""
    mlp_ratio = 4.0

    torch_model = TorchBasicLayer(
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
    ttnn_model = TtBasicLayer(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
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
        f"[SYNTHETIC - BasicLayer dim={dim}, num_heads={num_heads}, window_size={window_size}, depth={depth}] PCC: {pcc}"
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
