# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

# Set environment variable BEFORE any imports that might trigger mmcv/cv2
os.environ["MMCV_DISABLE_OPENCV"] = "1"

import pytest
import torch
from pathlib import Path

import ttnn
from loguru import logger

from models.experimental.bevformerv2.reference.groupmultiheadattention import GroupMultiheadAttention
from models.experimental.bevformerv2.tt.tt_groupmultiheadattention import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


def _load_weights_from_checkpoint(layer_idx=0):
    """Load GroupMultiheadAttention weights from the checkpoint file."""
    weights_path = Path(__file__).parent.parent.parent / "resources" / "group_attention_weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    # Extract weights for the specified layer
    prefix = f"pts_bbox_head.transformer.decoder.layers.{layer_idx}.attentions.0.attn"

    weights = {
        "in_proj_weight": checkpoint[f"{prefix}.in_proj_weight"],
        "in_proj_bias": checkpoint[f"{prefix}.in_proj_bias"],
        "out_proj.weight": checkpoint[f"{prefix}.out_proj.weight"],
        "out_proj.bias": checkpoint[f"{prefix}.out_proj.bias"],
    }

    # Infer embed_dims and num_heads from weight shapes
    in_proj_weight = weights["in_proj_weight"]
    embed_dims = in_proj_weight.shape[1]  # [3*embed_dims, embed_dims]
    num_heads = 8  # Default, can be inferred from embed_dims if needed

    return weights, embed_dims, num_heads


def _custom_preprocessor(model, name):
    """Custom preprocessor for GroupMultiheadAttention parameters."""
    parameters = {}
    if isinstance(model, GroupMultiheadAttention):
        # Use ttnn.from_torch for in_proj_weight (like uniad does) to avoid transpose
        # This keeps the weight in [3*embed_dims, embed_dims] format
        parameters = {
            "attn": {
                "in_proj_weight": ttnn.from_torch(model.attn.in_proj_weight, dtype=ttnn.bfloat16),
                "in_proj_bias": preprocess_linear_bias(model.attn.in_proj_bias, dtype=ttnn.bfloat16),
                "out_proj": {
                    "weight": preprocess_linear_weight(model.attn.out_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.attn.out_proj.bias, dtype=ttnn.bfloat16),
                },
            }
        }
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx, batch_first, num_queries, batch_size",
    [
        (0, False, 100, 1),
        (0, True, 100, 2),
        (1, False, 100, 1),
    ],
)
def test_groupmultiheadattention(
    device,
    reset_seeds,
    layer_idx,
    batch_first,
    num_queries,
    batch_size,
):
    """Test TTNN GroupMultiheadAttention against reference implementation using real weights."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load weights from checkpoint
    weights, embed_dims, num_heads = _load_weights_from_checkpoint(layer_idx=layer_idx)

    logger.info(f"Testing with weights from layer {layer_idx}: " f"embed_dims={embed_dims}, num_heads={num_heads}")

    # Create reference model
    reference_model = GroupMultiheadAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        group=1,  # Group splitting is training-only, not used in inference
        dropout_layer=dict(type="Dropout", drop_prob=0.0),  # build_dropout requires "type" key
        batch_first=batch_first,
    )

    # Load weights into the reference model
    reference_model.attn.in_proj_weight.data = weights["in_proj_weight"]
    reference_model.attn.in_proj_bias.data = weights["in_proj_bias"]
    reference_model.attn.out_proj.weight.data = weights["out_proj.weight"]
    reference_model.attn.out_proj.bias.data = weights["out_proj.bias"]

    reference_model.eval()  # Set to eval mode (inference) - disables group splitting

    # Prepare parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
        custom_preprocessor=_custom_preprocessor,
    )

    # Create test inputs
    if batch_first:
        query = torch.randn(batch_size, num_queries, embed_dims)
        query_pos = torch.randn(batch_size, num_queries, embed_dims)
        key = torch.randn(batch_size, num_queries, embed_dims)
        key_pos = torch.randn(batch_size, num_queries, embed_dims)
        identity = torch.randn(batch_size, num_queries, embed_dims)
    else:
        query = torch.randn(num_queries, batch_size, embed_dims)
        query_pos = torch.randn(num_queries, batch_size, embed_dims)
        key = torch.randn(num_queries, batch_size, embed_dims)
        key_pos = torch.randn(num_queries, batch_size, embed_dims)
        identity = torch.randn(num_queries, batch_size, embed_dims)

    # Run reference model
    with torch.no_grad():
        reference_output = reference_model(
            query=query,
            key=key,
            value=None,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=None,
            key_padding_mask=None,
        )

    # Create TTNN model
    ttnn_model = TtMultiheadAttention(
        params=parameters,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=batch_first,
    )

    # Convert inputs to TTNN format
    ttnn_query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_key_pos = ttnn.from_torch(key_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_identity = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN model
    ttnn_output = ttnn_model(
        query=ttnn_query,
        key=ttnn_key,
        value=None,
        identity=ttnn_identity,
        query_pos=ttnn_query_pos,
        key_pos=ttnn_key_pos,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=batch_first,
    )

    # Convert TTNN output back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(reference_output, ttnn_output_torch, pcc=0.99)

    assert pcc_passed, logger.error(f"PCC check failed for GroupMultiheadAttention: {pcc_message}")
    logger.info(
        f"GroupMultiheadAttention PCC passed: {pcc_message} "
        f"(layer_idx={layer_idx}, embed_dims={embed_dims}, num_heads={num_heads}, "
        f"batch_first={batch_first}, num_queries={num_queries}, batch_size={batch_size})"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_groupmultiheadattention_self_attention(device, reset_seeds):
    """Test self-attention case (query=key=value) using real weights."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load weights from checkpoint
    weights, embed_dims, num_heads = _load_weights_from_checkpoint(layer_idx=0)

    num_queries = 100
    batch_size = 1

    # Create reference model
    reference_model = GroupMultiheadAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        batch_first=False,
    )

    # Load weights into the reference model
    reference_model.attn.in_proj_weight.data = weights["in_proj_weight"]
    reference_model.attn.in_proj_bias.data = weights["in_proj_bias"]
    reference_model.attn.out_proj.weight.data = weights["out_proj.weight"]
    reference_model.attn.out_proj.bias.data = weights["out_proj.bias"]

    reference_model.eval()

    # Prepare parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
        custom_preprocessor=_custom_preprocessor,
    )

    # Create test inputs (self-attention: query=key=value)
    query = torch.randn(num_queries, batch_size, embed_dims)
    query_pos = torch.randn(num_queries, batch_size, embed_dims)
    identity = torch.randn(num_queries, batch_size, embed_dims)

    # Run reference model
    with torch.no_grad():
        reference_output = reference_model(
            query=query,
            key=None,  # Will default to query
            value=None,  # Will default to key
            identity=identity,
            query_pos=query_pos,
            key_pos=None,  # Will default to query_pos
            attn_mask=None,
            key_padding_mask=None,
        )

    # Create TTNN model
    ttnn_model = TtMultiheadAttention(
        params=parameters,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=False,
    )

    # Convert inputs to TTNN format
    ttnn_query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_identity = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN model
    ttnn_output = ttnn_model(
        query=ttnn_query,
        key=None,
        value=None,
        identity=ttnn_identity,
        query_pos=ttnn_query_pos,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
    )

    # Convert TTNN output back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(reference_output, ttnn_output_torch, pcc=0.99)

    assert pcc_passed, logger.error(f"PCC check failed for GroupMultiheadAttention self-attention: {pcc_message}")
    logger.info(f"GroupMultiheadAttention self-attention PCC passed: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_groupmultiheadattention_cross_attention(device, reset_seeds):
    """Test cross-attention case (query != key) using real weights."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load weights from checkpoint
    weights, embed_dims, num_heads = _load_weights_from_checkpoint(layer_idx=0)

    num_queries = 100
    num_keys = 200
    batch_size = 1

    # Create reference model
    reference_model = GroupMultiheadAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        batch_first=False,
    )

    # Load weights into the reference model
    reference_model.attn.in_proj_weight.data = weights["in_proj_weight"]
    reference_model.attn.in_proj_bias.data = weights["in_proj_bias"]
    reference_model.attn.out_proj.weight.data = weights["out_proj.weight"]
    reference_model.attn.out_proj.bias.data = weights["out_proj.bias"]

    reference_model.eval()

    # Prepare parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
        custom_preprocessor=_custom_preprocessor,
    )

    # Create test inputs (cross-attention: query != key)
    query = torch.randn(num_queries, batch_size, embed_dims)
    key = torch.randn(num_keys, batch_size, embed_dims)
    value = torch.randn(num_keys, batch_size, embed_dims)
    query_pos = torch.randn(num_queries, batch_size, embed_dims)
    key_pos = torch.randn(num_keys, batch_size, embed_dims)
    identity = torch.randn(num_queries, batch_size, embed_dims)

    # Run reference model
    with torch.no_grad():
        reference_output = reference_model(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=None,
            key_padding_mask=None,
        )

    # Create TTNN model
    ttnn_model = TtMultiheadAttention(
        params=parameters,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=False,
    )

    # Convert inputs to TTNN format
    ttnn_query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_value = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_key_pos = ttnn.from_torch(key_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_identity = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN model
    ttnn_output = ttnn_model(
        query=ttnn_query,
        key=ttnn_key,
        value=ttnn_value,
        identity=ttnn_identity,
        query_pos=ttnn_query_pos,
        key_pos=ttnn_key_pos,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
    )

    # Convert TTNN output back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(reference_output, ttnn_output_torch, pcc=0.99)

    assert pcc_passed, logger.error(f"PCC check failed for GroupMultiheadAttention cross-attention: {pcc_message}")
    logger.info(f"GroupMultiheadAttention cross-attention PCC passed: {pcc_message}")
