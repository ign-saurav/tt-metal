# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.dependency import MultiheadAttention
from models.experimental.MapTR.tt.mha import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefix for MultiheadAttention in decoder layer 0
# MapTR uses: pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn
# attentions.0 = MultiheadAttention (self-attention)
# attentions.1 = CustomMSDeformableAttention (cross-attention)
MHA_LAYER = "pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn."


def load_maptr_mha_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}. " "Please download the weights first.")

    # Load full checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        full_state_dict = checkpoint["model"]
    else:
        full_state_dict = checkpoint

    # Extract only MultiheadAttention weights
    mha_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(MHA_LAYER):
            # Remove the layer prefix to get the relative key
            relative_key = key[len(MHA_LAYER) :]
            mha_weights[relative_key] = value

    logger.info(f"Loaded {len(mha_weights)} weight tensors for MultiheadAttention")
    logger.info(f"Weight keys: {list(mha_weights.keys())}")

    return mha_weights


def load_torch_model_maptr(torch_model: MultiheadAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    mha_weights = load_maptr_mha_weights(weights_path)

    # Map the checkpoint keys to model keys
    # PyTorch MultiheadAttention uses attn.in_proj_weight, attn.in_proj_bias, attn.out_proj.weight, attn.out_proj.bias
    # Our model wraps it, so we need to map to attn.*
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        # Map model keys to checkpoint keys
        if model_key == "attn.in_proj_weight" and "in_proj_weight" in mha_weights:
            new_state_dict[model_key] = mha_weights["in_proj_weight"]
        elif model_key == "attn.in_proj_bias" and "in_proj_bias" in mha_weights:
            new_state_dict[model_key] = mha_weights["in_proj_bias"]
        elif model_key == "attn.out_proj.weight" and "out_proj.weight" in mha_weights:
            new_state_dict[model_key] = mha_weights["out_proj.weight"]
        elif model_key == "attn.out_proj.bias" and "out_proj.bias" in mha_weights:
            new_state_dict[model_key] = mha_weights["out_proj.bias"]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, MultiheadAttention):
        parameters["multihead_attention"] = {}
        parameters["multihead_attention"]["in_proj"] = {}
        parameters["multihead_attention"]["in_proj"]["weight"] = preprocess_linear_weight(
            model.attn.in_proj_weight, dtype=ttnn.bfloat16
        )
        if model.attn.in_proj_bias is not None:
            parameters["multihead_attention"]["in_proj"]["bias"] = preprocess_linear_bias(
                model.attn.in_proj_bias, dtype=ttnn.bfloat16
            )
        else:
            parameters["multihead_attention"]["in_proj"]["bias"] = None
        parameters["multihead_attention"]["out_proj"] = {}
        parameters["multihead_attention"]["out_proj"]["weight"] = preprocess_linear_weight(
            model.attn.out_proj.weight, dtype=ttnn.bfloat16
        )
        if model.attn.out_proj.bias is not None:
            parameters["multihead_attention"]["out_proj"]["bias"] = preprocess_linear_bias(
                model.attn.out_proj.bias, dtype=ttnn.bfloat16
            )
        else:
            parameters["multihead_attention"]["out_proj"]["bias"] = None

    return parameters


def create_maptr_model_parameters_mha(model: MultiheadAttention, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_multihead_attention(
    device,
    reset_seeds,
):
    # MapTR config parameters (matching maptr_tiny_r50_24e_bevformer.py)
    embed_dims = 256
    num_heads = 8
    batch_first = False
    dropout = 0.1  # Matching config

    # Create PyTorch model
    torch_model = MultiheadAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        dropout=dropout,  # This sets attn_drop via kwargs in reference
        batch_first=batch_first,
    )

    # Load MapTR weights
    torch_model = load_torch_model_maptr(torch_model)

    # Create test inputs
    batch_size = 1
    num_query = 200  # Number of query tokens
    num_key = 200  # Number of key/value tokens (same as query for self-attention)

    # Inputs: (num_query, bs, embed_dims) when batch_first=False
    query = torch.randn(num_query, batch_size, embed_dims)
    key = torch.randn(num_key, batch_size, embed_dims)
    value = torch.randn(num_key, batch_size, embed_dims)
    identity = query.clone()
    query_pos = torch.randn(num_query, batch_size, embed_dims)
    key_pos = torch.randn(num_key, batch_size, embed_dims)

    # Run PyTorch model
    torch_output = torch_model(
        query=query,
        key=key,
        value=value,
        identity=identity,
        query_pos=query_pos,
        key_pos=key_pos,
    )

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Prepare TT model parameters
    parameter = create_maptr_model_parameters_mha(torch_model, device=device)

    # Create TT model
    tt_model = TtMultiheadAttention(
        params=parameter.multihead_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=batch_first,
    )

    # Convert inputs to TT tensors
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    identity_tt = ttnn.from_torch(identity, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    key_pos_tt = ttnn.from_torch(key_pos, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    tt_output = tt_model(
        query=query_tt,
        key=key_tt,
        value=value_tt,
        identity=identity_tt,
        query_pos=query_pos_tt,
        key_pos=key_pos_tt,
    )

    # Compare outputs
    ttnn_output = ttnn.to_torch(tt_output)
    logger.info(f"TTNN output shape: {ttnn_output.shape}")
    logger.info(
        f"TTNN output stats: min={ttnn_output.min():.4f}, max={ttnn_output.max():.4f}, mean={ttnn_output.mean():.4f}"
    )

    # Verify output shapes match
    assert (
        torch_output.shape == ttnn_output.shape
    ), f"Output shapes don't match: {torch_output.shape} vs {ttnn_output.shape}"

    # Compare with PCC (expect high correlation > 0.99)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output.float(), torch_output.float(), 0.99)
    logger.info(f"PCC Result: {pcc_message}")

    assert pcc_passed, f"PCC check failed: {pcc_message}"
