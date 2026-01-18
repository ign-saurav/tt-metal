# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.decoder import (
    CustomMSDeformableAttention,
)
from models.experimental.MapTR.tt.custom_defrmble_attention import TtCustomMSDeformableAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefix for CustomMSDeformableAttention in decoder layer 0
# MapTR uses: pts_bbox_head.transformer.decoder.layers.0.attentions.1
# attentions.0 = MultiheadAttention (self-attention)
# attentions.1 = CustomMSDeformableAttention (cross-attention)
CUSTOM_MS_DEFORMABLE_ATTN_LAYER = "pts_bbox_head.transformer.decoder.layers.0.attentions.1."


def load_maptr_custom_ms_deformable_attention_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load and isolate CustomMSDeformableAttention weights from MapTR checkpoint.

    The weights structure for CustomMSDeformableAttention:
    - sampling_offsets.weight/bias
    - attention_weights.weight/bias
    - value_proj.weight/bias
    - output_proj.weight/bias

    Args:
        weights_path: Path to the MapTR checkpoint file.

    Returns:
        Dictionary containing only the CustomMSDeformableAttention weights.
    """
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

    # Extract only CustomMSDeformableAttention weights
    attn_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(CUSTOM_MS_DEFORMABLE_ATTN_LAYER):
            # Remove the layer prefix to get the relative key
            relative_key = key[len(CUSTOM_MS_DEFORMABLE_ATTN_LAYER) :]
            attn_weights[relative_key] = value

    logger.info(f"Loaded {len(attn_weights)} weight tensors for CustomMSDeformableAttention")
    logger.info(f"Weight keys: {list(attn_weights.keys())}")

    return attn_weights


def load_torch_model_maptr(torch_model: CustomMSDeformableAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load MapTR weights into the CustomMSDeformableAttention model.

    Args:
        torch_model: The CustomMSDeformableAttention model to load weights into.
        weights_path: Path to the MapTR checkpoint file.

    Returns:
        The model with loaded weights.
    """
    attn_weights = load_maptr_custom_ms_deformable_attention_weights(weights_path)

    # Map the checkpoint keys to model keys
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in attn_weights:
            new_state_dict[model_key] = attn_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    """Custom preprocessor for CustomMSDeformableAttention parameters."""
    parameters = {}

    if isinstance(model, CustomMSDeformableAttention):
        parameters["custom_ms_deformable_attention"] = {}
        parameters["custom_ms_deformable_attention"]["sampling_offsets"] = {}
        parameters["custom_ms_deformable_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["attention_weights"] = {}
        parameters["custom_ms_deformable_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["value_proj"] = {}
        parameters["custom_ms_deformable_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["output_proj"] = {}
        parameters["custom_ms_deformable_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )

    return parameters


def create_maptr_model_parameters_attn(model: CustomMSDeformableAttention, device=None):
    """Create TTNN parameters for CustomMSDeformableAttention model."""
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_custom_ms_deformable_attention(
    device,
    reset_seeds,
):
    """Test MapTR CustomMSDeformableAttention: compare reference vs TTNN implementation with MapTR weights."""
    # MapTR config parameters from maptr_tiny_r50_24e_bevformer.py
    # dict(type="CustomMSDeformableAttention", embed_dims=_dim_, num_levels=1)
    embed_dims = 256
    num_heads = 8
    num_levels = 1  # From config
    num_points = 4  # Default value
    batch_first = False
    dropout = 0.1

    # Create PyTorch model
    torch_model = CustomMSDeformableAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
        dropout=dropout,
    )

    # Load MapTR weights
    torch_model = load_torch_model_maptr(torch_model)

    # Create test inputs
    batch_size = 1
    num_query = 200  # Number of query tokens
    spatial_h = 100
    spatial_w = 100
    num_value = spatial_h * spatial_w  # Total spatial features

    # Inputs: (num_query, bs, embed_dims) when batch_first=False
    query = torch.randn(num_query, batch_size, embed_dims)
    value = torch.randn(num_value, batch_size, embed_dims)
    identity = query.clone()
    query_pos = torch.randn(num_query, batch_size, embed_dims)

    # Reference points: (bs, num_query, num_levels, 2) - normalized coordinates [0, 1]
    reference_points = torch.rand(batch_size, num_query, num_levels, 2)

    # Spatial shapes: (num_levels, 2) - [height, width] for each level
    spatial_shapes = torch.tensor([[spatial_h, spatial_w]], dtype=torch.long)

    # Level start index: (num_levels,) - starting index for each level
    level_start_index = torch.tensor([0], dtype=torch.long)

    # Run PyTorch model
    torch_output = torch_model(
        query=query,
        value=value,
        identity=identity,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Prepare TT model parameters
    parameter = create_maptr_model_parameters_attn(torch_model, device=device)

    # Create TT model
    tt_model = TtCustomMSDeformableAttention(
        params=parameter.custom_ms_deformable_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
        dropout=dropout,
    )

    # Convert inputs to TT tensors
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    identity_tt = ttnn.from_torch(identity, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    tt_output = tt_model(
        query=query_tt,
        value=value_tt,
        identity=identity_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
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
