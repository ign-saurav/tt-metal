# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.tt.tt_positional_encoding import TtLearnedPositionalEncoding
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
)


MAPTR_WEIGHTS_PATH = "models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefix for positional encoding in MapTR
# MapTR uses: pts_bbox_head.positional_encoding
POSITIONAL_ENCODING_LAYER = "pts_bbox_head.positional_encoding."


def load_maptr_positional_encoding_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load and isolate LearnedPositionalEncoding weights from MapTR checkpoint.

    The weights structure for LearnedPositionalEncoding:
    - row_embed.weight
    - col_embed.weight

    Args:
        weights_path: Path to the MapTR checkpoint file.

    Returns:
        Dictionary containing only the LearnedPositionalEncoding weights.
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

    # Extract only LearnedPositionalEncoding weights
    pe_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(POSITIONAL_ENCODING_LAYER):
            # Remove the layer prefix to get the relative key
            relative_key = key[len(POSITIONAL_ENCODING_LAYER) :]
            pe_weights[relative_key] = value

    logger.info(f"Loaded {len(pe_weights)} weight tensors for LearnedPositionalEncoding")
    logger.info(f"Weight keys: {list(pe_weights.keys())}")

    return pe_weights


def load_torch_model_maptr(torch_model: LearnedPositionalEncoding, weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load MapTR weights into the LearnedPositionalEncoding model.

    Args:
        torch_model: The LearnedPositionalEncoding model to load weights into.
        weights_path: Path to the MapTR checkpoint file.

    Returns:
        The model with loaded weights.
    """
    pe_weights = load_maptr_positional_encoding_weights(weights_path)

    # Map the checkpoint keys to model keys
    # Model keys: row_embed.weight, col_embed.weight
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        # Map model keys to checkpoint keys
        if model_key in pe_weights:
            # Check if shapes match
            if pe_weights[model_key].shape == model_state_dict[model_key].shape:
                new_state_dict[model_key] = pe_weights[model_key]
            else:
                logger.warning(
                    f"Shape mismatch for {model_key}: "
                    f"checkpoint {pe_weights[model_key].shape} vs model {model_state_dict[model_key].shape}"
                )
                new_state_dict[model_key] = model_state_dict[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    return torch_model


def infer_embedding_sizes_from_checkpoint(weights_path: str = MAPTR_WEIGHTS_PATH):
    """Infer row_num_embed and col_num_embed from checkpoint weights.

    Args:
        weights_path: Path to the MapTR checkpoint file.

    Returns:
        Tuple of (row_num_embed, col_num_embed, num_feats)
    """
    pe_weights = load_maptr_positional_encoding_weights(weights_path)

    if "row_embed.weight" in pe_weights and "col_embed.weight" in pe_weights:
        row_num_embed, num_feats = pe_weights["row_embed.weight"].shape
        col_num_embed, num_feats_col = pe_weights["col_embed.weight"].shape
        assert num_feats == num_feats_col, f"Feature dimension mismatch: row {num_feats} vs col {num_feats_col}"
        logger.info(
            f"Inferred from checkpoint: row_num_embed={row_num_embed}, col_num_embed={col_num_embed}, num_feats={num_feats}"
        )
        return row_num_embed, col_num_embed, num_feats
    else:
        # Default values if weights not found
        logger.warning("Could not infer embedding sizes from checkpoint, using defaults")
        return 200, 100, 128


def custom_preprocessor(model, name):
    """Custom preprocessor for LearnedPositionalEncoding parameters."""
    parameters = {}

    if isinstance(model, LearnedPositionalEncoding):
        parameters["row_embed"] = {}
        parameters["row_embed"]["weight"] = ttnn.from_torch(
            model.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["col_embed"] = {}
        parameters["col_embed"]["weight"] = ttnn.from_torch(
            model.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    return parameters


def create_maptr_model_parameters_pe(model: LearnedPositionalEncoding, input_tensor, device=None):
    """Create TTNN parameters for LearnedPositionalEncoding model."""
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input_tensor),
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_learned_positional_encoding(
    device,
    reset_seeds,
):
    """Test MapTR LearnedPositionalEncoding: compare reference vs TTNN implementation with MapTR weights."""
    # Try to infer embedding sizes from checkpoint
    try:
        row_num_embed, col_num_embed, num_feats = infer_embedding_sizes_from_checkpoint()
        logger.info(
            f"Using checkpoint values: row_num_embed={row_num_embed}, col_num_embed={col_num_embed}, num_feats={num_feats}"
        )
    except Exception as e:
        logger.warning(f"Could not infer sizes from checkpoint: {e}")
        # Default MapTR config parameters
        num_feats = 128  # Half of embed_dims (256/2)
        row_num_embed = 200  # BEV height
        col_num_embed = 100  # BEV width
        logger.info(
            f"Using default values: row_num_embed={row_num_embed}, col_num_embed={col_num_embed}, num_feats={num_feats}"
        )

    # Create PyTorch model with correct sizes
    torch_model = LearnedPositionalEncoding(
        num_feats=num_feats,
        row_num_embed=row_num_embed,
        col_num_embed=col_num_embed,
    )

    # Load MapTR weights (if available)
    try:
        torch_model = load_torch_model_maptr(torch_model)
        logger.info("Successfully loaded MapTR weights for positional encoding")
    except Exception as e:
        logger.warning(f"Could not load weights from checkpoint: {e}")
        logger.info("Proceeding with randomly initialized weights for testing")

    # Create test inputs - use sizes that match the embedding table
    # The height/width should be <= row_num_embed/col_num_embed to avoid index errors
    batch_size = 1
    height = min(200, row_num_embed)  # BEV height, but don't exceed embedding table size
    width = min(100, col_num_embed)  # BEV width, but don't exceed embedding table size

    # Input mask: (B, H, W)
    mask = torch.zeros(batch_size, height, width, dtype=torch.float32)

    # Run PyTorch model
    torch_output = torch_model(mask)

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Prepare input tensor for parameter inference
    input_tensor = mask

    # Prepare TT model parameters
    parameter = create_maptr_model_parameters_pe(torch_model, input_tensor, device=device)

    # Create TT model with same parameters as PyTorch model
    tt_model = TtLearnedPositionalEncoding(
        params=parameter,
        device=device,
        num_feats=num_feats,
        row_num_embed=row_num_embed,
        col_num_embed=col_num_embed,
    )

    logger.info(
        f"Created TT model with: row_num_embed={row_num_embed}, col_num_embed={col_num_embed}, num_feats={num_feats}"
    )

    # Convert inputs to TT tensors
    mask_tt = ttnn.from_torch(mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Run TT model
    tt_output = tt_model(mask_tt)

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
