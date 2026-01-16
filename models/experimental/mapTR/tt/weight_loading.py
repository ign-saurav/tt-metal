# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading utilities for MapTR models.

This module provides functions to load MapTR checkpoint weights and
extract parameters for the TTNN implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from loguru import logger


def load_maptr_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load MapTR checkpoint and return state dict.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        State dict with model weights.
    """
    logger.info(f"Loading MapTR checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    logger.info(f"Loaded checkpoint with {len(state_dict)} keys")
    return state_dict


def extract_head_params(state_dict: Dict[str, torch.Tensor], prefix: str = "pts_bbox_head.") -> Dict[str, torch.Tensor]:
    """Extract head parameters from checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix for head parameters.

    Returns:
        Dictionary of head parameters.
    """
    head_params = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            head_params[new_key] = value

    logger.info(f"Extracted {len(head_params)} head parameters")
    return head_params


def extract_backbone_params(
    state_dict: Dict[str, torch.Tensor], prefix: str = "img_backbone."
) -> Dict[str, torch.Tensor]:
    """Extract backbone parameters from checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix for backbone parameters.

    Returns:
        Dictionary of backbone parameters.
    """
    backbone_params = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            backbone_params[new_key] = value

    logger.info(f"Extracted {len(backbone_params)} backbone parameters")
    return backbone_params


def extract_fpn_params(state_dict: Dict[str, torch.Tensor], prefix: str = "img_neck.") -> Dict[str, torch.Tensor]:
    """Extract FPN parameters from checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix for FPN parameters.

    Returns:
        Dictionary of FPN parameters.
    """
    fpn_params = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            fpn_params[new_key] = value

    logger.info(f"Extracted {len(fpn_params)} FPN parameters")
    return fpn_params


def extract_transformer_params(
    state_dict: Dict[str, torch.Tensor], prefix: str = "pts_bbox_head.transformer."
) -> Dict[str, torch.Tensor]:
    """Extract transformer parameters from checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix for transformer parameters.

    Returns:
        Dictionary of transformer parameters.
    """
    transformer_params = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            transformer_params[new_key] = value

    logger.info(f"Extracted {len(transformer_params)} transformer parameters")
    return transformer_params


def load_pytorch_model_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load weights into a PyTorch model.

    Args:
        model: PyTorch model to load weights into.
        state_dict: State dict with weights.
        strict: Whether to strictly match keys.

    Returns:
        Tuple of (missing_keys, unexpected_keys).
    """
    result = model.load_state_dict(state_dict, strict=strict)
    missing_keys = result.missing_keys if hasattr(result, "missing_keys") else []
    unexpected_keys = result.unexpected_keys if hasattr(result, "unexpected_keys") else []

    if missing_keys:
        logger.warning(
            f"Missing keys: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}"
        )
    if unexpected_keys:
        logger.warning(
            f"Unexpected keys: {unexpected_keys[:10]}..."
            if len(unexpected_keys) > 10
            else f"Unexpected keys: {unexpected_keys}"
        )

    return missing_keys, unexpected_keys


def check_weight_loading_pcc(
    loaded_weights: Dict[str, torch.Tensor],
    torch_weights: Dict[str, torch.Tensor],
    component_name: str = "model",
    fail_on_missing: bool = False,
    fail_on_skipped: bool = False,
    allowed_skipped_keys: Optional[List[str]] = None,
) -> float:
    """Check PCC between loaded weights and target weights.

    Args:
        loaded_weights: Weights that were loaded into the model.
        torch_weights: Target weights to compare against.
        component_name: Name of the component for logging.
        fail_on_missing: Raise error if model has weights not in checkpoint.
        fail_on_skipped: Raise error if checkpoint has weights not loaded.
        allowed_skipped_keys: Keys that are allowed to be skipped.

    Returns:
        Average PCC across all weights.
    """
    if allowed_skipped_keys is None:
        allowed_skipped_keys = []

    pcc_values = []
    missing_keys = []
    skipped_keys = []

    # Check model keys against loaded weights
    for key in torch_weights.keys():
        if key in loaded_weights:
            w1 = torch_weights[key].float().flatten()
            w2 = loaded_weights[key].float().flatten()

            if w1.shape == w2.shape:
                if w1.numel() > 0:
                    pcc = torch.corrcoef(torch.stack([w1, w2]))[0, 1].item()
                    pcc_values.append(pcc)
                    if pcc < 0.99:
                        logger.warning(f"{component_name}: {key} PCC = {pcc:.6f}")
            else:
                logger.error(f"{component_name}: Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
        else:
            missing_keys.append(key)

    # Check for checkpoint keys not in model
    for key in loaded_weights.keys():
        if key not in torch_weights:
            if key not in allowed_skipped_keys:
                skipped_keys.append(key)

    avg_pcc = sum(pcc_values) / len(pcc_values) if pcc_values else 0.0

    logger.info(f"{component_name}: Checked {len(pcc_values)} weights, Average PCC = {avg_pcc:.6f}")

    if missing_keys:
        msg = f"{component_name}: {len(missing_keys)} model weights not in checkpoint"
        if fail_on_missing:
            raise ValueError(msg)
        logger.warning(msg)

    if skipped_keys:
        msg = f"{component_name}: {len(skipped_keys)} checkpoint weights not loaded"
        if fail_on_skipped:
            raise ValueError(msg)
        logger.warning(msg)

    return avg_pcc


def prepare_maptr_model_parameters(
    checkpoint_path: str,
    torch_model: nn.Module,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Prepare all parameters for MapTR TTNN model.

    Args:
        checkpoint_path: Path to checkpoint file.
        torch_model: PyTorch reference model.

    Returns:
        Dictionary containing all extracted parameters.
    """
    state_dict = load_maptr_checkpoint(checkpoint_path)

    # Load weights into torch model
    load_pytorch_model_weights(torch_model, state_dict, strict=False)

    # Extract component-wise parameters
    head_params = extract_head_params(state_dict)
    backbone_params = extract_backbone_params(state_dict)
    fpn_params = extract_fpn_params(state_dict)
    transformer_params = extract_transformer_params(state_dict)

    return {
        "head": head_params,
        "backbone": backbone_params,
        "fpn": fpn_params,
        "transformer": transformer_params,
        "full": state_dict,
    }


def get_head_params_from_torch_model(torch_head: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract head parameters from a PyTorch head module.

    This extracts the state dict with the proper keys for the TTNN head.

    Args:
        torch_head: PyTorch MapTRHead module.

    Returns:
        Dictionary of parameters for TTNN head initialization.
    """
    params = {}
    state_dict = torch_head.state_dict()

    for key, value in state_dict.items():
        params[key] = value

    return params
