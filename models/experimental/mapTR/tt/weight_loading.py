# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading utilities for MapTR models.

This module provides functions to load MapTR checkpoint weights and
extract parameters for the TTNN implementation.

Updated to follow VADv2's weight loading approach for better PCC.
"""

import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


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


# ============================================================================
# VADv2-style weight loading functions for better PCC
# ============================================================================


def extract_sequential_branch(module_list: nn.ModuleList, dtype: ttnn.DataType = ttnn.bfloat16) -> Dict[str, Any]:
    """Extract parameters from a sequential branch (cls/reg branches).

    This follows VADv2's approach using preprocess_linear_weight/bias for
    consistent weight preprocessing that yields better PCC.

    Args:
        module_list: ModuleList containing Sequential branches.
        dtype: Target TTNN data type.

    Returns:
        Dictionary with preprocessed parameters.
    """
    branch_params = {}
    for i, mod in enumerate(module_list):
        layer_params = {}
        layer_index = 0

        if isinstance(mod, nn.Sequential):
            layers = mod
        else:
            layers = [mod]

        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer_params[str(layer_index)] = {
                    "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                    "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
                }
                layer_index += 1
            elif isinstance(layer, nn.LayerNorm):
                layer_params[f"{layer_index}_norm"] = {
                    "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                    "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                }
                layer_index += 1

        branch_params[str(i)] = layer_params
    return branch_params


def custom_maptr_preprocessor(model: nn.Module, name: str) -> Dict[str, Any]:
    """Custom preprocessor for MapTR head parameters.

    This follows VADv2's approach for better PCC when loading weights.

    Args:
        model: PyTorch MapTR head module.
        name: Name of the module (unused, required by API).

    Returns:
        Dictionary with preprocessed parameters.
    """
    from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead

    parameters = {}

    if isinstance(model, MapTRHead):
        parameters["head"] = {}
        parameters["head"]["branches"] = {}

        # Extract classification branches
        if hasattr(model, "cls_branches"):
            parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
                model.cls_branches, dtype=ttnn.bfloat16
            )

        # Extract regression branches
        if hasattr(model, "reg_branches"):
            parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
                model.reg_branches, dtype=ttnn.bfloat16
            )

        # Extract query embeddings (instance_pts type)
        if hasattr(model, "instance_embedding") and model.instance_embedding is not None:
            parameters["head"]["instance_embedding"] = {
                "weight": ttnn.from_torch(model.instance_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        if hasattr(model, "pts_embedding") and model.pts_embedding is not None:
            parameters["head"]["pts_embedding"] = {
                "weight": ttnn.from_torch(model.pts_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        # Extract query embedding (all_pts type)
        if hasattr(model, "query_embedding") and model.query_embedding is not None:
            parameters["head"]["query_embedding"] = {
                "weight": ttnn.from_torch(model.query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        # Extract positional encoding
        if hasattr(model, "positional_encoding"):
            parameters["head"]["positional_encoding"] = {
                "row_embed": {
                    "weight": ttnn.from_torch(
                        model.positional_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                },
                "col_embed": {
                    "weight": ttnn.from_torch(
                        model.positional_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                },
            }

        # Extract BEV embedding
        if hasattr(model, "bev_embedding") and model.bev_embedding is not None:
            parameters["head"]["bev_embedding"] = {
                "weight": ttnn.from_torch(model.bev_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        # Extract transformer parameters
        if hasattr(model, "transformer") and model.transformer is not None:
            parameters["head"]["transformer"] = {}

            # Extract CAN bus MLP if available
            if hasattr(model.transformer, "can_bus_mlp"):
                parameters["head"]["transformer"]["can_bus_mlp"] = {
                    "0": {
                        "weight": preprocess_linear_weight(
                            model.transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                    },
                    "1": {
                        "weight": preprocess_linear_weight(
                            model.transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                    },
                }

            # Extract reference points if available
            if hasattr(model.transformer, "reference_points"):
                parameters["head"]["transformer"]["reference_points"] = {
                    "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
                }

    return parameters


def create_maptr_head_parameters(model: nn.Module, device: ttnn.Device = None) -> Any:
    """Create TTNN parameters for MapTR head using VADv2's approach.

    This uses preprocess_model_parameters with custom preprocessor for
    consistent weight loading that achieves better PCC.

    Args:
        model: PyTorch MapTR head module.
        device: TTNN device for tensor placement.

    Returns:
        Preprocessed parameters object.
    """
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_maptr_preprocessor,
        device=device,
    )
    return parameters


def load_maptr_weights_into_head(
    torch_model: nn.Module, weights_path: str, head_prefix: str = "pts_bbox_head."
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load MapTR weights into a head model with detailed tracking.

    This follows VADv2's approach for weight loading with shape validation
    and detailed statistics tracking.

    Args:
        torch_model: PyTorch head model to load weights into.
        weights_path: Path to MapTR checkpoint.
        head_prefix: Prefix for head weights in checkpoint.

    Returns:
        Tuple of (model with loaded weights, loading statistics).
    """
    import os

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        full_state_dict = checkpoint["model"]
    else:
        full_state_dict = checkpoint

    # Extract head weights
    head_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(head_prefix):
            relative_key = key[len(head_prefix) :]
            head_weights[relative_key] = value

    logger.info(f"Loaded {len(head_weights)} weight tensors for MapTR head")

    # Track statistics
    stats = {
        "loaded_count": 0,
        "missing_count": 0,
        "total_count": len(torch_model.state_dict()),
        "critical_weights_loaded": True,
        "loaded_keys": [],
        "missing_critical_keys": [],
        "skipped_keys": [],
        "maptr_weights_used": set(),
    }

    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    # Critical weight prefixes
    critical_prefixes = ["cls_branches", "reg_branches"]

    for model_key in model_state_dict.keys():
        loaded = False

        # Direct match
        if model_key in head_weights:
            maptr_weight = head_weights[model_key]
            model_weight = model_state_dict[model_key]

            if maptr_weight.shape == model_weight.shape:
                new_state_dict[model_key] = maptr_weight
                stats["loaded_count"] += 1
                stats["loaded_keys"].append(model_key)
                stats["maptr_weights_used"].add(model_key)
                loaded = True
            else:
                logger.warning(
                    f"Shape mismatch for {model_key}: "
                    f"MapTR shape {maptr_weight.shape} vs model shape {model_weight.shape}"
                )
                new_state_dict[model_key] = model_weight
        else:
            new_state_dict[model_key] = model_state_dict[model_key]

        # Check if critical weight is missing
        if not loaded:
            is_critical = any(model_key.startswith(prefix) for prefix in critical_prefixes)
            if is_critical and model_key not in stats["loaded_keys"]:
                stats["missing_count"] += 1
                stats["missing_critical_keys"].append(model_key)
                stats["critical_weights_loaded"] = False

    # Find unused MapTR weights
    all_maptr_keys = set(head_weights.keys())
    unused_maptr_keys = all_maptr_keys - stats["maptr_weights_used"]

    # Filter expected unused keys
    expected_unused_prefixes = [
        "transformer.decoder.layers.",
        "transformer.encoder.layers.",
        "transformer.level_embeds",
        "transformer.cams_embeds",
        "code_weights",
    ]

    for unused_key in unused_maptr_keys:
        is_expected_unused = any(unused_key.startswith(prefix) for prefix in expected_unused_prefixes)
        if not is_expected_unused:
            stats["skipped_keys"].append(unused_key)

    # Load weights
    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    # Log statistics
    logger.info(f"Weight loading statistics:")
    logger.info(f"  - Total model parameters: {stats['total_count']}")
    logger.info(
        f"  - Successfully loaded from MapTR: {stats['loaded_count']} ({100*stats['loaded_count']/stats['total_count']:.1f}%)"
    )
    logger.info(f"  - Missing critical weights: {stats['missing_count']}")
    logger.info(f"  - Critical weights loaded: {stats['critical_weights_loaded']}")
    logger.info(f"  - MapTR weights in checkpoint: {len(head_weights)}")
    logger.info(f"  - MapTR weights used: {len(stats['maptr_weights_used'])}")

    if stats["missing_critical_keys"]:
        logger.error(f"  - Missing critical keys: {stats['missing_critical_keys'][:10]}...")

    if stats["skipped_keys"]:
        logger.info(f"  - Skipped MapTR keys: {stats['skipped_keys'][:10]}...")

    return torch_model, stats


# ============================================================================
# Complete model preprocessing (like VADv2's create_vadv2_model_parameters_vad)
# ============================================================================


def custom_maptr_full_preprocessor(model: nn.Module, name: str) -> Dict[str, Any]:
    """Custom preprocessor for complete MapTR model parameters.

    This follows VADv2's approach for complete model weight loading.
    Handles backbone, neck, and head parameters.

    Args:
        model: PyTorch MapTR model.
        name: Name of the module (unused, required by API).

    Returns:
        Dictionary with preprocessed parameters.
    """
    from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
    from models.experimental.mapTR.reference.pytorch_maptr import MapTR
    from models.experimental.mapTR.reference.pytorch_resnet import ResNet
    from models.experimental.mapTR.reference.pytorch_fpn import FPN
    from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead

    parameters = {}

    if isinstance(model, MapTR):
        # Process backbone (ResNet50)
        if hasattr(model, "img_backbone") and isinstance(model.img_backbone, ResNet):
            backbone = model.img_backbone
            parameters["img_backbone"] = {}

            # Initial conv + bn
            weight, bias = fold_batch_norm2d_into_conv2d(backbone.conv1, backbone.bn1)
            parameters["img_backbone"]["conv1"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
                "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
            }

            # Loop over all layers (layer1 to layer4)
            for layer_idx in range(1, 5):
                layer = getattr(backbone, f"layer{layer_idx}")
                for block_idx, block in enumerate(layer):
                    prefix = f"layer{layer_idx}_{block_idx}"
                    parameters["img_backbone"][prefix] = {}

                    # conv1, conv2, conv3
                    for conv_name in ["conv1", "conv2", "conv3"]:
                        conv = getattr(block, conv_name)
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["img_backbone"][prefix][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                    # downsample (if present)
                    if hasattr(block, "downsample") and block.downsample is not None:
                        ds = block.downsample
                        if isinstance(ds, nn.Sequential):
                            conv = ds[0]
                            bn = ds[1]
                            w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                            parameters["img_backbone"][prefix]["downsample"] = {
                                "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                            }

        # Process neck (FPN)
        if hasattr(model, "img_neck") and isinstance(model.img_neck, FPN):
            neck = model.img_neck
            parameters["img_neck"] = {}

            # Lateral convs
            if hasattr(neck, "lateral_convs") and len(neck.lateral_convs) > 0:
                lateral_conv = (
                    neck.lateral_convs[0] if isinstance(neck.lateral_convs, nn.ModuleList) else neck.lateral_convs
                )
                conv = lateral_conv.conv if hasattr(lateral_conv, "conv") else lateral_conv
                parameters["img_neck"]["lateral_convs"] = {
                    "conv": {
                        "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                        if conv.bias is not None
                        else None,
                    }
                }

            # FPN convs
            if hasattr(neck, "fpn_convs") and len(neck.fpn_convs) > 0:
                fpn_conv = neck.fpn_convs[0] if isinstance(neck.fpn_convs, nn.ModuleList) else neck.fpn_convs
                conv = fpn_conv.conv if hasattr(fpn_conv, "conv") else fpn_conv
                parameters["img_neck"]["fpn_convs"] = {
                    "conv": {
                        "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                        if conv.bias is not None
                        else None,
                    }
                }

        # Process head (MapTRHead)
        if hasattr(model, "pts_bbox_head") and isinstance(model.pts_bbox_head, MapTRHead):
            head = model.pts_bbox_head
            parameters["head"] = {}
            parameters["head"]["branches"] = {}

            # Extract classification branches
            if hasattr(head, "cls_branches"):
                parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
                    head.cls_branches, dtype=ttnn.bfloat16
                )

            # Extract regression branches
            if hasattr(head, "reg_branches"):
                parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
                    head.reg_branches, dtype=ttnn.bfloat16
                )

            # Extract query embeddings
            if hasattr(head, "instance_embedding") and head.instance_embedding is not None:
                parameters["head"]["instance_embedding"] = {
                    "weight": ttnn.from_torch(
                        head.instance_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                }

            if hasattr(head, "pts_embedding") and head.pts_embedding is not None:
                parameters["head"]["pts_embedding"] = {
                    "weight": ttnn.from_torch(head.pts_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                }

            if hasattr(head, "query_embedding") and head.query_embedding is not None:
                parameters["head"]["query_embedding"] = {
                    "weight": ttnn.from_torch(head.query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                }

            # Extract positional encoding
            if hasattr(head, "positional_encoding"):
                parameters["head"]["positional_encoding"] = {
                    "row_embed": {
                        "weight": ttnn.from_torch(
                            head.positional_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                        )
                    },
                    "col_embed": {
                        "weight": ttnn.from_torch(
                            head.positional_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                        )
                    },
                }

            # Extract BEV embedding
            if hasattr(head, "bev_embedding") and head.bev_embedding is not None:
                parameters["head"]["bev_embedding"] = {
                    "weight": ttnn.from_torch(head.bev_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                }

            # Extract transformer parameters
            if hasattr(head, "transformer") and head.transformer is not None:
                parameters["head"]["transformer"] = {}

                # CAN bus MLP
                if hasattr(head.transformer, "can_bus_mlp"):
                    parameters["head"]["transformer"]["can_bus_mlp"] = {
                        "0": {
                            "weight": preprocess_linear_weight(
                                head.transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(head.transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                        },
                        "1": {
                            "weight": preprocess_linear_weight(
                                head.transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(head.transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                        },
                    }
                    # LayerNorm if present
                    if hasattr(head.transformer.can_bus_mlp, "norm"):
                        parameters["head"]["transformer"]["can_bus_mlp"]["norm"] = {
                            "weight": preprocess_layernorm_parameter(
                                head.transformer.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_layernorm_parameter(
                                head.transformer.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16
                            ),
                        }

                # Reference points
                if hasattr(head.transformer, "reference_points"):
                    parameters["head"]["transformer"]["reference_points"] = {
                        "weight": preprocess_linear_weight(
                            head.transformer.reference_points.weight, dtype=ttnn.bfloat16
                        ),
                        "bias": preprocess_linear_bias(head.transformer.reference_points.bias, dtype=ttnn.bfloat16),
                    }

                # Level and camera embeddings
                if hasattr(head.transformer, "level_embeds"):
                    parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                        head.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                if hasattr(head.transformer, "cams_embeds"):
                    parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                        head.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )

    return parameters


def create_maptr_model_parameters(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: ttnn.Device = None,
) -> Any:
    """Create complete TTNN parameters for MapTR model (like VADv2's approach).

    This is the main function for complete model preprocessing, handling
    backbone, neck, and head parameters with inference of conv arguments.

    Args:
        model: PyTorch MapTR model with loaded weights.
        input_tensor: Sample input tensor for conv arg inference.
        device: TTNN device for tensor placement.

    Returns:
        Parameters object with all components.
    """
    from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_maptr_full_preprocessor,
        device=device,
    )

    # Infer conv arguments for backbone and neck
    parameters.conv_args = {"img_backbone": {}, "img_neck": {}}

    img = input_tensor
    if isinstance(img, list):
        img = img[0]
    if img.dim() == 5 and img.size(0) == 1:
        img = img.squeeze(0)
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)

    # Infer backbone conv args
    parameters.conv_args["img_backbone"] = infer_ttnn_module_args(
        model=model.img_backbone,
        run_model=lambda m: m(img),
        device=None,
    )

    # Run backbone to get features for neck
    with torch.no_grad():
        img_feats = model.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

    # Infer neck conv args
    parameters.conv_args["img_neck"] = infer_ttnn_module_args(
        model=model.img_neck,
        run_model=lambda m: m(img_feats),
        device=None,
    )

    # Link module references to conv_args
    for key in parameters.conv_args.keys():
        if key == "img_backbone":
            for conv_key in parameters.conv_args[key].keys():
                conv_key_str = str(conv_key) if not isinstance(conv_key, str) else conv_key
                if hasattr(model.img_backbone, conv_key_str):
                    parameters.conv_args[key][conv_key].module = getattr(model.img_backbone, conv_key_str)
        elif key == "img_neck":
            for conv_key in parameters.conv_args[key].keys():
                conv_key_str = str(conv_key) if not isinstance(conv_key, str) else conv_key
                if hasattr(model.img_neck, conv_key_str):
                    parameters.conv_args[key][conv_key].module = getattr(model.img_neck, conv_key_str)

    return parameters


def load_maptr_torch_model(checkpoint_path: str = None) -> nn.Module:
    """Load complete MapTR PyTorch model with weights.

    Args:
        checkpoint_path: Path to MapTR checkpoint file.

    Returns:
        PyTorch MapTR model with loaded weights.
    """
    import os
    from models.experimental.mapTR.reference.pytorch_maptr import MapTR
    from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
    from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
    from models.experimental.mapTR.reference.pytorch_fpn import FPN
    from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
    from models.experimental.mapTR.reference.modules.transformer import MapTRPerceptionTransformer
    from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
    from models.experimental.mapTR.reference.modules.decoder import MapTRDecoder, BaseTransformerLayer
    from models.experimental.mapTR.reference.nms_free_coder import MapTRNMSFreeCoder

    # Default path
    if checkpoint_path is None:
        checkpoint_path = (
            "/home/ubuntu/christyv1/tt-metal/models/experimental/mapTR/resources/maptr_tiny_r50_24e_bevformer.pth"
        )

    # Build ResNet50 backbone
    backbone = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        out_indices=(3,),
    )

    # Build FPN neck
    fpn = FPN(
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        num_outs=1,
        relu_before_extra_convs=False,
    )

    # Build positional encoding
    positional_encoding = LearnedPositionalEncoding(
        num_feats=128,
        row_num_embed=200,
        col_num_embed=100,
    )

    # Build BEVFormer encoder
    encoder = BEVFormerEncoder(
        num_layers=6,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        embed_dims=256,
        num_heads=8,
        feedforward_channels=512,
        ffn_dropout=0.1,
    )

    # Build decoder layers
    decoder_layers = nn.ModuleList(
        [
            BaseTransformerLayer(
                attn_cfgs=[
                    dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                    dict(type="CustomMSDeformableAttention", embed_dims=256, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            )
            for _ in range(6)
        ]
    )

    decoder = MapTRDecoder(
        layers=decoder_layers,
        return_intermediate=True,
    )

    # Build transformer
    transformer = MapTRPerceptionTransformer(
        encoder=encoder,
        decoder=decoder,
        embed_dims=256,
        num_feature_levels=4,
        num_cams=6,
    )

    # Build bbox coder
    bbox_coder = MapTRNMSFreeCoder(
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        post_center_range=[-20.0, -35.0, -20.0, 35.0],
        max_num=50,
        num_classes=3,
    )

    # Build head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=positional_encoding,
        bbox_coder=bbox_coder,
        embed_dims=256,
        num_classes=3,
        num_reg_fcs=2,
        num_cls_fcs=2,
        code_size=2,
        bev_h=200,
        bev_w=100,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_vec=50,
        num_pts_per_vec=20,
    )

    # Build full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = load_maptr_checkpoint(checkpoint_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        logger.info("Loaded checkpoint weights (non-strict)")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")

    model.eval()
    return model
