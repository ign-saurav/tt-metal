# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Weight loading utilities for complete TTNN MapTR model.

This module provides comprehensive weight loading and parameter preprocessing
for the end-to-end TTNN MapTR implementation.
"""

import os
import torch
import torch.nn as nn
import ttnn
from typing import Dict, List, Tuple, Any
from loguru import logger
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
)
from models.tt_cnn.tt.builder import Conv2dConfiguration, AutoShardedStrategyConfiguration


# Default checkpoint path
MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefixes in checkpoint
BACKBONE_PREFIX = "img_backbone."
NECK_PREFIX = "img_neck."
HEAD_PREFIX = "pts_bbox_head."
TRANSFORMER_PREFIX = "pts_bbox_head.transformer."
ENCODER_PREFIX = "pts_bbox_head.transformer.encoder."
DECODER_PREFIX = "pts_bbox_head.transformer.decoder."


class ParamsWrapper:
    """Wrapper class to convert dict parameters to object attributes.

    Allows both attribute-style (params.weight) and dict-style (params["0"]) access.
    """

    def __init__(self, d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ParamsWrapper(v))
                else:
                    setattr(self, k, v)
        else:
            # If not a dict, just store it
            self._value = d

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __contains__(self, key):
        return hasattr(self, str(key))

    def get(self, key, default=None):
        return getattr(self, str(key), default)


class AttrDict(dict):
    """Dict that supports both dict['key'] and dict.key access."""

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                return AttrDict(value)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def load_maptr_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load MapTR checkpoint and return state dict.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        State dict with model weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading MapTR checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    logger.info(f"Loaded checkpoint with {len(state_dict)} keys")
    return state_dict


def extract_weights_by_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract weights with a specific prefix from state dict.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix to filter by.

    Returns:
        Dictionary of weights with prefix removed from keys.
    """
    extracted = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            extracted[new_key] = value
    return extracted


def preprocess_backbone_parameters(
    torch_model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    device: ttnn.Device = None,
) -> Tuple[Any, Any]:
    """Preprocess ResNet50 backbone parameters for TTNN.

    Args:
        torch_model: PyTorch ResNet50 model.
        state_dict: Backbone state dict.
        input_tensor: Sample input for conv args inference.
        device: TTNN device.

    Returns:
        Tuple of (preprocessed weights, conv args).
    """
    # Load weights into torch model
    # Map by order to handle key mismatches
    model_keys = list(torch_model.state_dict().keys())
    checkpoint_values = list(state_dict.values())

    if len(model_keys) <= len(checkpoint_values):
        new_state_dict = dict(zip(model_keys, checkpoint_values[: len(model_keys)]))
        torch_model.load_state_dict(new_state_dict)
    else:
        # Partial load
        torch_model.load_state_dict(state_dict, strict=False)

    torch_model.eval()

    # Custom preprocessor for backbone
    def backbone_preprocessor(model, name):
        parameters = {}

        # Initial conv + bn
        # Note: dependency.py uses norm1, pytorch_resnet uses bn1
        if hasattr(model, "norm1"):
            bn = model.norm1
        elif hasattr(model, "bn1"):
            bn = model.bn1
        else:
            raise ValueError("Cannot find batch norm layer")

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, bn)
        parameters["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters[prefix] = {}

                # conv1, conv2, conv3 with their batch norms
                for conv_idx in [1, 2, 3]:
                    conv_name = f"conv{conv_idx}"
                    # Handle different naming conventions
                    if hasattr(block, f"norm{conv_idx}"):
                        norm_name = f"norm{conv_idx}"
                    else:
                        norm_name = f"bn{conv_idx}"

                    conv = getattr(block, conv_name)
                    bn = getattr(block, norm_name)
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    parameters[prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, nn.Sequential) and len(ds) >= 2:
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters[prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

        return parameters

    # Preprocess parameters
    preprocessed = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=backbone_preprocessor,
        device=device,
    )

    # Infer conv args
    conv_args = infer_ttnn_module_args(
        model=torch_model,
        run_model=lambda model: model(input_tensor),
        device=None,
    )

    # Link module references
    for key in conv_args.keys():
        if hasattr(torch_model, str(key)):
            conv_args[key].module = getattr(torch_model, str(key))

    return ParamsWrapper(preprocessed), conv_args


def preprocess_fpn_parameters(
    torch_model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    input_feats: List[torch.Tensor],
    device: ttnn.Device = None,
) -> Tuple[Any, Any, Any]:
    """Preprocess FPN parameters for TTNN.

    Args:
        torch_model: PyTorch FPN model.
        state_dict: FPN state dict.
        input_feats: Sample features for conv args inference.
        device: TTNN device.

    Returns:
        Tuple of (lateral_config, fpn_config, conv_args).
    """
    # Load weights
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    # Infer conv args
    conv_args = infer_ttnn_module_args(
        model=torch_model,
        run_model=lambda model: model(input_feats),
        device=None,
    )

    # Create conv configs for FPN
    # Get lateral conv weights
    if hasattr(torch_model, "lateral_convs") and len(torch_model.lateral_convs) > 0:
        lateral_conv = torch_model.lateral_convs[0]
        if hasattr(lateral_conv, "conv"):
            conv_module = lateral_conv.conv
        else:
            conv_module = lateral_conv

        lateral_weight = conv_module.weight
        lateral_bias = conv_module.bias if conv_module.bias is not None else None

        # Find corresponding conv args
        lateral_args = None
        for key in conv_args.keys():
            if "lateral" in str(key).lower():
                lateral_args = conv_args[key]
                break

        if lateral_args is None:
            # Fallback: use first conv args
            lateral_args = list(conv_args.values())[0] if conv_args else None

        if lateral_args is not None:
            lateral_config = Conv2dConfiguration.from_model_args(
                conv2d_args=lateral_args,
                weights=ttnn.from_torch(lateral_weight, dtype=ttnn.float32),
                bias=ttnn.from_torch(lateral_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                if lateral_bias is not None
                else None,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
            )
        else:
            lateral_config = None
    else:
        lateral_config = None

    # Get FPN conv weights
    if hasattr(torch_model, "fpn_convs") and len(torch_model.fpn_convs) > 0:
        fpn_conv = torch_model.fpn_convs[0]
        if hasattr(fpn_conv, "conv"):
            conv_module = fpn_conv.conv
        else:
            conv_module = fpn_conv

        fpn_weight = conv_module.weight
        fpn_bias = conv_module.bias if conv_module.bias is not None else None

        # Find corresponding conv args
        fpn_args = None
        for key in conv_args.keys():
            if "fpn" in str(key).lower():
                fpn_args = conv_args[key]
                break

        if fpn_args is None:
            # Fallback: use second conv args if available
            args_list = list(conv_args.values())
            fpn_args = args_list[1] if len(args_list) > 1 else args_list[0] if args_list else None

        if fpn_args is not None:
            fpn_config = Conv2dConfiguration.from_model_args(
                conv2d_args=fpn_args,
                weights=ttnn.from_torch(fpn_weight, dtype=ttnn.float32),
                bias=ttnn.from_torch(fpn_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                if fpn_bias is not None
                else None,
                activation=None,
                sharding_strategy=AutoShardedStrategyConfiguration(),
            )
        else:
            fpn_config = None
    else:
        fpn_config = None

    return lateral_config, fpn_config, conv_args


def preprocess_transformer_parameters(
    torch_model: nn.Module,
    device: ttnn.Device = None,
) -> Dict[str, Any]:
    """Preprocess transformer parameters for TTNN.

    Args:
        torch_model: PyTorch MapTRPerceptionTransformer model.
        device: TTNN device.

    Returns:
        Dictionary with preprocessed transformer parameters.
    """
    parameters = {}

    def extract_layer_parameters(layer):
        """Extract parameters from a transformer layer."""
        layer_dict = {
            "attentions": {},
            "ffn": {},
            "norms": {},
        }

        # Norms
        norms = getattr(layer, "norms", [])
        for n, norm in enumerate(norms):
            if isinstance(norm, nn.LayerNorm):
                layer_dict["norms"][f"norm{n}"] = {
                    "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                }

        # FFNs
        ffns = getattr(layer, "ffns", [])
        for k, ffn in enumerate(ffns):
            if hasattr(ffn, "layers") and len(ffn.layers) >= 4:
                layer_dict["ffn"][f"ffn{k}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[3].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[3].bias, dtype=ttnn.bfloat16),
                    },
                }

        # Attentions
        attentions = getattr(layer, "attentions", [])
        for j, attn in enumerate(attentions):
            attn_params = {}

            # Check attention type and extract accordingly
            if hasattr(attn, "sampling_offsets"):  # Deformable attention
                attn_params["sampling_offsets"] = {
                    "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                }
                attn_params["attention_weights"] = {
                    "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                }
                attn_params["value_proj"] = {
                    "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                }
                attn_params["output_proj"] = {
                    "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                }

            elif hasattr(attn, "deformable_attention"):  # Spatial cross attention
                deform = attn.deformable_attention
                attn_params["sampling_offsets"] = {
                    "weight": preprocess_linear_weight(deform.sampling_offsets.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(deform.sampling_offsets.bias, dtype=ttnn.bfloat16),
                }
                attn_params["attention_weights"] = {
                    "weight": preprocess_linear_weight(deform.attention_weights.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(deform.attention_weights.bias, dtype=ttnn.bfloat16),
                }
                attn_params["value_proj"] = {
                    "weight": preprocess_linear_weight(deform.value_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(deform.value_proj.bias, dtype=ttnn.bfloat16),
                }
                attn_params["output_proj"] = {
                    "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                }

            elif hasattr(attn, "attn"):  # MultiheadAttention wrapper
                attn_params["in_proj"] = {
                    "weight": preprocess_linear_weight(attn.attn.in_proj_weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.attn.in_proj_bias, dtype=ttnn.bfloat16),
                }
                attn_params["out_proj"] = {
                    "weight": preprocess_linear_weight(attn.attn.out_proj.weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(attn.attn.out_proj.bias, dtype=ttnn.bfloat16),
                }

            layer_dict["attentions"][f"attn{j}"] = attn_params

        return layer_dict

    # Extract encoder parameters
    if hasattr(torch_model, "encoder"):
        encoder = torch_model.encoder
        parameters["encoder"] = {"layers": {}}
        for i, layer in enumerate(encoder.layers):
            parameters["encoder"]["layers"][f"layer{i}"] = extract_layer_parameters(layer)

    # Extract decoder parameters
    if hasattr(torch_model, "decoder"):
        decoder = torch_model.decoder
        parameters["decoder"] = {"layers": {}}
        for i, layer in enumerate(decoder.layers):
            parameters["decoder"]["layers"][f"layer{i}"] = extract_layer_parameters(layer)

    # Reference points
    if hasattr(torch_model, "reference_points"):
        parameters["reference_points"] = {
            "weight": preprocess_linear_weight(torch_model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(torch_model.reference_points.bias, dtype=ttnn.bfloat16),
        }

    # CAN bus MLP
    if hasattr(torch_model, "can_bus_mlp"):
        mlp = torch_model.can_bus_mlp
        parameters["can_bus_mlp"] = {
            "0": {
                "weight": preprocess_linear_weight(mlp[0].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(mlp[0].bias, dtype=ttnn.bfloat16),
            },
            "2": {
                "weight": preprocess_linear_weight(mlp[2].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(mlp[2].bias, dtype=ttnn.bfloat16),
            },
        }
        if hasattr(mlp, "norm"):
            parameters["can_bus_mlp"]["norm"] = {
                "weight": preprocess_layernorm_parameter(mlp.norm.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_layernorm_parameter(mlp.norm.bias, dtype=ttnn.bfloat16),
            }

    # Embeddings
    if hasattr(torch_model, "level_embeds"):
        parameters["level_embeds"] = ttnn.from_torch(
            torch_model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
    if hasattr(torch_model, "cams_embeds"):
        parameters["cams_embeds"] = ttnn.from_torch(
            torch_model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    return parameters


def preprocess_head_parameters(
    torch_model: nn.Module,
    device: ttnn.Device = None,
) -> Dict[str, Any]:
    """Preprocess MapTRHead parameters for TTNN.

    Args:
        torch_model: PyTorch MapTRHead model.
        device: TTNN device.

    Returns:
        Dictionary with preprocessed head parameters.
    """
    parameters = {}

    def extract_sequential_branch(module_list, dtype=ttnn.bfloat16):
        """Extract parameters from sequential branch."""
        branch_params = {}
        for i, mod in enumerate(module_list):
            layer_params = {}
            layer_index = 0

            if isinstance(mod, nn.Sequential):
                layers = list(mod)
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

    # Positional encoding
    if hasattr(torch_model, "positional_encoding") and torch_model.positional_encoding is not None:
        pos_encoding = torch_model.positional_encoding
        parameters["positional_encoding"] = {
            "row_embed": {
                "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            },
            "col_embed": {
                "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            },
        }

    # Embeddings
    if hasattr(torch_model, "bev_embedding") and torch_model.bev_embedding is not None:
        parameters["bev_embedding"] = {
            "weight": ttnn.from_torch(torch_model.bev_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }

    if hasattr(torch_model, "instance_embedding") and torch_model.instance_embedding is not None:
        parameters["instance_embedding"] = {
            "weight": ttnn.from_torch(
                torch_model.instance_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        }

    if hasattr(torch_model, "pts_embedding") and torch_model.pts_embedding is not None:
        parameters["pts_embedding"] = {
            "weight": ttnn.from_torch(torch_model.pts_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }

    if hasattr(torch_model, "query_embedding") and torch_model.query_embedding is not None:
        parameters["query_embedding"] = {
            "weight": ttnn.from_torch(torch_model.query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }

    # Branches
    parameters["branches"] = {}

    if hasattr(torch_model, "cls_branches"):
        parameters["branches"]["cls_branches"] = extract_sequential_branch(torch_model.cls_branches)

    if hasattr(torch_model, "reg_branches"):
        parameters["branches"]["reg_branches"] = extract_sequential_branch(torch_model.reg_branches)

    # Map reg branches for decoder box refinement
    if hasattr(torch_model, "reg_branches"):
        parameters["map_reg_branches"] = extract_sequential_branch(torch_model.reg_branches)

    return parameters


class MapTRModelParameters:
    """Container for all MapTR model parameters."""

    def __init__(self):
        self.backbone = None
        self.backbone_conv_args = None
        self.neck_lateral_config = None
        self.neck_fpn_config = None
        self.encoder = None
        self.decoder = None
        self.transformer = None
        self.head = None


def create_maptr_parameters_from_checkpoint(
    checkpoint_path: str,
    torch_backbone: nn.Module,
    torch_fpn: nn.Module,
    torch_transformer: nn.Module,
    torch_head: nn.Module,
    sample_input: torch.Tensor,
    device: ttnn.Device = None,
) -> MapTRModelParameters:
    """Create complete TTNN parameters from MapTR checkpoint.

    This is the main function to preprocess all components for TTNN inference.

    Args:
        checkpoint_path: Path to MapTR checkpoint.
        torch_backbone: PyTorch ResNet50 backbone model.
        torch_fpn: PyTorch FPN model.
        torch_transformer: PyTorch MapTRPerceptionTransformer model.
        torch_head: PyTorch MapTRHead model.
        sample_input: Sample input tensor for conv args inference.
        device: TTNN device.

    Returns:
        MapTRModelParameters with all preprocessed parameters.
    """
    # Load checkpoint
    state_dict = load_maptr_checkpoint(checkpoint_path)

    # Create parameters container
    params = MapTRModelParameters()

    # Extract weights by component
    backbone_weights = extract_weights_by_prefix(state_dict, BACKBONE_PREFIX)
    neck_weights = extract_weights_by_prefix(state_dict, NECK_PREFIX)
    head_weights = extract_weights_by_prefix(state_dict, HEAD_PREFIX)

    # Load weights into torch models and preprocess
    logger.info("Preprocessing backbone parameters...")
    params.backbone, params.backbone_conv_args = preprocess_backbone_parameters(
        torch_backbone, backbone_weights, sample_input, device
    )

    # Run backbone to get features for FPN
    torch_backbone.eval()
    with torch.no_grad():
        backbone_feats = torch_backbone(sample_input)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())
        elif not isinstance(backbone_feats, (list, tuple)):
            backbone_feats = [backbone_feats]

    logger.info("Preprocessing FPN parameters...")
    params.neck_lateral_config, params.neck_fpn_config, _ = preprocess_fpn_parameters(
        torch_fpn, neck_weights, backbone_feats, device
    )

    # Load head and transformer weights
    torch_head.load_state_dict(head_weights, strict=False)
    torch_head.eval()

    logger.info("Preprocessing transformer parameters...")
    transformer_params = preprocess_transformer_parameters(torch_transformer, device)
    params.encoder = ParamsWrapper(transformer_params.get("encoder", {}))
    params.decoder = ParamsWrapper(transformer_params.get("decoder", {}))
    params.transformer = AttrDict(transformer_params)

    logger.info("Preprocessing head parameters...")
    head_params = preprocess_head_parameters(torch_head, device)
    params.head = ParamsWrapper(head_params)

    logger.info("Parameter preprocessing complete")
    return params


def create_maptr_parameters_from_torch_model(
    torch_model: nn.Module,
    sample_input: torch.Tensor,
    device: ttnn.Device = None,
) -> MapTRModelParameters:
    """Create complete TTNN parameters from an already-loaded PyTorch MapTR model.

    Args:
        torch_model: Complete PyTorch MapTR model with loaded weights.
        sample_input: Sample input tensor for conv args inference.
        device: TTNN device.

    Returns:
        MapTRModelParameters with all preprocessed parameters.
    """
    params = MapTRModelParameters()

    # Extract backbone
    if hasattr(torch_model, "img_backbone"):
        logger.info("Preprocessing backbone parameters...")
        backbone = torch_model.img_backbone

        def backbone_preprocessor(model, name):
            parameters = {}

            # Initial conv + bn
            if hasattr(model, "norm1"):
                bn = model.norm1
            elif hasattr(model, "bn1"):
                bn = model.bn1
            else:
                raise ValueError("Cannot find batch norm layer")

            weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, bn)
            parameters["conv1"] = {
                "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
                "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
            }

            for layer_idx in range(1, 5):
                layer = getattr(model, f"layer{layer_idx}")
                for block_idx, block in enumerate(layer):
                    prefix = f"layer{layer_idx}_{block_idx}"
                    parameters[prefix] = {}

                    for conv_idx in [1, 2, 3]:
                        conv_name = f"conv{conv_idx}"
                        if hasattr(block, f"norm{conv_idx}"):
                            norm_name = f"norm{conv_idx}"
                        else:
                            norm_name = f"bn{conv_idx}"

                        conv = getattr(block, conv_name)
                        bn = getattr(block, norm_name)
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters[prefix][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                    if hasattr(block, "downsample") and block.downsample is not None:
                        ds = block.downsample
                        if isinstance(ds, nn.Sequential) and len(ds) >= 2:
                            w, b = fold_batch_norm2d_into_conv2d(ds[0], ds[1])
                            parameters[prefix]["downsample"] = {
                                "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                                "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                            }

            return parameters

        preprocessed = preprocess_model_parameters(
            initialize_model=lambda: backbone,
            custom_preprocessor=backbone_preprocessor,
            device=device,
        )
        params.backbone = ParamsWrapper(preprocessed)

        # Infer conv args
        params.backbone_conv_args = infer_ttnn_module_args(
            model=backbone,
            run_model=lambda model: model(sample_input),
            device=None,
        )

    # Run backbone to get features for FPN
    with torch.no_grad():
        backbone_feats = torch_model.img_backbone(sample_input)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())
        elif not isinstance(backbone_feats, (list, tuple)):
            backbone_feats = [backbone_feats]

    # Extract FPN
    if hasattr(torch_model, "img_neck"):
        logger.info("Preprocessing FPN parameters...")
        fpn = torch_model.img_neck

        # Infer conv args
        fpn_conv_args = infer_ttnn_module_args(
            model=fpn,
            run_model=lambda model: model(backbone_feats),
            device=None,
        )

        # Create conv configs
        if hasattr(fpn, "lateral_convs") and len(fpn.lateral_convs) > 0:
            lateral_conv = fpn.lateral_convs[0]
            if hasattr(lateral_conv, "conv"):
                conv_module = lateral_conv.conv
            else:
                conv_module = lateral_conv

            # Find lateral args
            lateral_args = None
            for key in fpn_conv_args.keys():
                if "lateral" in str(key).lower():
                    lateral_args = fpn_conv_args[key]
                    break
            if lateral_args is None:
                lateral_args = list(fpn_conv_args.values())[0] if fpn_conv_args else None

            if lateral_args is not None:
                params.neck_lateral_config = Conv2dConfiguration.from_model_args(
                    conv2d_args=lateral_args,
                    weights=ttnn.from_torch(conv_module.weight, dtype=ttnn.float32),
                    bias=ttnn.from_torch(conv_module.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                    if conv_module.bias is not None
                    else None,
                    activation=None,
                    sharding_strategy=AutoShardedStrategyConfiguration(),
                )

        if hasattr(fpn, "fpn_convs") and len(fpn.fpn_convs) > 0:
            fpn_conv = fpn.fpn_convs[0]
            if hasattr(fpn_conv, "conv"):
                conv_module = fpn_conv.conv
            else:
                conv_module = fpn_conv

            fpn_args = None
            args_list = list(fpn_conv_args.values())
            fpn_args = args_list[1] if len(args_list) > 1 else args_list[0] if args_list else None

            if fpn_args is not None:
                params.neck_fpn_config = Conv2dConfiguration.from_model_args(
                    conv2d_args=fpn_args,
                    weights=ttnn.from_torch(conv_module.weight, dtype=ttnn.float32),
                    bias=ttnn.from_torch(conv_module.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                    if conv_module.bias is not None
                    else None,
                    activation=None,
                    sharding_strategy=AutoShardedStrategyConfiguration(),
                )

    # Extract transformer
    if hasattr(torch_model, "pts_bbox_head") and hasattr(torch_model.pts_bbox_head, "transformer"):
        logger.info("Preprocessing transformer parameters...")
        transformer = torch_model.pts_bbox_head.transformer
        transformer_params = preprocess_transformer_parameters(transformer, device)
        params.encoder = ParamsWrapper(transformer_params.get("encoder", {}))
        params.decoder = ParamsWrapper(transformer_params.get("decoder", {}))
        params.transformer = AttrDict(transformer_params)

    # Extract head
    if hasattr(torch_model, "pts_bbox_head"):
        logger.info("Preprocessing head parameters...")
        head = torch_model.pts_bbox_head
        head_params = preprocess_head_parameters(head, device)
        params.head = ParamsWrapper(head_params)

    logger.info("Parameter preprocessing complete")
    return params
