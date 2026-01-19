# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading utilities for complete TTNN MapTR model.

This module provides functions to load and preprocess weights for the complete
TTNN MapTR model including backbone, neck, transformer (encoder + decoder), and head.
"""

import os
import torch
import torch.nn as nn
import ttnn
from typing import Dict, Optional
from loguru import logger

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
    infer_ttnn_module_args,
)
from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.modules.transformer import (
    MapTRPerceptionTransformer,
)
from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder
from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.modules.decoder import MapTRDecoder
from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import (
    TemporalSelfAttention,
)
from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
    SpatialCrossAttention,
)
from models.experimental.MapTR.dependency import ResNet
from models.experimental.MapTR.dependency import FPN
from models.experimental.MapTR.projects.mmdet3d_plugin.maptr.dense_heads.maptr_head import MapTRHead


def load_maptr_checkpoint(checkpoint_path: str) -> Dict:
    """Load MapTR checkpoint and return state dict.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        State dictionary from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    else:
        return checkpoint


def custom_preprocessor_transformer(model, name):
    """Custom preprocessor for MapTRPerceptionTransformer parameters."""
    parameters = {}

    def extract_transformer_parameters(transformer_module):
        """Extract parameters from encoder/decoder layers."""
        parameters = {"layers": {}}

        for i, layer in enumerate(transformer_module.layers):
            layer_dict = {
                "attentions": {},
                "ffn": {},
                "norms": {},
            }

            # ---- Norms ----
            for n, norm in enumerate(getattr(layer, "norms", [])):
                if isinstance(norm, nn.LayerNorm):
                    layer_dict["norms"][f"norm{n}"] = {
                        "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                    }

            # ---- FFNs ----
            for k, ffn in enumerate(getattr(layer, "ffns", [])):
                # FFN structure: layers[0] = Linear, layers[1] = activation, layers[2] = dropout, layers[3] = Linear
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

            # ---- Attentions ----
            for j, attn in enumerate(getattr(layer, "attentions", [])):
                if isinstance(attn, TemporalSelfAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif isinstance(attn, SpatialCrossAttention):
                    deform_attn = attn.deformable_attention
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(
                                deform_attn.sampling_offsets.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(deform_attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(
                                deform_attn.attention_weights.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(deform_attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(deform_attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(deform_attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif hasattr(attn, "attn"):  # MultiheadAttention wrapper
                    layer_dict["attentions"][f"attn{j}"] = {
                        "in_proj": {
                            "weight": preprocess_linear_weight(attn.attn.in_proj_weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attn.in_proj_bias, dtype=ttnn.bfloat16),
                        },
                        "out_proj": {
                            "weight": preprocess_linear_weight(attn.attn.out_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attn.out_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                else:  # CustomMSDeformableAttention
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

            parameters["layers"][f"layer{i}"] = layer_dict
        return parameters

    if isinstance(model, MapTRPerceptionTransformer):
        parameters = {}

        # Extract encoder parameters
        if hasattr(model, "encoder") and isinstance(model.encoder, BEVFormerEncoder):
            parameters["encoder"] = extract_transformer_parameters(model.encoder)

        # Extract decoder parameters
        if hasattr(model, "decoder") and isinstance(model.decoder, MapTRDecoder):
            parameters["decoder"] = extract_transformer_parameters(model.decoder)

        # Reference points
        parameters["reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }

        # CAN bus MLP: [0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU, [norm]=LayerNorm
        parameters["can_bus_mlp"] = {
            "0": {
                "weight": preprocess_linear_weight(model.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
            },
            "2": {
                "weight": preprocess_linear_weight(model.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
            },
            "norm": {
                "weight": preprocess_layernorm_parameter(model.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_layernorm_parameter(model.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16),
            },
        }

        # Embeddings
        parameters["level_embeds"] = ttnn.from_torch(
            model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
        )
        parameters["cams_embeds"] = ttnn.from_torch(
            model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
        )

    return parameters


def custom_preprocessor_backbone(model, name):
    """Custom preprocessor for ResNet backbone parameters."""
    from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

    parameters = {}
    if isinstance(model, ResNet):
        # Initial conv + bn (norm1 in dependency.py ResNet)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.norm1)
        # Weight should be in (out_channels, in_channels, kernel_h, kernel_w) format
        # Bias should be reshaped to (1, 1, 1, out_channels) for Conv2dConfiguration
        bias_reshaped = bias.reshape((1, 1, 1, -1))
        parameters["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters[prefix] = {}

                # conv1, conv2, conv3 with norm1, norm2, norm3 (dependency.py naming)
                for conv_idx in [1, 2, 3]:
                    conv_name = f"conv{conv_idx}"
                    norm_name = f"norm{conv_idx}"
                    conv = getattr(block, conv_name)
                    bn = getattr(block, norm_name)
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    # Weight should be in (out_channels, in_channels, kernel_h, kernel_w) format
                    # Bias should be reshaped to (1, 1, 1, out_channels) for Conv2dConfiguration
                    b_reshaped = b.reshape((1, 1, 1, -1))
                    parameters[prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                        "bias": ttnn.from_torch(b_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        # Weight should be in (out_channels, in_channels, kernel_h, kernel_w) format
                        # Bias should be reshaped to (1, 1, 1, out_channels) for Conv2dConfiguration
                        b_reshaped = b.reshape((1, 1, 1, -1))
                        parameters[prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                            "bias": ttnn.from_torch(b_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                        }
    return parameters


def custom_preprocessor_fpn(model, name):
    """Custom preprocessor for FPN neck parameters."""
    parameters = {}
    if isinstance(model, FPN):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name not in parameters:
                    parameters[name] = {}
                parameters[name]["weight"] = preprocess_linear_weight(module.weight, dtype=ttnn.bfloat16)
                if module.bias is not None:
                    parameters[name]["bias"] = preprocess_linear_bias(module.bias, dtype=ttnn.bfloat16)
            elif isinstance(module, nn.BatchNorm2d):
                if name not in parameters:
                    parameters[name] = {}
                parameters[name]["weight"] = preprocess_layernorm_parameter(module.weight, dtype=ttnn.bfloat16)
                parameters[name]["bias"] = preprocess_layernorm_parameter(module.bias, dtype=ttnn.bfloat16)
                parameters[name]["running_mean"] = module.running_mean
                parameters[name]["running_var"] = module.running_var
    return parameters


def custom_preprocessor_head(model, name):
    """Custom preprocessor for MapTRHead parameters."""
    parameters = {}
    if isinstance(model, MapTRHead):
        parameters["head"] = {}

        # Positional encoding
        if hasattr(model, "positional_encoding") and model.positional_encoding is not None:
            parameters["head"]["positional_encoding"] = {}
            pos_encoding = model.positional_encoding
            parameters["head"]["positional_encoding"]["row_embed"] = {
                "weight": ttnn.from_torch(
                    pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }
            parameters["head"]["positional_encoding"]["col_embed"] = {
                "weight": ttnn.from_torch(
                    pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }

        # Embeddings
        if hasattr(model, "bev_embedding") and model.bev_embedding is not None:
            parameters["head"]["bev_embedding"] = {
                "weight": ttnn.from_torch(
                    model.bev_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }

        if hasattr(model, "instance_embedding") and model.instance_embedding is not None:
            parameters["head"]["instance_embedding"] = {
                "weight": ttnn.from_torch(
                    model.instance_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }

        if hasattr(model, "pts_embedding") and model.pts_embedding is not None:
            parameters["head"]["pts_embedding"] = {
                "weight": ttnn.from_torch(
                    model.pts_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }

        if hasattr(model, "query_embedding") and model.query_embedding is not None:
            parameters["head"]["query_embedding"] = {
                "weight": ttnn.from_torch(
                    model.query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None
                )
            }

        # Branches
        parameters["head"]["branches"] = {}

        # Classification branches
        parameters["head"]["branches"]["cls_branches"] = {}
        for i, branch in enumerate(model.cls_branches):
            branch_params = {}
            for j, layer in enumerate(branch):
                if isinstance(layer, nn.Linear):
                    branch_params[str(j)] = {
                        "weight": preprocess_linear_weight(layer.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(layer.bias, dtype=ttnn.bfloat16),
                    }
                elif isinstance(layer, nn.LayerNorm):
                    branch_params[f"{j}_norm"] = {
                        "weight": preprocess_layernorm_parameter(layer.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(layer.bias, dtype=ttnn.bfloat16),
                    }
            parameters["head"]["branches"]["cls_branches"][str(i)] = branch_params

        # Regression branches
        parameters["head"]["branches"]["reg_branches"] = {}
        for i, branch in enumerate(model.reg_branches):
            branch_params = {}
            for j, layer in enumerate(branch):
                if isinstance(layer, nn.Linear):
                    branch_params[str(j)] = {
                        "weight": preprocess_linear_weight(layer.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(layer.bias, dtype=ttnn.bfloat16),
                    }
            parameters["head"]["branches"]["reg_branches"][str(i)] = branch_params

    return parameters


class ParamsWrapper:
    """Wrapper class to convert dict parameters to object attributes."""

    def __init__(self, params_dict):
        for k, v in params_dict.items():
            # Convert key to string for attribute access
            key_str = str(k) if not isinstance(k, str) else k
            setattr(self, key_str, self._dict_to_obj(v))

    def _dict_to_obj(self, d):
        if isinstance(d, dict):
            obj = type("obj", (object,), {})()
            for k, v in d.items():
                # Convert key to string for attribute access
                key_str = str(k) if not isinstance(k, str) else k
                setattr(obj, key_str, self._dict_to_obj(v))
            return obj
        return d

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __contains__(self, key):
        return hasattr(self, str(key))


def create_maptr_complete_model_parameters(
    torch_model, input_tensor, device: Optional[ttnn.Device] = None
) -> ParamsWrapper:
    """Create complete TTNN parameters for full MapTR model.

    Args:
        torch_model: PyTorch MapTR model with all components.
        input_tensor: Input tensor for inferring module args.
        device: TTNN device (optional).

    Returns:
        ParamsWrapper with all preprocessed parameters.
    """
    logger.info("Creating complete MapTR model parameters...")

    parameters = {}

    # 1. Backbone parameters
    logger.info("  - Processing backbone...")
    backbone_params = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=custom_preprocessor_backbone,
        device=device,
    )
    try:
        backbone_conv_args = infer_ttnn_module_args(
            model=torch_model.img_backbone, run_model=lambda model: model(input_tensor), device=device
        )
        if backbone_conv_args is not None:
            for key in backbone_conv_args.keys():
                backbone_conv_args[key].module = getattr(torch_model.img_backbone, key)
        parameters["conv_args"] = {"img_backbone": backbone_conv_args}
    except Exception as e:
        logger.warning(f"Could not infer backbone conv args: {e}")
        parameters["conv_args"] = {"img_backbone": {}}
    parameters["img_backbone"] = backbone_params

    # 2. FPN neck parameters
    logger.info("  - Processing FPN neck...")
    fpn_params = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=custom_preprocessor_fpn,
        device=device,
    )
    try:
        # Get FPN output for inferring args
        with torch.no_grad():
            backbone_output = torch_model.img_backbone(input_tensor)
        if isinstance(backbone_output, dict):
            backbone_output = list(backbone_output.values())
        fpn_conv_args = infer_ttnn_module_args(
            model=torch_model.img_neck, run_model=lambda model: model(backbone_output), device=device
        )
        if fpn_conv_args is not None:
            parameters["conv_args"]["img_neck"] = fpn_conv_args
        else:
            parameters["conv_args"]["img_neck"] = {}
    except Exception as e:
        logger.warning(f"Could not infer FPN conv args: {e}")
        parameters["conv_args"]["img_neck"] = {}
    parameters["img_neck"] = fpn_params

    # 3. Transformer parameters
    logger.info("  - Processing transformer...")
    transformer_params = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head.transformer,
        custom_preprocessor=custom_preprocessor_transformer,
        device=device,
    )
    parameters["transformer"] = transformer_params

    # 4. Head parameters
    logger.info("  - Processing head...")
    head_params = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=custom_preprocessor_head,
        device=device,
    )
    parameters["head"] = head_params.get("head", {})

    # Move transformer params to head for compatibility
    parameters["head"]["transformer"] = transformer_params

    logger.info("Complete MapTR model parameters created successfully!")
    return ParamsWrapper(parameters)


def create_maptr_head_parameters(torch_head: MapTRHead, device: Optional[ttnn.Device] = None) -> ParamsWrapper:
    """Create TTNN parameters for MapTRHead (for backward compatibility).

    Args:
        torch_head: PyTorch MapTRHead model.
        device: TTNN device (optional).

    Returns:
        ParamsWrapper with head parameters.
    """
    head_params = preprocess_model_parameters(
        initialize_model=lambda: torch_head,
        custom_preprocessor=custom_preprocessor_head,
        device=device,
    )
    return ParamsWrapper(head_params)
