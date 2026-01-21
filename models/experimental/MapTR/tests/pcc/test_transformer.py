# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
import torch.nn as nn

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
from models.experimental.MapTR.tt.transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.decoder import TtMapTRDecoder
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def custom_preprocessor(model, name):
    parameters = {}

    def extract_transformer_parameters(transformer_module):
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
        parameters["level_embeds"] = ttnn.from_torch(model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["cams_embeds"] = ttnn.from_torch(model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return parameters


class ParamsWrapper:
    def __init__(self, params_dict):
        for k, v in params_dict.items():
            setattr(self, k, self._dict_to_obj(v))

    def _dict_to_obj(self, d):
        if isinstance(d, dict):
            obj = type("obj", (object,), {})()
            for k, v in d.items():
                setattr(obj, k, self._dict_to_obj(v))
            return obj
        return d


def create_maptr_model_parameters(model: MapTRPerceptionTransformer, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_transformer(
    device,
    reset_seeds,
):
    # Config
    embed_dims = 256
    num_feature_levels = 1
    num_cams = 6
    bev_h, bev_w = 50, 32
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_vec = 10
    num_pts_per_vec = 10
    num_query = num_vec * num_pts_per_vec
    grid_length = [0.512, 0.512]

    # Encoder config
    encoder_cfg = dict(
        type="BEVFormerEncoder",
        num_layers=1,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type="BEVFormerLayer",
            attn_cfgs=[
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    deformable_attention=dict(
                        type="MSDeformableAttention3D",
                        embed_dims=embed_dims,
                        num_points=8,
                        num_levels=1,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    # Decoder config
    decoder_cfg = dict(
        type="MapTRDecoder",
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type="DetrTransformerDecoderLayer",
            attn_cfgs=[
                dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                dict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    # Create PyTorch model
    torch_model = MapTRPerceptionTransformer(
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        embed_dims=embed_dims,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[bev_h // 2, bev_w // 2],
    )
    torch_model.eval()

    # Disable dropout for deterministic results
    for module in torch_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    # Create test inputs
    feat_h, feat_w = 28, 50
    mlvl_feats_torch = torch.randn(1, num_cams, embed_dims, feat_h, feat_w)
    bev_queries = torch.randn(bev_h * bev_w, embed_dims)
    object_query_embed = torch.randn(num_query, embed_dims * 2)
    bev_pos = torch.randn(1, embed_dims, bev_h, bev_w)

    img_metas = [
        {
            "can_bus": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.60694152,
                    -0.07634412,
                    9.87149385,
                    -0.02108691,
                    -0.01243972,
                    -0.023067,
                    8.5640597,
                    0.0,
                    0.0,
                    5.78155401,
                    0.0,
                ],
                dtype=np.float32,
            ),
            "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
            "img_shape": [(900, 1600, 3)] * num_cams,
        }
    ]

    # Run PyTorch model
    with torch.no_grad():
        torch_outputs = torch_model(
            mlvl_feats=[mlvl_feats_torch],
            lidar_feat=None,
            bev_queries=bev_queries,
            object_query_embed=object_query_embed,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
        )

    # Preprocess parameters
    parameters = create_maptr_model_parameters(torch_model, device=device)

    # Create encoder params wrapper (encoder needs .layers.layer0 attribute access)
    encoder_params = ParamsWrapper(parameters.get("encoder", {}).get("layers", {}))
    encoder_params.layers = encoder_params

    # Create decoder params wrapper
    decoder_params = ParamsWrapper(parameters.get("decoder", {}).get("layers", {}))
    decoder_params.layers = decoder_params

    # Create TT encoder
    tt_encoder = TtBEVFormerEncoder(
        params=encoder_params,
        device=device,
        num_layers=1,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=embed_dims,
        feedforward_channels=512,
        num_levels=1,
        num_points=8,
    )

    # Create TT decoder
    tt_decoder = TtMapTRDecoder(
        num_layers=6,
        embed_dims=embed_dims,
        num_heads=8,
        params=decoder_params,
        params_branches=None,
        device=device,
        feedforward_channels=512,
    )

    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                value = self[key]
                if isinstance(value, dict):
                    return AttrDict(value)
                return value
            except KeyError:
                raise AttributeError(key)

    class TransformerParams:
        def __init__(self, params_dict):
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)

    transformer_params = TransformerParams(parameters)

    # Create TT transformer
    tt_model = TtMapTRPerceptionTransformer(
        params=transformer_params,
        device=device,
        encoder=tt_encoder,
        decoder=tt_decoder,
        embed_dims=embed_dims,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[bev_h // 2, bev_w // 2],
    )

    # Convert inputs to TTNN
    tt_mlvl_feats = [ttnn.from_torch(mlvl_feats_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)]
    tt_bev_queries = ttnn.from_torch(bev_queries, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_object_query_embed = ttnn.from_torch(
        object_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_bev_pos = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run TT model
    tt_outputs = tt_model(
        mlvl_feats=tt_mlvl_feats,
        lidar_feat=None,
        bev_queries=tt_bev_queries,
        object_query_embed=tt_object_query_embed,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=tt_bev_pos,
        img_metas=img_metas,
    )

    # Compare outputs
    # torch_outputs: (bev_embed, inter_states, init_reference, inter_references)
    # tt_outputs: same structure

    # BEV embed (encoder output)
    result_bev = assert_with_pcc(torch_outputs[0], ttnn.to_torch(tt_outputs[0]).float(), 0.90)
    print(f"BEV embed PCC: {result_bev}")

    # Inter states (decoder intermediate outputs)
    result_states = assert_with_pcc(torch_outputs[1], ttnn.to_torch(tt_outputs[1]).float(), 0.90)
    print(f"Inter states PCC: {result_states}")

    # Init reference points
    result_init_ref = assert_with_pcc(torch_outputs[2], ttnn.to_torch(tt_outputs[2]).float(), 0.95)
    print(f"Init reference PCC: {result_init_ref}")

    # Inter references
    result_inter_ref = assert_with_pcc(torch_outputs[3], ttnn.to_torch(tt_outputs[3]).float(), 0.90)
    print(f"Inter references PCC: {result_inter_ref}")
