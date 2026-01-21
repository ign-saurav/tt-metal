# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
    SpatialCrossAttention,
)
from models.experimental.MapTR.tt.spatial_cross_attention import TtSpatialCrossAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"

# Layer prefix for spatial cross attention in encoder layer 0
# MapTR uses: pts_bbox_head.transformer.encoder.layers.0.attentions.1
# attentions.0 = TemporalSelfAttention
# attentions.1 = SpatialCrossAttention (GeometrySptialCrossAttention in actual mapTR config)
SPATIAL_CROSS_ATTN_LAYER = "pts_bbox_head.transformer.encoder.layers.0.attentions.1."


def load_maptr_spatial_cross_attention_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}. " "Please download the weights first.")

    # Load full checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
    else:
        full_state_dict = checkpoint

    # Extract only spatial cross attention weights
    sca_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(SPATIAL_CROSS_ATTN_LAYER):
            # Remove the layer prefix to get the relative key
            relative_key = key[len(SPATIAL_CROSS_ATTN_LAYER) :]
            sca_weights[relative_key] = value

    logger.info(f"Loaded {len(sca_weights)} weight tensors for spatial cross attention")
    logger.info(f"Weight keys: {list(sca_weights.keys())}")

    return sca_weights


def load_torch_model_maptr(torch_model: SpatialCrossAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    sca_weights = load_maptr_spatial_cross_attention_weights(weights_path)

    # Map the checkpoint keys to model keys
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in sca_weights:
            new_state_dict[model_key] = sca_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, SpatialCrossAttention):
        parameters["spatial_cross_attention"] = {}
        parameters["spatial_cross_attention"]["sampling_offsets"] = {}
        parameters["spatial_cross_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["attention_weights"] = {}
        parameters["spatial_cross_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["value_proj"] = {}
        parameters["spatial_cross_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["output_proj"] = {}
        parameters["spatial_cross_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )

    return parameters


def create_maptr_model_parameters_sca(model: SpatialCrossAttention, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            key=input_tensor[1],
            value=input_tensor[2],
            reference_points=input_tensor[3],
            spatial_shapes=input_tensor[4],
            reference_points_cam=input_tensor[5],
            bev_mask=input_tensor[6],
            level_start_index=input_tensor[7],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_spatial_cross_attention(
    device,
    reset_seeds,
):
    # MapTR config parameters (matching maptr_tiny_r50_24e_bevformer.py)
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_first = True
    embed_dims = 256
    num_levels = 1  # _num_levels_ in config
    num_points = 8  # num_points in MSDeformableAttention3D

    # Create PyTorch model with matching deformable_attention config
    torch_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
        ),
    )

    # Load mapTR weights
    torch_model = load_torch_model_maptr(torch_model)

    # Create input tensors
    # MapTR uses bev_h=200, bev_w=100, so num_query = 200*100 = 20000
    # For testing we use smaller values
    num_query = 10000
    num_cams = 6
    num_points_per_level = 240  # 12 * 20 for spatial shape [12, 20]
    num_z_anchors = 4

    query = torch.randn(1, num_query, embed_dims)
    key = torch.randn(num_cams, num_points_per_level, 1, embed_dims)
    value = torch.randn(num_cams, num_points_per_level, 1, embed_dims)
    reference_points = torch.randn(1, num_z_anchors, num_query, 3)
    spatial_shapes = torch.tensor([[12, 20]])
    reference_points_cam = torch.randn(num_cams, 1, num_query, num_z_anchors, 2)
    bev_mask = torch.randn(num_cams, 1, num_query, num_z_anchors)
    level_start_index = torch.tensor([0])

    # Run PyTorch model
    torch_output = torch_model(
        query,
        key,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )

    # Prepare TT model parameters
    parameter = create_maptr_model_parameters_sca(
        torch_model,
        [query, key, value, reference_points, spatial_shapes, reference_points_cam, bev_mask, level_start_index],
        device,
    )

    # Create TT model
    tt_model = TtSpatialCrossAttention(
        device=device,
        params=parameter.spatial_cross_attention,
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
        ),
    )

    # Convert inputs to TT tensors
    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    bev_mask_tt = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    tt_output = tt_model(
        query_tt,
        key_tt,
        value_tt,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes_tt,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask_tt,
        level_start_index=level_start_index_tt,
    )

    # Compare outputs
    ttnn_output = ttnn.to_torch(tt_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(pcc_message)
