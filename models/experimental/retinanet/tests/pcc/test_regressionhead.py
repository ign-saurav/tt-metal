# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import pickle
import os
from loguru import logger
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.retinanet.tt.tt_regression_head import ttnn_retinanet_regression_head


def create_regression_head_parameters(torch_head, device, model_config):
    parameters = {}

    for fpn_idx in range(5):
        for conv_idx in range(4):
            conv_key = f"conv_block_{fpn_idx}_{conv_idx}"
            parameters[conv_key] = {}

            conv_block = torch_head.conv[conv_idx]
            conv_layer = conv_block[0]
            norm_layer = conv_block[1]

            parameters[conv_key]["input_height"] = 100
            parameters[conv_key]["input_width"] = 100

            parameters[conv_key]["conv_weight"] = ttnn.from_torch(
                conv_layer.weight, dtype=model_config["WEIGHTS_DTYPE"]
            )
            if conv_layer.bias is not None:
                parameters[conv_key]["conv_bias"] = ttnn.from_torch(
                    conv_layer.bias.reshape(1, 1, 1, -1), dtype=model_config["WEIGHTS_DTYPE"]
                )
            else:
                out_channels = conv_layer.out_channels
                torch_dtype = torch.bfloat16 if model_config["WEIGHTS_DTYPE"] == ttnn.bfloat16 else torch.float32
                parameters[conv_key]["conv_bias"] = ttnn.from_torch(
                    torch.zeros((1, 1, 1, out_channels), dtype=torch_dtype), dtype=model_config["WEIGHTS_DTYPE"]
                )

            parameters[conv_key]["norm_weight"] = ttnn.from_torch(
                norm_layer.weight, dtype=model_config["WEIGHTS_DTYPE"]
            )
            parameters[conv_key]["norm_bias"] = ttnn.from_torch(norm_layer.bias, dtype=model_config["WEIGHTS_DTYPE"])

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_retinanet_v2_regression_head_ttnn_5_fpn_final(device, pcc, reset_seeds):
    torch.manual_seed(0)

    torch_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    torch_model.eval()
    torch_model = torch_model.to(dtype=torch.bfloat16)
    regression_head = torch_model.head.regression_head

    pickle_path = "models/experimental/retinanet/data/fpn_features.pkl"

    if os.path.exists(pickle_path):
        logger.info(f"Loading FPN features from {pickle_path}")
        with open(pickle_path, "rb") as f:
            fpn_features = pickle.load(f)
        torch_features = [fpn_features[f"fpn_{i}"] for i in range(5)]
    else:
        logger.info(f"Pickle file not found at {pickle_path}, using random features")
        batch_size = 1
        in_channels = 256
        input_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]

        torch_features = [torch.randn(batch_size, in_channels, H, W, dtype=torch.bfloat16) for H, W in input_shapes]

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    input_shapes = [(feat.shape[-2], feat.shape[-1]) for feat in torch_features]
    in_channels = 256
    num_anchors = 9
    batch_size = 1

    ttnn_features = []
    for feat in torch_features:
        ttnn_feat = ttnn.from_torch(
            feat.permute(0, 2, 3, 1),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        ttnn_features.append(ttnn_feat)

    with torch.no_grad():
        torch_output = regression_head(torch_features)

    ttnn_parameters = create_regression_head_parameters(regression_head, device, model_config)

    ttnn_output = ttnn_retinanet_regression_head(
        ttnn_features,
        parameters=ttnn_parameters,
        device=device,
        model_config=model_config,
        num_anchors=num_anchors,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    passed, pcc_msg = assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)
    logger.info(f"Regression Head PCC: {pcc_msg}")
    assert passed, f"PCC test failed: {pcc_msg}"
