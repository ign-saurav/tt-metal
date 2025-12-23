# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import pickle
import os
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger
from models.experimental.retinanet.tt.tt_classification_head import TtnnRetinaNetClassificationHead
from models.experimental.retinanet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_classification_head_full(device, pcc, reset_seeds):
    """Test TTNN classification head implementation with 5 FPN levels using real or random features."""
    torch.manual_seed(0)

    torch_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    torch_model.eval()
    torch_model = torch_model.to(dtype=torch.bfloat16)
    classification_head = torch_model.head.classification_head

    pickle_path = "fpn_features.pkl"

    if os.path.exists(pickle_path):
        logger.info(f"✓ Loading real FPN features from {pickle_path}")
        with open(pickle_path, "rb") as f:
            saved_data = pickle.load(f)

        torch_features = saved_data["features"]
        input_shapes = saved_data["input_shapes"]
        batch_size = saved_data["batch_size"]
        in_channels = saved_data["in_channels"]

        logger.info(f"  Loaded {len(torch_features)} FPN levels:")
        for i, feat in enumerate(torch_features):
            logger.info(f"    Level {i}: {feat.shape}")
    else:
        logger.info(f"Pickle file not found at {pickle_path}, using random features")
        batch_size = 1
        in_channels = 256
        input_shapes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]

        torch_features = [torch.randn(batch_size, in_channels, H, W, dtype=torch.bfloat16) for H, W in input_shapes]

    num_anchors = 9
    num_classes = 91

    with torch.no_grad():
        pickle_path = "torch_output_classification.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                torch_output = pickle.load(f)
        else:
            print("running : reference model")
            torch_output = classification_head(torch_features)
            with open(pickle_path, "wb") as f:
                pickle.dump(torch_output, f)
            print("finished running : reference model")

    ttnn_features = [
        ttnn.from_torch(
            feature.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for feature in torch_features
    ]

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: classification_head,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )

    ttnn_head = TtnnRetinaNetClassificationHead(
        parameters=ttnn_parameters,
        device=device,
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        batch_size=batch_size,
        input_shapes=input_shapes,
        model_config=model_config,
        optimization_profile="optimized",
    )

    ttnn_output = ttnn_head.forward(feature_maps=ttnn_features)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    passed, pcc_msg = assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)
    logger.info(f"Classification Head PCC: {pcc_msg}")
    assert passed, f"Classification Head test failed: {pcc_msg}"
