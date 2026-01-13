# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test for MapTR ResNet50 backbone using tt_cnn format.
"""

import os
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mapTR.reference import pytorch_resnet as backbone
from models.experimental.mapTR.reference.pytorch_resnet import ResNet
from models.experimental.mapTR.tt.ttcnn_backbone import TtResNet50
from tests.ttnn.utils_for_testing import assert_with_pcc


MAPTR_WEIGHTS_PATH = "models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e.pth"
BACKBONE_LAYER = "img_backbone."


def load_maptr_backbone_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load and isolate backbone weights from mapTR checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MapTR weights not found at {weights_path}.")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
    else:
        full_state_dict = checkpoint

    backbone_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(BACKBONE_LAYER):
            relative_key = key[len(BACKBONE_LAYER) :]
            backbone_weights[relative_key] = value

    logger.info(f"Loaded {len(backbone_weights)} weight tensors for backbone")
    return backbone_weights


def load_torch_model_maptr(torch_model: ResNet, weights_path: str = MAPTR_WEIGHTS_PATH):
    """Load mapTR weights into the ResNet model."""
    backbone_weights = load_maptr_backbone_weights(weights_path)
    state_dict = {k: v for k, v in backbone_weights.items()}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 11 * 8192}], indirect=True)
def test_maptr_ttcnn_backbone(
    device,
    reset_seeds,
):
    """Test MapTR backbone using tt_cnn format."""
    batch_size = 6
    input_height = 384
    input_width = 640

    # Create and load PyTorch model
    torch_model = backbone.ResNet(
        layers=[3, 4, 6, 3],
        out_indices=(3,),
        block=backbone.Bottleneck,
    )
    torch_model = load_torch_model_maptr(torch_model)

    # Create input tensor
    torch_input = torch.randn((batch_size, 3, input_height, input_width), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    # Run PyTorch model
    torch_output = torch_model(torch_input)[0]

    # Create TT model using tt_cnn format
    tt_model = TtResNet50(
        torch_model=torch_model,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    # Prepare input for TT model (NHWC format, flattened)
    ttnn_input = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn_input.reshape(
        1,
        1,
        batch_size * input_height * input_width,
        3,
    )
    ttnn_input = ttnn.from_torch(ttnn_input, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    ttnn_output = tt_model(ttnn_input)[0]

    # Convert output back to PyTorch format
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0],
        torch_output.shape[2],
        torch_output.shape[3],
        torch_output.shape[1],
    ).to(torch.float32)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.96)
    logger.info(pcc_message)
