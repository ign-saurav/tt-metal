# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import os
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mapTR.reference import pytorch_resnet as backbone
from models.experimental.mapTR.reference.pytorch_resnet import ResNet
from models.experimental.mapTR.tt.ttcnn_backbone import TtResNet50
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)


MAPTR_WEIGHTS_PATH = "models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth"
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


def custom_preprocessor(model, name):
    """Custom preprocessor for MapTR backbone following vadv2 pattern."""
    parameters = {}
    if isinstance(model, ResNet):
        parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters["res_model"][prefix] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    bn = getattr(block, f"bn{conv_name[-1]}")
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    parameters["res_model"][prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

        return parameters


def create_maptr_model_parameters(model: ResNet, input_tensor, device=None):
    """Create MapTR model parameters following vadv2 pattern."""
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    # Infer convolution arguments (device=None to avoid device being added to conv_args)
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input_tensor),
        device=None,
    )

    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 11 * 8192}], indirect=True)
def test_maptr_ttcnn_backbone(
    device,
    reset_seeds,
):
    """Test MapTR backbone using tt_cnn format"""
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

    # Create model parameters following vadv2 pattern
    parameter = create_maptr_model_parameters(torch_model, torch_input, device=device)

    # Create TT model using tt_cnn format
    tt_model = TtResNet50(
        parameter.conv_args,
        parameter.res_model,
        device,
    )

    # Prepare input for TT model (NHWC format, flattened)
    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        batch_size * input_height * input_width,
        3,
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16)

    # Run TT model
    ttnn_output = tt_model(ttnn_input_tensor, batch_size=batch_size)[0]

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
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.97)
    logger.info(pcc_message)
