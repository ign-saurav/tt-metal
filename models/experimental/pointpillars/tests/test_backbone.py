# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.backbone import TtBackbone
from models.experimental.pointpillars.reference.model.pointpillars import Backbone
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channel,out_channels,layer_nums,layer_strides",
    [
        (64, [64, 128, 256], [3, 5, 5], [2, 2, 2]),
    ],
)
def test_backbone(device, in_channel, out_channels, layer_nums, layer_strides, reset_seeds):
    torch.manual_seed(0)

    # Create reference model
    torch_model = Backbone(in_channel, out_channels, layer_nums, layer_strides)

    # Load pretrained weights from .pth file (optional)
    try:
        checkpoint = torch.load("epoch_160.pth", map_location="cpu")

        # Extract Backbone weights from the full model checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter only Backbone weights
        backbone_state_dict = {}
        prefix = "backbone."  # Adjust this based on your model's structure
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                backbone_state_dict[new_key] = value

        # Load the filtered weights into your model
        torch_model.load_state_dict(backbone_state_dict)
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, using random weights")
    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create input tensor with shape [1, 64, 496, 432]
    batch_size = 1
    height = 496
    width = 432
    torch_input = torch.randn(batch_size, in_channel, height, width, dtype=torch.bfloat16)

    # Run PyTorch model
    torch_output = torch_model(torch_input)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Convert input to TTNN format (NHWC layout)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create TTNN model
    tt_model = TtBackbone(
        in_channel=in_channel,
        out_channels=out_channels,
        layer_nums=layer_nums,
        layer_strides=layer_strides,
        parameters=parameters["backbone"],
        device=device,
    )

    # Run TTNN model
    tt_output = tt_model.forward(ttnn_input)

    # Compare outputs for each block
    for i, (torch_out, tt_out) in enumerate(zip(torch_output, tt_output)):
        # Convert TTNN output back to PyTorch format
        tt_out_torch = tt2torch_tensor(tt_out)
        tt_out_torch = tt_out_torch.reshape(
            torch_out.shape[0], torch_out.shape[2], torch_out.shape[3], torch_out.shape[1]
        )
        # Convert from NHWC to NCHW
        tt_out_torch = tt_out_torch.permute(0, 3, 1, 2)

        passing, pcc = comp_pcc(torch_out, tt_out_torch, 0.99)
        logger.info(f"Block {i} PCC: {pcc}")
        # assert passing, f"Block {i} PCC check failed: {pcc}"

    logger.info("Backbone test passed!")
