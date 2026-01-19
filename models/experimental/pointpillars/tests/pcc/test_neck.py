# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.neck import TtNeck
from models.experimental.pointpillars.reference.model.pointpillars import Neck
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channels,upsample_strides,out_channels",
    [
        ([64, 128, 256], [1, 2, 4], [128, 128, 128]),
    ],
)
def test_neck(device, in_channels, upsample_strides, out_channels, reset_seeds):
    torch.manual_seed(0)

    # Create reference model
    torch_model = Neck(in_channels, upsample_strides, out_channels)

    try:
        checkpoint = torch.load("epoch_160.pth", map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter only Neck weights
        neck_state_dict = {}
        prefix = "neck."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                neck_state_dict[new_key] = value

        torch_model.load_state_dict(neck_state_dict)
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, using random weights")

    # Convert model to bfloat16
    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create input tensors matching the shapes from Backbone outputs
    torch_inputs = [
        torch.randn(1, 64, 248, 216, dtype=torch.bfloat16),  # Block 0 output
        torch.randn(1, 128, 124, 108, dtype=torch.bfloat16),  # Block 1 output
        torch.randn(1, 256, 62, 54, dtype=torch.bfloat16),  # Block 2 output
    ]

    # Run PyTorch model
    torch_output = torch_model(torch_inputs)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Convert inputs to TTNN format (NHWC layout)
    ttnn_inputs = []
    for torch_input in torch_inputs:
        ttnn_input = ttnn.from_torch(
            torch_input.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_inputs.append(ttnn_input)

    # Create TTNN model
    tt_model = TtNeck(
        in_channels=in_channels,
        upsample_strides=upsample_strides,
        out_channels=out_channels,
        parameters=parameters["neck"],
        device=device,
    )

    # Run TTNN model
    tt_output = tt_model.forward(ttnn_inputs)

    # Convert TTNN output back to PyTorch format
    tt_output_torch = tt2torch_tensor(tt_output)
    # Convert from NHWC to NCHW
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)

    # Compare outputs
    passing, pcc = comp_pcc(torch_output, tt_output_torch, 0.99)
    logger.info(f"Neck PCC: {pcc}")
    assert passing, f"Neck PCC check failed: {pcc}"

    # Verify output shape
    assert (
        torch_output.shape == tt_output_torch.shape
    ), f"Shape mismatch: {torch_output.shape} vs {tt_output_torch.shape}"

    logger.info("Neck test passed!")
