# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.neck import TtNeck
from models.experimental.pointpillars.reference.pointpillars import Neck
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.pointpillars.common import load_checkpoint, extract_component_state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channels,upsample_strides,out_channels",
    [
        ([64, 128, 256], [1, 2, 4], [128, 128, 128]),
    ],
)
def test_neck(device, in_channels, upsample_strides, out_channels, reset_seeds):
    """Test TtNeck against PyTorch reference."""
    torch.manual_seed(0)

    torch_model = Neck(in_channels, upsample_strides, out_channels)

    state_dict = load_checkpoint("epoch_160.pth")
    if state_dict is not None:
        neck_state_dict = extract_component_state_dict(state_dict, "neck.")
        torch_model.load_state_dict(neck_state_dict)

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create input tensors
    torch_inputs = [
        torch.randn(1, 64, 248, 216, dtype=torch.bfloat16),
        torch.randn(1, 128, 124, 108, dtype=torch.bfloat16),
        torch.randn(1, 256, 62, 54, dtype=torch.bfloat16),
    ]

    torch_output = torch_model(torch_inputs)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    # Convert inputs to TTNN format
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

    tt_model = TtNeck(
        in_channels=in_channels,
        upsample_strides=upsample_strides,
        out_channels=out_channels,
        parameters=parameters["neck"],
        device=device,
    )

    tt_output = tt_model.forward(ttnn_inputs)

    tt_output_torch = tt2torch_tensor(tt_output).permute(0, 3, 1, 2)

    passing, pcc = comp_pcc(torch_output, tt_output_torch, 0.99)
    logger.info(f"Neck PCC: {pcc}")
    assert passing, f"Neck PCC check failed: {pcc}"

    assert (
        torch_output.shape == tt_output_torch.shape
    ), f"Shape mismatch: {torch_output.shape} vs {tt_output_torch.shape}"

    logger.info("Neck test passed!")
