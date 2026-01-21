# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.experimental.centernet.reference.dlav0 import DLA, BasicBlock
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.dlav0 import DLAUp
from models.experimental.centernet.tt.dlaup import TtDLAUp
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.demos.utils.common_demo_utils import get_mesh_mappers


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_dla_up(device):
    """Test TTNN DLAUp with feature maps from PyTorch backbone."""
    torch.manual_seed(42)
    # Create DLA-34 model with return_levels=True
    dla_model = DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block=BasicBlock,
        return_levels=True,
    )

    dla_model.eval()

    # Create DLAUp
    channels = [64, 128, 256, 512]
    scales = [1, 2, 4, 8]

    dla_up = DLAUp(channels=channels, scales=scales)

    dla_up.eval()

    # Create input for the full DLA model
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 512, 512)

    # Get the actual intermediate outputs from DLA
    with torch.no_grad():
        dla_outputs = dla_model(input_tensor)
        # Extract the levels that DLAUp expects (starting from first_level)
        first_level = 2
        actual_inputs = dla_outputs[first_level:]

        logger.info(f"Actual DLA output shapes: {[x.shape for x in actual_inputs]}")

        # Now feed these to DLAUp
        dla_up_output = dla_up(actual_inputs)
        logger.info(f"✓ DLAUp forward successful! Output shape: {dla_up_output.shape}")

    # Preprocess parameters for TTNN
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: dla_up,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )

    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=dla_up, run_model=lambda model: dla_up(actual_inputs), device=device
    )

    # Extract the ACTUAL input channels from DLA backbone outputs
    actual_input_channels = []
    for tensor in actual_inputs:
        actual_input_channels.append(tensor.shape[1])

    logger.info(f"Using actual input channels from DLA outputs: {actual_input_channels}")

    # Create TTNN model with correct channels
    tt_model = TtDLAUp(
        channels=actual_input_channels,
        scales=scales,
        parameters=parameters.dla_up,
        layer_args=parameters.layer_args,
        device=device,
    )

    # Convert PyTorch backbone outputs to TTNN format
    tt_layers = []
    for torch_layer in actual_inputs:
        tt_layer = ttnn.from_torch(torch_layer.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        tt_layer = ttnn.to_device(tt_layer, device)
        tt_layers.append(tt_layer)

    # Get TTNN output
    tt_output = tt_model.forward(tt_layers)

    tt_output_torch = tt2torch_tensor(tt_output)
    logger.info(f"TTNN output shape (NHWC): {tt_output_torch.shape}")
    logger.info(f"PyTorch output shape (NCHW): {dla_up_output.shape}")

    # Convert TTNN output from NHWC to NCHW for comparison
    tt_output_nchw = tt_output_torch.permute(0, 3, 1, 2)
    logger.info(f"TTNN output shape (NCHW): {tt_output_nchw.shape}")

    passing, pcc_value = comp_pcc(dla_up_output, tt_output_nchw, pcc=0.99)

    logger.info(f"DLAUp PCC: {pcc_value}")

    assert passing, f"PCC check failed: {pcc_value} < 0.99"
