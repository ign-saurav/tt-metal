# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.network.dlav0 import DLAUp
from models.experimental.centernet.tt.tt_dlaup import TtDLAUp
from models.experimental.centernet.tt.custom_preprocessor import create_dla_up_preprocessor

WEIGHTS_PATH = "models/experimental/centernet/ctdet_coco_dlav0_1x.pth"


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_dla_up_debug(device):
    """Debug test to inspect model structure AFTER loading weights."""
    torch.manual_seed(42)

    channels = [64, 128, 256, 512]
    scales = [1, 2, 4, 8]

    # Create and load model
    pytorch_dla_up = DLAUp(channels=channels, scales=scales)

    # Load pretrained weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    dla_up_state_dict = {k.replace("dla_up.", ""): v for k, v in state_dict.items() if k.startswith("dla_up.")}
    pytorch_dla_up.load_state_dict(dla_up_state_dict, strict=False)

    # INSPECT THE ACTUAL MODEL STRUCTURE AFTER LOADING WEIGHTS
    logger.info("=== Model structure AFTER loading weights ===")

    for i in range(len(channels) - 1):
        ida = getattr(pytorch_dla_up, f"ida_{i}")
        logger.info(f"\nIDAUp {i}:")
        logger.info(f"  channels: {ida.channels}")
        logger.info(f"  out_dim: {ida.out_dim}")

        # Check what each projection layer ACTUALLY expects
        for j in range(len(ida.channels)):
            proj = getattr(ida, f"proj_{j}")
            if hasattr(proj, "__class__") and proj.__class__.__name__ != "Identity":
                # This is the ACTUAL expected input channels
                expected_in = proj[0].weight.shape[1]
                expected_out = proj[0].weight.shape[0]
                logger.info(f"  proj_{j}: expects {expected_in} channels, outputs {expected_out}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_dla_up(device):
    """Test TTNN DLAUp with feature maps from PyTorch backbone."""
    torch.manual_seed(42)

    # Create the full DLA model to get correct intermediate outputs
    from models.experimental.centernet.reference.network.dlav0 import DLA, BasicBlock

    # Create DLA-34 model with return_levels=True
    dla_model = DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block=BasicBlock,
        return_levels=True,  # Enable returning intermediate feature maps
    )

    # Load the same pretrained weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load weights into full DLA model
    dla_model.load_state_dict(state_dict, strict=False)
    dla_model.eval()

    # Create DLAUp
    channels = [64, 128, 256, 512]
    scales = [1, 2, 4, 8]

    dla_up = DLAUp(channels=channels, scales=scales)

    # Load DLAUp weights
    dla_up_state_dict = {k.replace("dla_up.", ""): v for k, v in state_dict.items() if k.startswith("dla_up.")}
    dla_up.load_state_dict(dla_up_state_dict, strict=False)
    dla_up.eval()

    # Create input for the full DLA model
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 512, 512)

    # Get the actual intermediate outputs from DLA
    with torch.no_grad():
        dla_outputs = dla_model(input_tensor)
        # Extract the levels that DLAUp expects (starting from first_level)
        first_level = 2  # Based on DLASeg configuration
        actual_inputs = dla_outputs[first_level:]

        logger.info(f"Actual DLA output shapes: {[x.shape for x in actual_inputs]}")

        # Now feed these to DLAUp
        dla_up_output = dla_up(actual_inputs)
        logger.info(f"✓ DLAUp forward successful! Output shape: {dla_up_output.shape}")

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: dla_up,
        custom_preprocessor=create_dla_up_preprocessor(),
        device=None,
    )

    # Extract the ACTUAL input channels from DLA backbone outputs
    actual_input_channels = []
    for tensor in actual_inputs:
        actual_input_channels.append(tensor.shape[1])  # Get channel dimension

    logger.info(f"Using actual input channels from DLA outputs: {actual_input_channels}")

    # Create TTNN model with correct channels
    tt_model = TtDLAUp(
        channels=actual_input_channels,
        scales=scales,
        parameters=parameters,
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

    # Convert back and compare
    # TTNN output is in NHWC format, PyTorch is in NCHW format
    tt_output_torch = tt2torch_tensor(tt_output)  # Returns in NHWC [N, H, W, C]
    logger.info(f"TTNN output shape (NHWC): {tt_output_torch.shape}")
    logger.info(f"PyTorch output shape (NCHW): {dla_up_output.shape}")

    # Convert TTNN output from NHWC to NCHW for comparison
    tt_output_nchw = tt_output_torch.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    logger.info(f"TTNN output shape (NCHW): {tt_output_nchw.shape}")

    passing, pcc_value = comp_pcc(dla_up_output, tt_output_nchw, pcc=0.99)

    logger.info(f"DLAUp PCC: {pcc_value}")

    assert passing, f"PCC check failed: {pcc_value} < 0.99"
