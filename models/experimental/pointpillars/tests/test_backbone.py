# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.pointpillars.reference.model.pointpillars import Backbone as PyTorchBackbone
from models.experimental.pointpillars.tt.tt_backbone import (
    TtPointPillarsBackbone,
    preprocess_backbone_parameters,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_pointpillars_backbone_complete(device, batch_size, reset_seeds):
    """Test complete PointPillars backbone PCC between PyTorch and TTNN."""

    # Input dimensions (BEV feature map from pillar encoder)
    input_height = 496
    input_width = 432
    in_channels = 64

    # Expected output dims: stride=2 at every block
    expected_output_dims = [
        (248, 216, 64),  # Block 0: H/2, W/2, 64
        (124, 108, 128),  # Block 1: H/4, W/4, 128
        (62, 54, 256),  # Block 2: H/8, W/8, 256
    ]

    # ---------------------------------------------------------
    #  Create PyTorch Backbone
    # ---------------------------------------------------------
    torch_backbone = PyTorchBackbone(in_channel=in_channels, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])

    # Load pretrained weights
    checkpoint = torch.load("models/experimental/pointpillars/reference/model/epoch_160.pth", map_location="cpu")

    backbone_state = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if k.startswith("backbone.")}

    torch_backbone.load_state_dict(backbone_state, strict=True)
    torch_backbone = torch_backbone.eval().to(torch.bfloat16)
    logger.info("PyTorch model loaded successfully")

    # ---------------------------------------------------------
    #  Create Input
    # ---------------------------------------------------------
    torch.manual_seed(0)
    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_outputs = torch_backbone(torch_input)

    logger.info(f"PyTorch outputs: {[out.shape for out in torch_outputs]}")

    # ---------------------------------------------------------
    #  Preprocess params for TTNN
    # ---------------------------------------------------------
    parameters = preprocess_backbone_parameters(
        torch_backbone,
        backbone_state,
        device,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
    )

    # Convert input to NHWC for TTNN
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    # ---------------------------------------------------------
    #  TTNN Backbone Forward
    # ---------------------------------------------------------
    ttnn_backbone = TtPointPillarsBackbone(device=device, parameters=parameters, batch_size=batch_size)

    ttnn_outputs = ttnn_backbone(ttnn_input)
    logger.info(f"TTNN outputs: {[out.shape for out in ttnn_outputs]}")

    # ---------------------------------------------------------
    #  Validate Output Count
    # ---------------------------------------------------------
    assert len(ttnn_outputs) == len(torch_outputs), "Number of outputs mismatch"

    pcc_results = []
    all_tests_pass = True

    # ---------------------------------------------------------
    #  PCC Comparison for All 3 Blocks
    # ---------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PCC Results for Each Block Output:")
    logger.info("=" * 80)

    pcc_results = []
    all_tests_pass = True

    for block_idx, (torch_out, ttnn_out) in enumerate(zip(torch_outputs, ttnn_outputs)):
        # Convert TTNN output from NHWC to NCHW
        ttnn_out_torch = ttnn.to_torch(ttnn_out).permute(0, 3, 1, 2)

        # Log shapes for verification
        logger.info(f"\nBlock {block_idx}:")
        logger.info(f"  TTNN output shape (NHWC): {ttnn_out.shape}")
        logger.info(f"  TTNN output shape (NCHW): {ttnn_out_torch.shape}")
        logger.info(f"  PyTorch output shape: {torch_out.shape}")

        # Compute PCC
        passed, pcc_value = assert_with_pcc(torch_out, ttnn_out_torch, pcc=0.99)

        # Log PCC result
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {status} - PCC: {pcc_value:.6f}")

        if not passed:
            all_tests_pass = False

        pcc_results.append((block_idx, passed, pcc_value))

    # Summary
    logger.info("=" * 80)
    if all_tests_pass:
        logger.info("✅ All blocks passed PCC check!")
    else:
        failed_blocks = [idx for idx, passed, _ in pcc_results if not passed]
        logger.info(f"❌ Failed blocks: {failed_blocks}")
    logger.info("=" * 80)

    assert all_tests_pass, f"PCC test failed for some outputs. Results: {pcc_results}"
