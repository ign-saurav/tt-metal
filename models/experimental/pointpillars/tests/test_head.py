# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.pointpillars.reference.model.pointpillars import Head as PyTorchHead
from models.experimental.pointpillars.tt.tt_head import (
    TtPointPillarsHead,
    preprocess_head_parameters,
    PointPillarsHeadConfig,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_pointpillars_head(device, batch_size, reset_seeds):
    """
    Test PointPillars Head implementation comparing PyTorch and TTNN outputs.
    Tests all three head outputs: classification, regression, and direction classification.
    """

    # Model configuration
    n_classes = 3
    n_anchors = 6  # 2 anchors per class × 3 classes
    in_channels = 384  # From neck output

    # Input dimensions from neck output
    input_height = 248
    input_width = 216

    logger.info("=" * 80)
    logger.info("PointPillars Head PCC Test")
    logger.info("=" * 80)

    # Create model configuration
    config = PointPillarsHeadConfig()
    logger.info(f"Using configuration:")
    logger.info(f"  Input dtype: {config.input_dtype}")
    logger.info(f"  Weight dtype: {config.weights_dtype}")
    # logger.info(f"  Compute kernel config: {config.compute_kernel_config}")

    # Load PyTorch model
    logger.info("\nLoading PyTorch Head model...")
    pytorch_model = PyTorchHead(n_anchors=n_anchors, n_classes=n_classes, in_channel=in_channels)

    # Load checkpoint weights
    checkpoint_path = "models/experimental/pointpillars/reference/model/epoch_160.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract head weights from checkpoint (keys are like "head.conv_cls.weight")
    head_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("head."):
            # Remove "head." prefix to match PyTorch model's state_dict keys
            new_key = key.replace("head.", "")
            head_state_dict[new_key] = value

    # Load weights into PyTorch model
    pytorch_model.load_state_dict(head_state_dict)
    pytorch_model = pytorch_model.to(torch.bfloat16)
    pytorch_model.eval()
    logger.info("PyTorch Head model loaded successfully")

    # Create input tensor (from neck output)
    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)
    logger.info(f"\nInput shape (NCHW): {torch_input.shape}")

    # Run PyTorch model
    logger.info("Running PyTorch Head forward pass...")
    with torch.no_grad():
        torch_cls, torch_reg, torch_dir = pytorch_model(torch_input)

    logger.info(f"\nPyTorch outputs:")
    logger.info(f"  Classification: {torch_cls.shape}")
    logger.info(f"  Regression: {torch_reg.shape}")
    logger.info(f"  Direction: {torch_dir.shape}")

    # Preprocess parameters for TTNN
    logger.info("\nPreprocessing Head parameters for TTNN...")
    parameters = preprocess_head_parameters(
        checkpoint,
        device,
        config=config,
        batch_size=batch_size,
    )

    # Create TTNN model with configuration
    logger.info("Creating TTNN Head model...")
    ttnn_model = TtPointPillarsHead(
        device=device,
        parameters=parameters,
        config=config,
        batch_size=batch_size,
    )

    # Convert input to TTNN format (NCHW -> NHWC)
    logger.info("\nConverting input to TTNN format...")
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # NCHW -> NHWC
        device=device,
        dtype=config.input_dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"TTNN input shape (NHWC): {ttnn_input.shape}")

    # Run TTNN model
    logger.info("Running TTNN Head forward pass...")
    ttnn_cls, ttnn_reg, ttnn_dir = ttnn_model(ttnn_input)

    logger.info(f"\nTTNN outputs (NHWC):")
    logger.info(f"  Classification: {ttnn_cls.shape}")
    logger.info(f"  Regression: {ttnn_reg.shape}")
    logger.info(f"  Direction: {ttnn_dir.shape}")

    # Convert TTNN outputs to PyTorch (NHWC -> NCHW)
    logger.info("\nConverting TTNN outputs to PyTorch format...")
    ttnn_cls_torch = ttnn.to_torch(ttnn_cls).permute(0, 3, 1, 2)
    ttnn_reg_torch = ttnn.to_torch(ttnn_reg).permute(0, 3, 1, 2)
    ttnn_dir_torch = ttnn.to_torch(ttnn_dir).permute(0, 3, 1, 2)

    logger.info(f"Converted TTNN outputs (NCHW):")
    logger.info(f"  Classification: {ttnn_cls_torch.shape}")
    logger.info(f"  Regression: {ttnn_reg_torch.shape}")
    logger.info(f"  Direction: {ttnn_dir_torch.shape}")

    # Compare outputs
    logger.info("\n" + "=" * 80)
    logger.info("PCC Results:")
    logger.info("=" * 80)

    all_tests_pass = True
    pcc_results = []

    # Test classification head
    logger.info("\n1. Classification Head (conv_cls):")
    logger.info(f"   PyTorch shape: {torch_cls.shape}")
    logger.info(f"   TTNN shape: {ttnn_cls_torch.shape}")
    try:
        passed_cls, pcc_cls = assert_with_pcc(torch_cls, ttnn_cls_torch, pcc=0.99)
        status_cls = "✅ PASSED" if passed_cls else "❌ FAILED"
        logger.info(f"   {status_cls} - PCC: {pcc_cls:.6f}")
        pcc_results.append(("Classification", passed_cls, pcc_cls))
        if not passed_cls:
            all_tests_pass = False
    except AssertionError as e:
        logger.warning(f"   ❌ FAILED - {str(e)}")
        pcc_results.append(("Classification", False, 0.0))
        all_tests_pass = False

    # Test regression head
    logger.info("\n2. Regression Head (conv_reg):")
    logger.info(f"   PyTorch shape: {torch_reg.shape}")
    logger.info(f"   TTNN shape: {ttnn_reg_torch.shape}")
    try:
        passed_reg, pcc_reg = assert_with_pcc(torch_reg, ttnn_reg_torch, pcc=0.99)
        status_reg = "✅ PASSED" if passed_reg else "❌ FAILED"
        logger.info(f"   {status_reg} - PCC: {pcc_reg:.6f}")
        pcc_results.append(("Regression", passed_reg, pcc_reg))
        if not passed_reg:
            all_tests_pass = False
    except AssertionError as e:
        logger.warning(f"   ❌ FAILED - {str(e)}")
        pcc_results.append(("Regression", False, 0.0))
        all_tests_pass = False

    # Test direction classification head
    logger.info("\n3. Direction Classification Head (conv_dir_cls):")
    logger.info(f"   PyTorch shape: {torch_dir.shape}")
    logger.info(f"   TTNN shape: {ttnn_dir_torch.shape}")
    try:
        passed_dir, pcc_dir = assert_with_pcc(torch_dir, ttnn_dir_torch, pcc=0.99)
        status_dir = "✅ PASSED" if passed_dir else "❌ FAILED"
        logger.info(f"   {status_dir} - PCC: {pcc_dir:.6f}")
        pcc_results.append(("Direction", passed_dir, pcc_dir))
        if not passed_dir:
            all_tests_pass = False
    except AssertionError as e:
        logger.warning(f"   ❌ FAILED - {str(e)}")
        pcc_results.append(("Direction", False, 0.0))
        all_tests_pass = False

    # Summary
    logger.info("\n" + "=" * 80)
    if all_tests_pass:
        logger.info("✅ All head outputs passed PCC check!")
    else:
        failed_heads = [name for name, passed, _ in pcc_results if not passed]
        logger.info(f"❌ Failed heads: {failed_heads}")
    logger.info("=" * 80)

    # Print detailed results
    logger.info("\nDetailed Results:")
    for name, passed, pcc in pcc_results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {name}: {status} (PCC: {pcc:.6f})")

    assert all_tests_pass, f"PCC test failed for some head outputs. Results: {pcc_results}"
    logger.info("\n✓ PointPillars Head test PASSED!")
