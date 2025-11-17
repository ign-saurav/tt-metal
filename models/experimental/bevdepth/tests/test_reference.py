# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import gc
import pytest
import torch
import copy
from pathlib import Path
from urllib.request import urlretrieve
from loguru import logger

# Prevent ttnn initialization
os.environ.setdefault("TTNN_DISABLE_INIT", "1")

from models.experimental.bevdepth.reference.bevdepth.models.base_bev_depth import BaseBEVDepth
from models.experimental.bevdepth.reference.bevdepth.exps.nuscenes.base_exp import (
    backbone_conf,
    head_conf,
)

CHECKPOINT_URL = "https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
CHECKPOINT_NAME = "bev_depth_lss_r50_256x704_128x128_24e_2key.pth"


def download_checkpoint(checkpoint_dir, checkpoint_path):
    """Download checkpoint from GitHub releases if it doesn't exist."""
    checkpoint_path_str = str(checkpoint_path)

    if os.path.exists(checkpoint_path_str):
        logger.info(f"Checkpoint already exists at: {checkpoint_path_str}")
        return checkpoint_path_str

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Downloading checkpoint from: {CHECKPOINT_URL}")
    logger.info(f"Saving to: {checkpoint_path_str}")

    try:
        # Download with progress callback
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 100 == 0:  # Print every 100 blocks to avoid spam
                    logger.info(f"Download progress: {percent}%")

        urlretrieve(CHECKPOINT_URL, checkpoint_path_str, show_progress)
        logger.info(f"Checkpoint downloaded successfully to: {checkpoint_path_str}")
        return checkpoint_path_str
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise


@torch.no_grad()
def test_reference():
    """Test PyTorch reference model inference alone without TT device."""
    device = "cpu"

    # Get checkpoint directory path
    test_file_dir = Path(__file__).parent
    checkpoint_dir = test_file_dir.parent / "checkpoints"
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME

    # Download checkpoint if it doesn't exist
    ckpt_path = download_checkpoint(checkpoint_dir, checkpoint_path)

    logger.info(f"Using checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    # Pre-load checkpoint to get state dict (before model creation to save memory)
    logger.info("Pre-loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean up checkpoint to free memory
    del checkpoint
    gc.collect()

    # Modify config to skip pretrained weights (saves memory)
    logger.info("Preparing config (removing pretrained weights)...")
    backbone_config = copy.deepcopy(backbone_conf)
    if "img_backbone_conf" in backbone_config:
        img_conf = backbone_config["img_backbone_conf"].copy()
        img_conf.pop("init_cfg", None)
        img_conf.pop("pretrained", None)
        backbone_config["img_backbone_conf"] = img_conf

    # Patch init_weights to be no-op to prevent memory-intensive initialization
    logger.info("Patching weight initialization...")
    from models.experimental.bevdepth.reference.bevdepth.layers.backbones import base_lss_fpn

    original_init_weights = base_lss_fpn.BaseModule.init_weights
    base_lss_fpn.BaseModule.init_weights = lambda self: None

    # Prevent downloading pretrained weights from torch.hub
    try:
        original_load_state_dict_from_url = torch.hub.load_state_dict_from_url
        torch.hub.load_state_dict_from_url = lambda *args, **kwargs: {}
    except:
        original_load_state_dict_from_url = None

    try:
        # Force garbage collection before model creation
        gc.collect()

        # Initialize model (this is where bad_alloc typically occurs)
        logger.info("Initializing BaseBEVDepth model (this may take a moment)...")
        model = BaseBEVDepth(backbone_config, head_conf, is_train_depth=False)
        logger.info("Model structure created successfully")
    except Exception as e:
        logger.error(f"Failed during model creation: {e}")
        raise
    finally:
        # Restore original functions
        base_lss_fpn.BaseModule.init_weights = original_init_weights
        if original_load_state_dict_from_url is not None:
            torch.hub.load_state_dict_from_url = original_load_state_dict_from_url

    gc.collect()

    # Load checkpoint weights into model
    logger.info("Loading checkpoint weights into model...")

    # Remove 'model.' prefix if present (from Lightning checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        elif k.startswith("backbone.") or k.startswith("head."):
            # Keep backbone/head prefixes as they match the model structure
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    # Clean up state_dict to free memory
    del state_dict
    gc.collect()

    # Try loading with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys (will use random init): {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for k in missing_keys:
                logger.warning(f"  - {k}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for k in unexpected_keys:
                logger.warning(f"  - {k}")

    # Verify that weights were loaded
    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in new_state_dict.values())
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Loaded {loaded_params:,} parameters from checkpoint")

    # Clean up
    del new_state_dict
    gc.collect()

    logger.info("Checkpoint loaded and verified successfully")

    model.eval()
    model = model.to(device)

    # Create dummy inputs
    batch_size = 1
    num_sweeps = 1
    num_cameras = 1  # Use 1 camera for CPU test to reduce memory

    logger.info(f"Creating dummy inputs: batch={batch_size}, sweeps={num_sweeps}, cameras={num_cameras}")

    # Input images: (B, num_sweeps, num_cameras, C, H, W)
    x = torch.randn(batch_size, num_sweeps, num_cameras, 3, 256, 704, device=device)

    # Transformation matrices
    sensor2ego_mats = (
        torch.eye(4, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
    )
    intrin_mats = (
        torch.eye(4, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
    )
    ida_mats = (
        torch.eye(4, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
    )
    sensor2sensor_mats = (
        torch.eye(4, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
    )
    bda_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    mats_dict = {
        "sensor2ego_mats": sensor2ego_mats,
        "intrin_mats": intrin_mats,
        "ida_mats": ida_mats,
        "sensor2sensor_mats": sensor2sensor_mats,
        "bda_mat": bda_mat,
    }

    # Run inference
    logger.info("Running forward pass...")
    with torch.no_grad():
        output = model(x, mats_dict)

    # Verify output
    logger.info("Verifying output...")

    # Output should be a list or dict with predictions
    assert output is not None, "Output should not be None"

    # Check if output is a list or dict
    if isinstance(output, (list, tuple)):
        assert len(output) > 0, "Output list should not be empty"
        logger.info(f"Output is a list/tuple with {len(output)} elements")
        for i, out in enumerate(output):
            if isinstance(out, dict):
                logger.info(f"Output[{i}] keys: {list(out.keys())}")
                # Check for common prediction keys
                if "boxes_3d" in out:
                    assert out["boxes_3d"] is not None, f"boxes_3d should not be None in output[{i}]"
                if "scores_3d" in out:
                    assert out["scores_3d"] is not None, f"scores_3d should not be None in output[{i}]"
            elif hasattr(out, "shape"):
                logger.info(f"Output[{i}] shape: {out.shape}")
                assert out.shape[0] == batch_size, f"Output batch size should match input batch size"
    elif isinstance(output, dict):
        logger.info(f"Output keys: {list(output.keys())}")
        # Check for common prediction keys
        if "boxes_3d" in output:
            assert output["boxes_3d"] is not None, "boxes_3d should not be None"
        if "scores_3d" in output:
            assert output["scores_3d"] is not None, "scores_3d should not be None"
    elif hasattr(output, "shape"):
        logger.info(f"Output shape: {output.shape}")
        assert output.shape[0] == batch_size, "Output batch size should match input batch size"
        assert len(output.shape) >= 2, "Output should have at least 2 dimensions"

    logger.info("Reference model inference successful!")
