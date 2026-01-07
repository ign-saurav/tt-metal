# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for BEVDepth tests and demos.
This module consolidates shared functions used across multiple test files.
"""

import os
import urllib.request
import torch
from loguru import logger


def download_bevdepth_weights():
    """Download BEVDepth pretrained weights"""
    url = "https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
    weights_path = "/tmp/bevdepth_weights.pth"

    if not os.path.exists(weights_path):
        logger.info(f"Downloading weights from {url}")
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Downloaded weights to {weights_path}")

    return weights_path


def load_reference_model():
    """Load the reference BEVDepth model."""
    from models.experimental.BevDepth.reference.bev_depth_lss_r50_256x704_128x128_24e_2key import (
        BEVDepthLightningModel,
    )

    logger.info("Loading reference BEVDepth model...")
    lightning_model = BEVDepthLightningModel()
    checkpoint_path = download_bevdepth_weights()

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return None

    lightning_model.load_checkpoint(checkpoint_path, verbose=False)
    lightning_model.model.eval()
    return lightning_model


def create_reference_inputs(batch_size=1, num_sweeps=2, num_cameras=6, img_h=256, img_w=704):
    """Create reference input images and transformation matrices."""
    imgs = torch.randn((batch_size, num_sweeps, num_cameras, 3, img_h, img_w), dtype=torch.float32, requires_grad=False)

    mats_dict = {
        "sensor2ego_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "intrin_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "ida_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "sensor2sensor_mats": torch.eye(4)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_sweeps, num_cameras, 1, 1),
        "bda_mat": torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
    }

    return imgs, mats_dict


def run_torch_inference(model=None, imgs=None, mats_dict=None):
    """
    Run Torch inference on BEVDepth model.
    Can optionally load the model if not provided.

    Args:
        model: Optional BEVDepth model. If None, will load the reference model automatically.
        imgs: Input images tensor. Required.
        mats_dict: Transformation matrices dictionary. Required.

    Returns:
        preds: Model predictions, or None if model loading failed
    """
    if model is None:
        model = load_reference_model()
        if model is None:
            logger.error("Failed to load reference model")
            return None

    logger.info("Running Torch inference...")
    with torch.no_grad():
        preds = model.model(imgs, mats_dict)
    return preds
