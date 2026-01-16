# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.request
import torch
import ttnn
from loguru import logger
from models.experimental.BevDepth.reference.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from loguru import logger
from models.experimental.BevDepth.tt.ttnn_bevdepth import TtBEVDepth


def download_bevdepth_weights():
    """Download BEVDepth pretrained weights"""
    url = "https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
    weights_path = "models/experimental/BevDepth/resources/bevdepth_weights.pth"

    if not os.path.exists(weights_path):
        logger.info(f"Downloading weights from {url}")
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Downloaded weights to {weights_path}")

    return weights_path


def load_reference_model():
    """Load the reference BEVDepth model."""

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
    """Run Torch inference on BEVDepth model."""
    if model is None:
        model = load_reference_model()
        if model is None:
            logger.error("Failed to load reference model")
            return None

    logger.info("Running Torch inference...")
    with torch.no_grad():
        preds = model.model(imgs, mats_dict)
    return preds


def run_ttnn_inference(device, params, imgs, mats_dict):
    """Run TTNN inference on BEVDepth model."""
    logger.info("Running TTNN inference...")

    _, _, _, _, img_h, img_w = imgs.shape
    logger.info(f"TTNN input image size: {img_h}x{img_w}")

    lss_conf = {
        "x_bound": [-51.2, 51.2, 0.8],
        "y_bound": [-51.2, 51.2, 0.8],
        "z_bound": [-5.0, 3.0, 0.2],
        "d_bound": [2.0, 58.0, 0.5],
        "final_dim": [img_h, img_w],
        "downsample_factor": 16,
        "output_channels": 80,
    }

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "batch_size": 1,
        "neck_in_channels": [256, 512, 1024, 2048],
        "neck_out_channels": [128, 128, 128, 128],
        "neck_upsample_strides": [0.25, 0.5, 1, 2],
        "depthnet_in_channels": 512,
        "depthnet_mid_channels": 512,
        "depthnet_context_channels": 80,
        "depthnet_depth_channels": 112,
    }

    model = TtBEVDepth(
        device=device,
        backbone_parameters=params["backbone"],
        neck_parameters=params["neck"],
        depthnet_parameters=params["depthnet"],
        head_parameters=params["head"],
        lss_conf=lss_conf,
        model_config=model_config,
    )

    ttnn_output = model(imgs, mats_dict)

    output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
    torch_preds = []

    for task_idx in range(len(ttnn_output)):
        task_dict = {}
        for key in output_keys:
            ttnn_tensor, shape = ttnn_output[task_idx][key]
            tensor_torch = ttnn.to_torch(ttnn_tensor)

            if len(tensor_torch.shape) == 4:
                tensor_torch = tensor_torch.permute(0, 3, 1, 2).contiguous()

            task_dict[key] = tensor_torch
        torch_preds.append([task_dict])

    return torch_preds
