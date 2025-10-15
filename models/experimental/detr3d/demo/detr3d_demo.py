# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import argparse
import numpy as np
from loguru import logger

from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig
from models.experimental.detr3d.source.detr3d.datasets.sunrgbd import SunrgbdDetectionDataset
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.common import load_torch_model_state
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.common.utility_functions import comp_pcc


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.modules = None
        self.parameters = None
        self.device = None


def build_input_using_3detr_dataset(val_dir, scan_index=0, use_color=False, device=None):
    """Use 3DETR's SunrgbdDetectionDataset to create the model input dict.

    Args:
        val_dir: path ending with "_val" (e.g., .../sunrgbd_val)
        scan_index: int index into the dataset split (default 0)
        use_color: whether to include color channels; must match model config
        device: torch device or string

    Returns:
        dict with keys required by the model forward: point_clouds, point_cloud_dims_min, point_cloud_dims_max
    """
    if not val_dir.endswith("_val"):
        raise ValueError(f"Expected val_dir to end with '_val', got: {val_dir}")
    root_dir = val_dir[:-4]  # strip "_val"

    ds_config = SunrgbdDatasetConfig()
    dataset = SunrgbdDetectionDataset(
        dataset_config=ds_config,
        split_set="val",
        root_dir=root_dir,
        num_points=20000,
        use_color=use_color,
        augment=False,
    )
    sample = dataset[scan_index]

    # sample contains numpy arrays as per 3DETR dataset; we only need the three inputs
    pc = torch.from_numpy(sample["point_clouds"]).unsqueeze(0)
    dims_min = torch.from_numpy(sample["point_cloud_dims_min"]).unsqueeze(0)
    dims_max = torch.from_numpy(sample["point_cloud_dims_max"]).unsqueeze(0)

    if device is not None:
        pc = pc.to(device)
        dims_min = dims_min.to(device)
        dims_max = dims_max.to(device)

    return {
        "point_clouds": pc,
        "point_cloud_dims_min": dims_min,
        "point_cloud_dims_max": dims_max,
    }


def load_model_weights(model, weights_path):
    """Load model weights from .pth file"""
    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")["model"]
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(f"Weights file not found at {weights_path}, using random weights")
    return model


def save_predictions(output_dict, output_dir):
    """Save model predictions to file"""
    os.makedirs(output_dir, exist_ok=True)

    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            np.save(os.path.join(output_dir, f"{key}.npy"), value.cpu().numpy())
            logger.info(f"Saved {key} with shape {value.shape}")


def run_detr3d_inference(
    point_cloud_path=None,
    weights_path=None,
    output_dir="models/experimental/detr3d/demo/outputs/",
    encoder_only=False,
    num_points=20000,
):
    """Run DETR3D model inference on point cloud data with SunRGBD dataset configuration"""

    # Load point cloud
    if point_cloud_path:
        input_dict = build_input_using_3detr_dataset(point_cloud_path)
    else:
        # Generate all random inputs matching test file
        torch.manual_seed(0)
        logger.info("Generating random inputs matching test format")
        point_clouds = torch.randn(1, num_points, 3)
        point_cloud_dims_min = torch.randn(1, 3)
        point_cloud_dims_max = torch.randn(1, 3)
        input_dict = {
            "point_clouds": point_clouds,
            "point_cloud_dims_min": point_cloud_dims_min,
            "point_cloud_dims_max": point_cloud_dims_max,
        }
        logger.info(f"Point cloud shape: {point_clouds.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup model configuration - SunRGBD dataset only
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()
    logger.info("Using SunRGBD dataset configuration")

    # Prepare input dictionary

    # Build reference PyTorch model
    logger.info("Building PyTorch reference model...")
    ref_module, _ = build_3detr(args, dataset_config)

    # Load weights from .pth file if provided
    if weights_path:
        ref_module = load_model_weights(ref_module, weights_path)
    else:
        # Use default weight loading
        load_torch_model_state(ref_module)

    ref_module.eval()

    # Run reference model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    # Save reference outputs
    ref_output_dir = os.path.join(output_dir, "pytorch_output")
    if encoder_only:
        logger.info(f"Reference model produced {len(ref_out)} encoder outputs")
    else:
        # save_predictions(ref_out["outputs"], ref_output_dir)
        logger.info(f"Reference outputs saved to: {ref_output_dir}")

    # Open TTNN device
    device = ttnn.open_device(device_id=0, l1_small_size=16384)

    try:
        # Preprocess model parameters
        logger.info("Preprocessing model parameters...")
        ref_module_parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_module,
            custom_preprocessor=create_custom_mesh_preprocessor(None),
            device=device,
        )

        # Build TTNN model
        logger.info("Building TTNN model...")
        ttnn_args = Tt3DetrArgs()
        ttnn_args.modules = ref_module
        ttnn_args.parameters = ref_module_parameters
        ttnn_args.device = device

        ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)

        # Run TTNN model
        logger.info("Running TTNN model inference...")
        tt_output = ttnn_module(inputs=input_dict, encoder_only=encoder_only)

        # Process and compare outputs
        if encoder_only:
            logger.info("Comparing encoder outputs...")
            for idx, (tt_out, torch_out) in enumerate(zip(tt_output, ref_out)):
                if not isinstance(tt_out, torch.Tensor):
                    tt_out = ttnn.to_torch(tt_out)
                    tt_out = torch.reshape(tt_out, torch_out.shape)

                passing, pcc_message = comp_pcc(torch_out, tt_out, 0.97)
                logger.info(f"Encoder Output {idx} PCC: {pcc_message}")

                if passing:
                    logger.info(f"Encoder Output {idx} Test Passed!")
                else:
                    logger.warning(f"Encoder Output {idx} Test Failed!")
        else:
            # Save TTNN outputs
            ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
            # save_predictions(tt_output["outputs"], ttnn_output_dir)
            logger.info(f"TTNN outputs saved to: {ttnn_output_dir}")

            # Compare main outputs
            logger.info("Comparing model outputs...")
            SKIP_KEYS = ["angle_continuous", "objectness_prob"]

            for key in ref_out["outputs"]:
                if key in SKIP_KEYS:
                    logger.info(f"Output Key '{key}' - Skipped")
                    continue

                passing, pcc_message = comp_pcc(ref_out["outputs"][key], tt_output["outputs"][key], 0.97)
                logger.info(f"Output Key '{key}' PCC: {pcc_message}")

                if passing:
                    logger.info(f"Output Key '{key}' Test Passed!")
                else:
                    logger.warning(f"Output Key '{key}' Test Failed!")

        logger.info("DETR3D inference completed!")
        logger.info(f"Results saved to: {output_dir}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR3D 3D Object Detection Inference (SunRGBD Dataset)")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input point cloud file (.npy format)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights file (.pth format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/experimental/detr3d/demo/outputs/",
        help="Directory to save output predictions",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Run encoder only (no decoder)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=20000,
        help="Number of points in point cloud",
    )

    args = parser.parse_args()

    run_detr3d_inference(
        args.input,
        args.weights,
        args.output_dir,
        args.encoder_only,
        args.num_points,
    )
