# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
from loguru import logger
from typing import Dict, Optional


# =============================================================================
# Model Configuration Constants
# =============================================================================

VOXEL_SIZE = [0.16, 0.16, 4]
POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
MAX_NUM_POINTS = 32
MAX_VOXELS = (16000, 40000)
NCLASSES = 3

# Label mapping for KITTI dataset
LABEL2CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}

# Default checkpoint filename
DEFAULT_CHECKPOINT = "epoch_160.pth"


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================


def load_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Optional[Dict]:
    """
    Load checkpoint and extract state dict.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        State dict from checkpoint, or None if file not found.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return state_dict
    except FileNotFoundError:
        logger.warning(f"Checkpoint file '{checkpoint_path}' not found, using random weights")
        return None


def extract_component_state_dict(state_dict: Dict, prefix: str) -> Dict:
    """
    Extract component-specific weights from full model state dict.

    Args:
        state_dict: Full model state dict.
        prefix: Component prefix (e.g., "backbone.", "neck.", "head.").

    Returns:
        State dict containing only the component's weights.
    """
    component_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key.replace(prefix, "")
            component_state_dict[new_key] = value
    return component_state_dict


# =============================================================================
# Download Utilities
# =============================================================================


def download_test_data(resources_dir: str) -> None:
    """
    Download sample test data from PointPillars repository.

    Args:
        resources_dir: Directory to save downloaded files.
    """
    import subprocess
    import tempfile
    import shutil

    os.makedirs(resources_dir, exist_ok=True)
    test_file = os.path.join(resources_dir, "000002.bin")

    if os.path.exists(test_file):
        logger.info("Test data already exists, skipping download")
        return

    logger.info("Downloading test data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/zhulf0804/PointPillars.git",
                tmpdir,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", tmpdir, "sparse-checkout", "set", "pointpillars/dataset/demo_data/test"],
            check=True,
            capture_output=True,
        )
        src_dir = os.path.join(tmpdir, "pointpillars/dataset/demo_data/test")
        for filename in ["000002.bin", "000002.txt", "000002.png"]:
            src = os.path.join(src_dir, filename)
            dst = os.path.join(resources_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
    logger.info(f"Test data downloaded to {resources_dir}")


def download_checkpoint(checkpoint_dir: str) -> str:
    """
    Download pretrained checkpoint from PointPillars repository.

    Args:
        checkpoint_dir: Directory to save checkpoint.

    Returns:
        Path to the downloaded checkpoint file.
    """
    import requests

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, DEFAULT_CHECKPOINT)

    if os.path.exists(checkpoint_path):
        logger.info("Checkpoint already exists, skipping download")
        return checkpoint_path

    logger.info("Downloading checkpoint...")
    url = "https://github.com/zhulf0804/PointPillars/raw/main/pretrained/epoch_160.pth"
    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Checkpoint downloaded to {checkpoint_path}")
    return checkpoint_path


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================


def multi_device_to_torch(tt_tensor: ttnn.Tensor, device) -> torch.Tensor:
    """
    Convert ttnn tensor to torch, handling multi-device case.

    Args:
        tt_tensor: TTNN tensor to convert.
        device: TTNN device (single or mesh).

    Returns:
        PyTorch tensor.
    """
    num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    if num_devices > 1:
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        result = tt_output.to_torch(mesh_composer=mesh_composer)
        return result[: tt_output.shape[0]]
    return tt_output.to_torch()


# =============================================================================
# Model Creation Utilities
# =============================================================================


def create_pointpillars_model(
    nclasses: int = NCLASSES,
    voxel_size: list = None,
    point_cloud_range: list = None,
    max_num_points: int = MAX_NUM_POINTS,
    max_voxels: tuple = MAX_VOXELS,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Create and initialize PointPillars PyTorch model.

    Args:
        nclasses: Number of detection classes.
        voxel_size: Voxel dimensions [x, y, z].
        point_cloud_range: Point cloud boundaries [x_min, y_min, z_min, x_max, y_max, z_max].
        max_num_points: Maximum points per pillar.
        max_voxels: Maximum voxels (train, test).
        checkpoint_path: Path to checkpoint file.
        dtype: Model data type.

    Returns:
        Initialized PointPillars model in eval mode.
    """
    from models.experimental.pointpillars.reference.model.pointpillars import PointPillars

    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    if point_cloud_range is None:
        point_cloud_range = POINT_CLOUD_RANGE

    model = PointPillars(
        nclasses=nclasses,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
    )

    state_dict = load_checkpoint(checkpoint_path)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    model = model.to(dtype=dtype)
    model.eval()
    return model


def get_model_config() -> Dict:
    """
    Get default model configuration as a dictionary.

    Returns:
        Dictionary with model configuration parameters.
    """
    return {
        "nclasses": NCLASSES,
        "voxel_size": VOXEL_SIZE,
        "point_cloud_range": POINT_CLOUD_RANGE,
        "max_num_points": MAX_NUM_POINTS,
        "max_voxels": MAX_VOXELS,
    }
