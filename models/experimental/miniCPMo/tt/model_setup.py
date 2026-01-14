# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM-o Model Setup Utilities.

Handles downloading required model files to the local reference folder
and audio assets to the local assets folder.
This ensures tests and demos work without needing to commit large files.

Usage:
    from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, ensure_audio_assets, REFERENCE_DIR, ASSETS_DIR

    ensure_model_files()  # Downloads missing model files
    ensure_audio_assets()  # Downloads missing audio assets
    model = AutoModel.from_pretrained(str(REFERENCE_DIR), trust_remote_code=True, ...)
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Path to local reference folder (contains patched Python files)
REFERENCE_DIR = Path(__file__).parent.parent / "reference"

# Path to local assets folder (audio files for demos)
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Files that need to be downloaded (large files not committed to git)
REQUIRED_FILES = [
    # Model weights (17GB total)
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    # Model index file
    "model.safetensors.index.json",
    # Tokenizer files (~10MB total)
    "tokenizer.json",
    "vocab.json",
]

# Audio assets for demos (downloaded from HuggingFace assets folder)
AUDIO_ASSETS = [
    "assets/input_examples/audio_understanding.mp3",
    "assets/input_examples/Trump_WEF_2018_10s.mp3",
]

# HuggingFace repo ID
HF_REPO_ID = "openbmb/MiniCPM-o-2_6"


def ensure_model_files():
    """
    Download required model files to reference folder if not present.

    This downloads safetensors and tokenizer files that are too large
    to commit to git. The local reference folder contains patched
    Python files that are committed.
    """
    for filename in REQUIRED_FILES:
        local_path = REFERENCE_DIR / filename

        # Skip if file exists and is not a stub (LFS stubs are ~135 bytes)
        if local_path.exists() and local_path.stat().st_size > 1000:
            continue

        logger.info(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(REFERENCE_DIR),
            local_dir_use_symlinks=False,
        )
        logger.info(f"  Downloaded {filename}")

    logger.info("All required model files are present")


def ensure_audio_assets():
    """
    Download audio assets to the local assets folder if not present.

    This downloads audio files used by demos from the HuggingFace repo
    to a local assets folder instead of the HuggingFace cache.
    """
    # Create assets directory if it doesn't exist
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for hf_path in AUDIO_ASSETS:
        # hf_path is like "assets/input_examples/audio_understanding.mp3"
        # We want to save to ASSETS_DIR/input_examples/audio_understanding.mp3
        relative_path = hf_path.replace("assets/", "", 1)
        local_path = ASSETS_DIR / relative_path

        # Skip if file exists and has content
        if local_path.exists() and local_path.stat().st_size > 1000:
            continue

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {hf_path}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            local_dir=str(ASSETS_DIR.parent),  # Download to miniCPMo folder
            local_dir_use_symlinks=False,
        )
        logger.info(f"  Downloaded to {local_path}")

    logger.info("All audio assets are present")


def get_audio_asset_path(asset_name: str) -> Path:
    """
    Get the local path for an audio asset.

    Args:
        asset_name: Name of the asset file (e.g., "audio_understanding.mp3")

    Returns:
        Path to the local asset file
    """
    return ASSETS_DIR / "input_examples" / asset_name


def get_reference_dir() -> Path:
    """Get path to the local reference directory."""
    return REFERENCE_DIR


def get_assets_dir() -> Path:
    """Get path to the local assets directory."""
    return ASSETS_DIR
