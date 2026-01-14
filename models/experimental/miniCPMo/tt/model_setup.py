# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM-o Model Setup Utilities.

Handles downloading required model files to the local reference folder.
This ensures tests and demos work without needing to commit large files.

Usage:
    from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR

    ensure_model_files()  # Downloads missing files
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

# Files that need to be downloaded (large files not committed to git)
REQUIRED_FILES = [
    # Model weights (17GB total)
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    # Tokenizer files (~10MB total)
    "tokenizer.json",
    "vocab.json",
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


def get_reference_dir() -> Path:
    """Get path to the local reference directory."""
    return REFERENCE_DIR
