# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo: MiniCPM-o vision understanding.

This demo demonstrates vision understanding:
1. Load model from local reference folder (downloaded from HuggingFace on first run)
2. Enable TT acceleration for vision encoder
3. Process image input and generate text response

Model and weights are loaded from local reference folder.
TT implementations are used as drop-in replacements for accelerated inference.

Usage:
    python models/experimental/miniCPMo/demo/demo_image.py
"""

import ttnn
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from loguru import logger

# Path to sample_data folder (relative to this script)
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "sample_data"

from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR


def main():
    """
    Run vision understanding with TT-accelerated vision encoder.
    """
    torch.manual_seed(42)

    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        # Ensure model files are downloaded
        ensure_model_files()

        logger.info(f"Loading model from local reference: {REFERENCE_DIR}")

        model = AutoModel.from_pretrained(
            str(REFERENCE_DIR),
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        )
        model = model.eval()

        # This replaces model.vpm and model.llm with TT-accelerated versions
        logger.info("Enabling TT acceleration for vision encoder and LLM...")
        model = enable_tt_acceleration(model, device, components=["vision", "llm"], model_path=str(REFERENCE_DIR))

        tokenizer = AutoTokenizer.from_pretrained(str(REFERENCE_DIR), trust_remote_code=True)

        image_file = SAMPLE_DATA_DIR / "huggingface_cat_image.jpg"
        question = "What is in the image?"

        if not image_file.exists():
            logger.error(f"Image file {image_file} not found")
            raise FileNotFoundError(f"Required image file not found: {image_file}")

        logger.info(f"Loading image: {image_file}")
        image = Image.open(image_file).convert("RGB")
        msgs = [{"role": "user", "content": [image, question]}]

        logger.info("Running model.chat (vision understanding)...")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=128,
        )

        logger.info(f"Chat result: {res}")

        if res is None or len(res) == 0:
            logger.error("Expected non-empty response")
            return 1

        logger.info("✅ Demo completed successfully!")
        return 0

    finally:
        logger.info("Closing TT device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
