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
import os
import sys
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from loguru import logger

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

        # This replaces model.vpm with TT-accelerated DropInVisionEncoder
        logger.info("Enabling TT acceleration for vision encoder...")
        model = enable_tt_acceleration(model, device, components=["vision", "llm"])

        tokenizer = AutoTokenizer.from_pretrained(str(REFERENCE_DIR), trust_remote_code=True)

        image_file = "cat_img.jpg"
        question = "What is in the image?"

        if not os.path.exists(image_file):
            logger.warning(f"Image file {image_file} not found, using text-only prompt")
            msgs = [{"role": "user", "content": "Describe a typical house cat."}]
        else:
            logger.info(f"Loading image: {image_file}")
            image = Image.open(image_file).convert("RGB")
            msgs = [{"role": "user", "content": [image, question]}]

        logger.info("Running model.chat (vision understanding)...")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
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
