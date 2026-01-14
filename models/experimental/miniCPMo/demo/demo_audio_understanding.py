# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo: MiniCPM-o audio understanding with TT-accelerated LLM.

This demo demonstrates audio understanding:
1. Load model from local reference folder (downloaded from HuggingFace on first run)
2. Enable TT acceleration for LLM (main computational bottleneck)
3. Process audio input and generate text response

Model and weights are loaded from local reference folder.
TT implementations are used as drop-in replacements for accelerated inference.

Usage:
    python models/experimental/miniCPMo/demo/demo_audio_understanding.py
"""

import ttnn
import sys
import librosa
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration
from models.experimental.miniCPMo.tt.model_setup import (
    ensure_model_files,
    ensure_audio_assets,
    get_audio_asset_path,
    REFERENCE_DIR,
)


def main():
    """
    Run audio understanding with TT-accelerated LLM.
    """
    torch.manual_seed(42)

    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        # Ensure model files and audio assets are downloaded
        ensure_model_files()
        ensure_audio_assets()

        logger.info(f"Loading model from local reference: {REFERENCE_DIR}")

        model = AutoModel.from_pretrained(
            str(REFERENCE_DIR),
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=False,
        )
        model = model.eval()

        # This replaces model.llm with TT-accelerated DropInQwen2LLM
        # The LLM is the main computational bottleneck - this provides the biggest speedup
        logger.info("Enabling TT acceleration for LLM...")
        model = enable_tt_acceleration(model, device, components=["audio", "llm"], model_path=str(REFERENCE_DIR))

        tokenizer = AutoTokenizer.from_pretrained(str(REFERENCE_DIR), trust_remote_code=True)

        task_prompt = "Please listen to the audio snippet carefully and describe what you hear.\n"

        audio_file = get_audio_asset_path("audio_understanding.mp3")

        logger.info(f"Loading audio: {audio_file}")
        audio_input, _ = librosa.load(audio_file, sr=16000, mono=True)
        msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

        logger.info("Running model.chat (audio understanding with TT LLM)...")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,  # Greedy decoding for reproducibility
            max_new_tokens=20,  # Generate enough tokens for meaningful output
            use_tts_template=False,
            generate_audio=False,
        )

        logger.info(f"Chat result: {res}")

        if res is None or len(res) == 0:
            logger.error("Expected non-empty response")
            return 1

        # Check for meaningful output
        expected_keywords = ["sounds", "like", "park", "birds", "audio", "hear"]
        found_keywords = [kw for kw in expected_keywords if kw in res.lower()]
        logger.info(f"Found keywords: {found_keywords}")

        logger.info("✅ Demo completed successfully!")
        return 0

    finally:
        logger.info("Closing TT device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
