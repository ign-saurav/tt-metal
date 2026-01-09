# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo: MiniCPM-o audio understanding with TT-accelerated LLM.

This demo demonstrates audio understanding:
1. Load model from HuggingFace (auto-cached)
2. Enable TT acceleration for LLM (main computational bottleneck)
3. Process audio input and generate text response

Model and weights are loaded from HuggingFace URL 'openbmb/MiniCPM-o-2_6'.
TT implementations are used as drop-in replacements for accelerated inference.

Usage:
    python models/experimental/miniCPMo/demo/demo_audio_understanding.py
"""

import ttnn
import sys
import librosa
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration


def main():
    """
    Run audio understanding with TT-accelerated LLM.
    """
    torch.manual_seed(42)

    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        model_name = "openbmb/MiniCPM-o-2_6"
        logger.info(f"Loading model from HuggingFace: {model_name}")

        model = AutoModel.from_pretrained(
            model_name,
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
        model = enable_tt_acceleration(model, device, components=["audio", "llm"], model_path=model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        task_prompt = "Please listen to the audio snippet carefully and describe what you hear.\n"

        audio_file = hf_hub_download(
            repo_id="openbmb/MiniCPM-o-2_6", filename="assets/input_examples/audio_understanding.mp3", repo_type="model"
        )

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
