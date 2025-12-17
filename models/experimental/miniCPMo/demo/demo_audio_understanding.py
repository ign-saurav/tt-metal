# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo: MiniCPM-o audio understanding (without TTS).

This demo demonstrates audio understanding:
1. Load model from HuggingFace (auto-cached)
2. Enable TT acceleration for audio encoder
3. Process audio input and generate text response

Model and weights are loaded from HuggingFace URL 'openbmb/MiniCPM-o-2_6'.
TT implementations are used as drop-in replacements for accelerated inference.

Usage:
    python models/experimental/miniCPMo/demo/test_mini_cpm_o_audio.py
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
    Run audio understanding with TT-accelerated audio encoder.
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

        # This replaces model.apm with TT-accelerated DropInAudioEncoder
        logger.info("Enabling TT acceleration for audio encoder...")
        model = enable_tt_acceleration(model, device, components=["audio"])

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        task_prompt = "Please listen to the audio snippet carefully and transcribe the content.\n"

        audio_file = hf_hub_download(
            repo_id="openbmb/MiniCPM-o-2_6", filename="assets/input_examples/audio_understanding.mp3", repo_type="model"
        )

        logger.info(f"Loading audio: {audio_file}")
        audio_input, _ = librosa.load(audio_file, sr=16000, mono=True)
        msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

        logger.info("Running model.chat (audio understanding, no TTS)...")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=False,
            generate_audio=False,
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
