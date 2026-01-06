# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo: MiniCPM-o audio mimicking.

This demo demonstrates the mimick task which reflects the model's end-to-end
speech modeling capability. The model takes audio input and outputs an ASR
transcription, then reconstructs the original audio with high similarity.

Model and weights are loaded from HuggingFace URL 'openbmb/MiniCPM-o-2_6'.
TT implementations are used as drop-in replacements for accelerated inference.

Usage:
    python models/experimental/miniCPMo/demo/test_mini_cpm_o_audio_tts.py
"""

import ttnn
import os
import sys
import librosa
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration


def main():
    """
    Run mimick task with TT-accelerated TTS decoder.
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
            init_tts=True,
        )

        # Initialize TTS components (vocos vocoder, TTS tokenizer)
        logger.info("Initializing TTS components...")
        model.init_tts()
        model.tts.float()  # DVAE/vocos need float32 for stability
        model = model.eval()

        logger.info("Enabling TT acceleration for TTS decoder...")
        model = enable_tt_acceleration(model, device, components=["tts", "llm"])

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        audio_file = hf_hub_download(
            repo_id="openbmb/MiniCPM-o-2_6", filename="assets/input_examples/Trump_WEF_2018_10s.mp3", repo_type="model"
        )

        logger.info(f"Loading audio to mimick")
        audio_input, _ = librosa.load(audio_file, sr=16000, mono=True)

        mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
        msgs = [{"role": "user", "content": [mimick_prompt, audio_input]}]

        output_audio_path = "result_mimick.wav"

        logger.info("Running model.chat with mimick prompt (TT-accelerated TTS)...")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            temperature=0.3,
            generate_audio=True,
            output_audio_path=output_audio_path,
        )

        logger.info(f"Chat result (transcription): {res}")

        if os.path.exists(output_audio_path):
            file_size = os.path.getsize(output_audio_path)
            logger.info(f"✓ Output audio saved: {output_audio_path} ({file_size} bytes)")
            if file_size < 1000:
                logger.error(f"Audio file too small ({file_size} bytes), likely empty or corrupted")
                return 1
        else:
            logger.error(f"✗ Output audio not found: {output_audio_path}")
            return 1

        logger.info("✅ Demo completed successfully!")
        return 0

    finally:
        logger.info("Closing TT device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
