import ttnn
import pytest
import os
import sys
import shutil
import librosa
import torch
from transformers import AutoModel
from loguru import logger

from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o_audio_tts(device, input_dtype, weight_dtype):
    """
    Test audio understanding with TTS response using TT decoder.

    This test:
    1. Loads audio input for understanding
    2. Processes it through the model
    3. Generates TTS audio response using TT decoder (prefill + decode)
    4. Saves output audio file

    Uses AutoModel.from_pretrained (not load_checkpoint_and_dispatch)
    because load_checkpoint_and_dispatch causes numerical precision issues.
    """
    torch.manual_seed(42)

    # Clear HuggingFace cache to ensure local reference is used
    cache_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/reference")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    # Clear Python's module cache for any cached transformers_modules
    modules_to_remove = [key for key in sys.modules if "transformers_modules.reference" in key]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Load model using AutoModel.from_pretrained with LOCAL path
    local_model_path = "models/experimental/miniCPMo/reference"
    logger.info("Loading model with AutoModel.from_pretrained...")
    model = AutoModel.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,  # Enable audio understanding
        init_tts=True,  # Enable TTS
        local_files_only=True,
    )

    # Initialize TTS components
    logger.info("Initializing TTS components...")
    model.init_tts()
    model.tts.float()  # DVAE/vocos need float32 for stability
    model = model.eval()

    # Initialize TT device for TT decoder (prefill + decode)
    logger.info("Initializing TT device for decoder...")
    model.init_tt_device(device)

    # Load tokenizer directly from local reference folder
    tokenizer_path = "models/experimental/miniCPMo/reference"
    tokenizer = MiniCPMOTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")

    # Prepare task prompt and audio input
    task_prompt = "Please listen to the audio snippet carefully and transcribe the content.\n"

    # Check if audio file exists
    audio_file = "audio_understanding.mp3"
    if not os.path.exists(audio_file):
        logger.warning(f"Audio file {audio_file} not found, using a simple text prompt instead")
        msgs = [{"role": "user", "content": "Say hello world"}]
    else:
        audio_input, _ = librosa.load(audio_file, sr=16000, mono=True)
        msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

    output_audio_path = "result_audio_understanding_mini_cpmo_audio_tts.wav"

    # Run chat with TTS enabled - this uses TT decoder for mel spec generation
    logger.info("Running model.chat with TTS (using TT decoder)...")
    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=True,
        generate_audio=True,
        temperature=0.3,
        output_audio_path=output_audio_path,
    )

    logger.info(f"Chat result: {res}")

    # Verify audio was generated
    if os.path.exists(output_audio_path):
        file_size = os.path.getsize(output_audio_path)
        logger.info(f"✓ Output audio saved: {output_audio_path} ({file_size} bytes)")
        assert file_size > 1000, f"Audio file too small ({file_size} bytes), likely empty or corrupted"
    else:
        logger.error(f"✗ Output audio not found: {output_audio_path}")
        assert False, f"Expected output audio file not created: {output_audio_path}"

    logger.info("Test passed!")
