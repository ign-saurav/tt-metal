import ttnn
import pytest
import torch
import shutil
import os
import sys
from transformers import AutoModel


class AttrDict(dict):
    """Dict that allows attribute access"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o_tts_only(device, input_dtype, weight_dtype):
    """
    Test TTS audio generation using local reference model.

    IMPORTANT: Must use AutoModel.from_pretrained (not load_checkpoint_and_dispatch)
    because load_checkpoint_and_dispatch causes numerical precision issues that
    make the TTS model produce garbage audio.
    """
    torch.manual_seed(42)

    # Clear HuggingFace cache to ensure local reference is used
    cache_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/reference")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    # Also clear Python's module cache for any cached transformers_modules
    modules_to_remove = [key for key in sys.modules if "transformers_modules.reference" in key]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Load model using AutoModel.from_pretrained with LOCAL path
    local_model_path = "models/experimental/miniCPMo/reference"
    model = AutoModel.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,
        init_tts=True,
        local_files_only=True,
    )

    # Initialize TTS components
    model.init_tts()
    model.tts.float()  # DVAE/vocos need float32 for stability
    model = model.eval()
    model.init_tt_device(device)

    # Load saved inputs/outputs from gen_audio.py run
    inputs = AttrDict(torch.load("/home/ubuntu/_generate_mel_spec_inputs.pt"))
    outputs = AttrDict(torch.load("/home/ubuntu/_generate_mel_spec_outputs.pt"))
    answer = torch.load("/home/ubuntu/_generate_mel_spec_answer.pt")

    # Generate mel spectrogram and audio (TT prefill + PyTorch decode)
    mel_spec = model._generate_mel_spec(inputs, outputs, answer)
    wav, sr = model.decode_mel_to_audio(mel_spec, output_path="result_audio_tts_test.wav")
