import ttnn
import pytest
import torch
import logging
from transformers import AutoModel

# Enable debug logging to see EOS detection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o_tts_only(device, input_dtype, weight_dtype):
    # Set seed for reproducibility - must match gen_audio.py
    torch.manual_seed(42)

    # Local path to the model - no network access needed
    local_model_path = "models/experimental/miniCPMo/reference"

    # Load model using AutoModel.from_pretrained with LOCAL path
    # This loads from local files without any network access
    model = AutoModel.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,  # skip audio input
        init_tts=True,  # only TTS
        local_files_only=True,  # Ensure no network access
    )

    # Initialize TTS components (tokenizer, vocos)
    model.init_tts()

    # Convert TTS to float32 for stability (matches gen_audio.py behavior)
    # DVAE and vocos work better with float32 precision
    model.tts.float()

    model = model.eval()

    # Load inputs, outputs, and answer from saved files (matching HuggingFace pattern)
    # These files should be created by running the model with generate_audio=True
    inputs = torch.load("/home/ubuntu/_generate_mel_spec_inputs.pt")
    outputs = torch.load("/home/ubuntu/_generate_mel_spec_outputs.pt")
    answer = torch.load("/home/ubuntu/_generate_mel_spec_answer.pt")

    print(inputs.keys())
    print(outputs.keys())
    inputs = AttrDict(inputs)
    outputs = AttrDict(outputs)

    # Now call _generate_mel_spec with loaded inputs, outputs, and answer (matching HuggingFace pattern)
    # This handles all the TTS generation internally
    mel_spec = model._generate_mel_spec(inputs, outputs, answer)

    # Convert mel spectrograms to audio waveform
    wav, sr = model.decode_mel_to_audio(mel_spec, output_path="result_audio_tts_test.wav")

    print(f"TTS audio generated successfully! Audio shape: {wav.shape}, Sample rate: {sr}")
