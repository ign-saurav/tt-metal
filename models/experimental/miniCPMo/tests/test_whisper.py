import ttnn
import pytest
import json
import librosa
import torch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from models.experimental.minicpm_o_2_6.tt.ttnn_whisper_encoder import TtnnWhisperEncoder


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_ttnn_whisper_encoder(device, input_dtype, weight_dtype):
    # Load config directly from local JSON file
    config_path = "models/experimental/miniCPMo/reference/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = MiniCPMOConfig.from_dict(
        config_dict,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )

    print("Initializing MiniCPM-o model...")
    # Initialize the model directly with the config
    # with torch.device("meta"):
    with init_empty_weights():
        model = MiniCPMO(config)

    # local_checkpoint_path = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/509805e84db1c84f154034d71a21c4f2331e6e11"
    local_checkpoint_path = "models/experimental/miniCPMo/reference/safetensors"
    load_checkpoint_and_dispatch(
        model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    # Set model to eval mode
    model = model.eval()

    # Load tokenizer directly from local reference folder files
    tokenizer_path = "models/experimental/miniCPMo/reference"
    tokenizer = MiniCPMOTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")

    task_prompt = (
        "Please listen to the audio snippet carefully and transcribe the content." + "\n"
    )  # can change to other prompts.
    audio_input, _ = librosa.load("audio_understanding.mp3", sr=16000, mono=True)  # load the audio to be captioned

    msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

    # res = model.chat(
    #     msgs=msgs,
    #     tokenizer=tokenizer,
    #     sampling=True,
    #     max_new_tokens=128,
    #     use_tts_template=False,
    #     generate_audio=False,
    #     # temperature=0.3,
    #     # output_audio_path='result_audio_understanding.wav',
    # )
    # print(res)

    apm = model.apm
    apm_state_dict = apm.state_dict()
    print(f"APM state dict has {len(apm_state_dict)} parameters")

    wavforms = torch.load("wavforms.pt")
    audio_attention_mask = torch.load("audio_attention_mask.pt")
    apm.audio_encoder_layer = -1
    audio_states = apm(wavforms, output_hidden_states=True, attention_mask=audio_attention_mask).hidden_states[
        apm.audio_encoder_layer
    ]
    print(audio_states.shape)

    ttnn_model = TtnnWhisperEncoder(
        mesh_device=device,
        config=config.audio_config.to_dict(),
    )

    # Load weights into TTNN model
    ttnn_model.load_weights(apm_state_dict)

    # TTNN forward pass using adapted MiniCPM functions
    ttnn_output = ttnn_model.forward(wavforms, attention_mask=audio_attention_mask)

    ttnn_output = tt2torch_tensor(ttnn_output)

    ttnn_output = ttnn_output.reshape(audio_states.shape)
    does_pass, pcc_message = check_with_pcc(ttnn_output, audio_states, 0.98)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
