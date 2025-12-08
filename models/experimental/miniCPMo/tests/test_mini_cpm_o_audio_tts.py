import ttnn
import pytest
import json
import librosa
import torch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o(device, input_dtype, weight_dtype):
    # Load config directly from local JSON file
    config_path = "models/experimental/miniCPMo/reference/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = MiniCPMOConfig.from_dict(
        config_dict,
        init_vision=False,
        init_audio=True,
        init_tts=True,
        generate_audio=True,
        use_tts_template=True,
        temperature=0.3,
        output_audio_path="result_audio_understanding.wav",
    )

    print("Initializing MiniCPM-o model...")
    # Initialize the model directly with the config
    # with torch.device("meta"):
    with init_empty_weights():
        model = MiniCPMO(config)
    model.init_tts()
    model.tts.float()

    # Move model from meta device to CPU before loading checkpoint
    # This is required because load_checkpoint_and_dispatch needs real tensors
    model = model.to_empty(device="cpu")

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

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=False,
        generate_audio=True,
        temperature=0.3,
        output_audio_path="result_audio_understanding.wav",
    )
    print("RESULT: ")
    print(res)

    # proj_layer_state_dict = model.audio_projection_layer.state_dict()
    # apm_state_dict = model.apm.state_dict()

    # # Extract TTS weights from the loaded model
    # tts_state_dict = model.tts.state_dict() if hasattr(model.tts, "state_dict") else {}

    # state_dict = {
    #     "apm": apm_state_dict,
    #     "audio_projection_layer": proj_layer_state_dict,
    #     "tts": tts_state_dict,
    # }

    # with init_empty_weights():
    #     config._name_or_path = "models/experimental/miniCPMo/reference"
    #     tt_model = TTMiniCPMO(config, state_dict=state_dict, tt_device=device).eval()

    # # Move model from meta device to CPU before initializing TTS and loading checkpoint
    # tt_model = tt_model.to_empty(device="cpu")
    # tt_model.init_tts()

    # load_checkpoint_and_dispatch(
    #     tt_model,
    #     local_checkpoint_path,
    #     device_map="auto",
    #     dtype=torch.bfloat16,
    # )
    # tt_res = tt_model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer,
    #     sampling=True,
    #     max_new_tokens=128,
    #     use_tts_template=True,
    #     generate_audio=True,
    #     temperature=0.3,
    #     output_audio_path="result_audio_understanding.wav",
    # )

    # print(tt_res)
