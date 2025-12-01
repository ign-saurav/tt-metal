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

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=False,
        generate_audio=False,
        # temperature=0.3,
        # output_audio_path='result_audio_understanding.wav',
    )
    print(res)

    # apm = model.apm
    # apm_state_dict = apm.state_dict()
    # print(f"APM state dict has {len(apm_state_dict)} parameters")
    # embeddings_model = model.vpm.embeddings
    # emb_parameters = preprocess_model_parameters(
    #     initialize_model=lambda: embeddings_model,
    #     custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, weight_dtype),
    #     device=device,
    # )
    # patch_size = embeddings_model.patch_size
    # num_patches_per_side = embeddings_model.num_patches_per_side

    # resampler = model.resampler
    # resampler_parameters = preprocess_model_parameters(
    #     initialize_model=lambda: resampler,
    #     custom_preprocessor=create_resampler_preprocessor(device, weight_dtype),
    #     device=device,
    # )

    # parameters = {
    #     "embeddings": emb_parameters,
    #     "resampler": resampler_parameters,
    # }
    # with init_empty_weights():
    #     config._name_or_path = "models/experimental/miniCPMo/reference"
    #     tt_model = TTMiniCPMO(config, device, parameters, vpm_state_dict, patch_size, num_patches_per_side).eval()

    # load_checkpoint_and_dispatch(
    #     tt_model,
    #     local_checkpoint_path,
    #     device_map="auto",
    #     dtype=torch.bfloat16,
    # )
    # tt_res = tt_model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    # print(tt_res)
