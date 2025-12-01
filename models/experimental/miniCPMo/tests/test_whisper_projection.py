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
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from models.experimental.miniCPMo.tt.ttnn_audio_projector import TtnnAudioProjector


def create_attn_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        # import pdb
        # pdb.set_trace()
        parameters = {}

        if (
            hasattr(torch_model, "k_proj")
            and hasattr(torch_model, "v_proj")
            and hasattr(torch_model, "q_proj")
            and hasattr(torch_model, "out_proj")
        ):
            parameters["key"] = {}
            parameters["value"] = {}
            parameters["query"] = {}
            parameters["out_proj"] = {}

            parameters["key"]["weight"] = preprocess_linear_weight(torch_model.k_proj.weight, dtype=weight_dtype)
            parameters["value"]["weight"] = preprocess_linear_weight(torch_model.v_proj.weight, dtype=weight_dtype)
            parameters["query"]["weight"] = preprocess_linear_weight(torch_model.q_proj.weight, dtype=weight_dtype)
            parameters["out_proj"]["weight"] = preprocess_linear_weight(torch_model.out_proj.weight, dtype=weight_dtype)

            if torch_model.k_proj.bias is not None:
                parameters["key"]["bias"] = preprocess_linear_bias(torch_model.k_proj.bias, dtype=weight_dtype)
            else:
                parameters["key"]["bias"] = None
            if torch_model.v_proj.bias is not None:
                parameters["value"]["bias"] = preprocess_linear_bias(torch_model.v_proj.bias, dtype=weight_dtype)
            else:
                parameters["value"]["bias"] = None
            if torch_model.q_proj.bias is not None:
                parameters["query"]["bias"] = preprocess_linear_bias(torch_model.q_proj.bias, dtype=weight_dtype)
            else:
                parameters["query"]["bias"] = None
            if torch_model.out_proj.bias is not None:
                parameters["out_proj"]["bias"] = preprocess_linear_bias(torch_model.out_proj.bias, dtype=weight_dtype)
            else:
                parameters["out_proj"]["bias"] = None

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_ttnn_whisper_attn(device, input_dtype, weight_dtype):
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

    audio_states = torch.load("audio_states.pt")
    proj_layer = model.audio_projection_layer.eval()
    proj_output = proj_layer(audio_states)
    proj_output = proj_output.transpose(1, 2)
    proj_output = model.audio_avg_pooler(proj_output)

    ttnn_audio_projector = TtnnAudioProjector(
        device=device,
        input_dim=config.audio_config.encoder_ffn_dim // 4,
        output_dim=model.embed_dim,
        pool_step=config.audio_pool_step,
    )

    ttnn_audio_projector.load_weights(proj_layer.state_dict())
    tt_audio_states = ttnn.from_torch(
        audio_states, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_proj_output = ttnn_audio_projector.forward(tt_audio_states)

    # Final output comparison
    ttnn_proj_output = tt2torch_tensor(ttnn_proj_output)

    ttnn_proj_output = ttnn_proj_output.transpose(2, 1)
    ttnn_proj_output = ttnn_proj_output.reshape(proj_output.shape)
    does_pass, pcc_message = check_with_pcc(ttnn_proj_output, proj_output, 0.98)
    logger.info(f"Final Output PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
