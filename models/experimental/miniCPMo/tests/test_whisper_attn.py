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
from models.experimental.miniCPMo.tt.ttnn_whisper_encoder import whisper_attention_minicpm
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


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

    hidden_states = torch.load("encoder_layer_0_after_layernorm.pt")
    attention_mask = torch.load("encoder_layer_0_attention_mask.pt")
    layer_head_mask = torch.load("encoder_layer_0_layer_head_mask.pt")
    output_attentions = torch.load("encoder_layer_0_output_attentions.pt")
    past_key_values = torch.load("encoder_layer_0_past_key_values.pt")
    attn = model.apm.layers[0].self_attn.eval()
    torch_hidden_states, attn_weights, past_key_values = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        past_key_value=past_key_values,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: attn,
        custom_preprocessor=create_attn_preprocessor(device, weight_dtype),
        device=device,
    )
    hidden_states = ttnn.from_torch(hidden_states, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    attention_mask = ttnn.from_torch(attention_mask, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    # layer_head_mask = ttnn.from_torch(layer_head_mask, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    # output_attentions = ttnn.from_torch(output_attentions, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    # past_key_values = ttnn.from_torch(past_key_values, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_attn = whisper_attention_minicpm(
        config.audio_config,
        hidden_states,
        attention_mask=attention_mask,
        is_decode=False,
        parameters=parameters,
    )

    # Final output comparison
    ttnn_attn = tt2torch_tensor(ttnn_attn)

    ttnn_attn = ttnn_attn.reshape(torch_hidden_states.shape)
    does_pass, pcc_message = check_with_pcc(ttnn_attn, torch_hidden_states, 0.98)
    logger.info(f"Final Output PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
