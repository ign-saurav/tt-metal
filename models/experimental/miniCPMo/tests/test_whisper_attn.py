import ttnn
import pytest
import torch
from transformers import AutoModel

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.miniCPMo.tt.ttnn_whisper_encoder import whisper_attention_minicpm
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


def create_attn_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
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
    ensure_model_files()
    logger.info(f"Loading model from local reference: {REFERENCE_DIR}")

    model = AutoModel.from_pretrained(
        str(REFERENCE_DIR),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval()

    hidden_states = torch.randn((1, 500, 1024), dtype=torch.bfloat16)
    attention_mask = torch.full((1, 1, 500, 500), float("-inf"), dtype=torch.bfloat16)
    attention_mask = torch.triu(attention_mask, diagonal=1)
    layer_head_mask = None
    output_attentions = False
    past_key_values = None
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

    ttnn_attn = whisper_attention_minicpm(
        model.config.audio_config,
        hidden_states,
        attention_mask=attention_mask,
        is_decode=False,
        parameters=parameters,
    )

    ttnn_attn = tt2torch_tensor(ttnn_attn)

    ttnn_attn = ttnn_attn.reshape(torch_hidden_states.shape)
    does_pass, pcc_message = check_with_pcc(ttnn_attn, torch_hidden_states, 0.90)
    logger.info(f"Final Output PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
