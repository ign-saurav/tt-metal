import ttnn
import pytest
import torch

from transformers import AutoModel
from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from models.experimental.miniCPMo.tt.ttnn_audio_projector import TtnnAudioProjector
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR


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

    audio_states = torch.randn(1, 500, 1024, dtype=torch.bfloat16)

    proj_layer = model.audio_projection_layer.eval()

    proj_output = proj_layer(audio_states)
    proj_output = proj_output.transpose(1, 2)
    proj_output = model.audio_avg_pooler(proj_output)

    ttnn_audio_projector = TtnnAudioProjector(
        device=device,
        input_dim=model.config.audio_config.encoder_ffn_dim // 4,
        output_dim=model.embed_dim,
        pool_step=model.config.audio_pool_step,
    )

    ttnn_audio_projector.load_weights(proj_layer.state_dict())
    tt_audio_states = ttnn.from_torch(
        audio_states, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_proj_output = ttnn_audio_projector.forward(tt_audio_states)

    ttnn_proj_output = tt2torch_tensor(ttnn_proj_output)

    ttnn_proj_output = ttnn_proj_output.transpose(2, 1)
    ttnn_proj_output = ttnn_proj_output.reshape(proj_output.shape)
    does_pass, pcc_message = check_with_pcc(ttnn_proj_output, proj_output, 0.98)
    logger.info(f"Final Output PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
