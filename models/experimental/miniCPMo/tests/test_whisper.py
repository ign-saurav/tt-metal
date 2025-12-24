import ttnn
import pytest
import torch

from transformers import AutoModel
from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.miniCPMo.tt.ttnn_whisper_encoder import TtnnWhisperEncoder


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_ttnn_whisper_encoder(device, input_dtype, weight_dtype):
    model_name = "openbmb/MiniCPM-o-2_6"
    logger.info(f"Loading model from HuggingFace: {model_name}")

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval()

    apm = model.apm
    apm_state_dict = apm.state_dict()

    wavforms = torch.rand(1, 80, 1000) * 2 - 1
    audio_attention_mask = torch.full((1, 1, 500, 500), float("-inf"), dtype=torch.bfloat16)
    audio_attention_mask = torch.triu(audio_attention_mask, diagonal=1)
    apm.audio_encoder_layer = -1
    audio_states = apm(wavforms, output_hidden_states=True, attention_mask=audio_attention_mask).hidden_states[
        apm.audio_encoder_layer
    ]

    ttnn_model = TtnnWhisperEncoder(
        mesh_device=device,
        config=model.config.audio_config.to_dict(),
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
