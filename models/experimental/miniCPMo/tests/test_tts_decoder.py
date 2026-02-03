import ttnn
import pytest
import torch

from models.common.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc

from loguru import logger
from transformers import AutoModel
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from models.experimental.miniCPMo.tt.common import torch_to_ttnn, get_activations_memory_config
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_tts_decoder(device, input_dtype, weight_dtype):
    """
    Test TTS decoder in comparing TTNN implementation against PyTorch reference.
    """
    torch.manual_seed(42)

    ensure_model_files()
    logger.info(f"Loading model from local reference: {REFERENCE_DIR}")

    model = AutoModel.from_pretrained(
        str(REFERENCE_DIR),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,
        init_tts=True,
    )
    # Initialize TTS components
    model.init_tts()
    model.tts.float()  # DVAE/vocos need float32 for stability
    model = model.eval()

    causal_mask = torch.cat([torch.zeros(12), torch.full((303 - 12,), torch.finfo(torch.float32).min)]).view(
        1, 1, 1, 303
    )
    position_ids = torch.tensor([[302]], dtype=torch.int64)
    inputs_embeds = (torch.randn(1, 1, 768) * 0.725).add_(0.01)
    past_key_values = [(torch.randn(1, 12, 302, 64), torch.randn(1, 12, 302, 64)) for _ in range(20)]

    cache_position = torch.tensor([[302]], dtype=torch.int64)

    outputs = model.tts.model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        cache_position=cache_position,
    )

    ttnn_decoder = TtnnChatTTSDecoder(
        device=device,
        llm_dim=model.embed_dim,
        hidden_size=model.tts.config.hidden_size,
        num_attention_heads=model.tts.config.num_attention_heads,
        num_hidden_layers=model.tts.config.num_hidden_layers,
        intermediate_size=model.tts.config.intermediate_size,
        num_text_tokens=model.tts.emb_text.num_embeddings,
        num_audio_tokens=model.tts.num_audio_tokens,
        num_vq=model.tts.num_vq,
        num_spk_embs=model.tts.num_spk_embs,
        max_position_embeddings=model.tts.config.max_position_embeddings,
    )

    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )

    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )

    # Compare outputs
    tt_output = tt2torch_tensor(hidden_states_ttnn).float()
    ref_outputs = outputs.last_hidden_state.float()

    passing, pcc_message = check_with_pcc(tt_output, ref_outputs, 0.90)
    logger.info(f"Final output PCC: {pcc_message}")
    assert passing, f"Final output PCC check failed: {pcc_message}"
