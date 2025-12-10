import ttnn
import pytest
import torch
import shutil
import os

from models.common.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger


# Compare with TTNN implementation if available
from transformers import AutoModel
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from models.experimental.miniCPMo.tt.common import torch_to_ttnn, get_activations_memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o_tts_only(device, input_dtype, weight_dtype):
    """
    Test TTS audio generation using local reference model.

    IMPORTANT: Must use AutoModel.from_pretrained (not load_checkpoint_and_dispatch)
    because load_checkpoint_and_dispatch causes numerical precision issues that
    make the TTS model produce garbage audio.
    """
    torch.manual_seed(42)

    # Clear HuggingFace cache to ensure local reference is used
    cache_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/reference")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    # Load model using AutoModel.from_pretrained with LOCAL path
    local_model_path = "models/experimental/miniCPMo/reference"
    model = AutoModel.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,
        init_tts=True,
        local_files_only=True,
    )

    # Initialize TTS components
    model.init_tts()
    model.tts.float()  # DVAE/vocos need float32 for stability
    model = model.eval()

    causal_mask = torch.load("tt_decoder_test/causal_mask.pt")
    position_ids = torch.load("tt_decoder_test/position_ids.pt")
    inputs_embeds = torch.load("tt_decoder_test/inputs_embeds.pt")
    past_key_values = torch.load("tt_decoder_test/past_key_values.pt")
    cache_position = torch.load("tt_decoder_test/cache_position.pt")

    outputs = model.tts.model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        # past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=False,
        # cache_position=cache_position,
    )

    # print(outputs.last_hidden_state)

    logger.info("Setting up TTNN ChatTTS Decoder for comparison...")

    # Initialize TTNN decoder with same config
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

    # Load weights from reference model
    logger.info("Loading weights into TTNN decoder...")
    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    # Convert to TTNN format
    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )

    # Run TTNN forward pass
    logger.info("Running TTNN decoder forward pass...")
    hidden_states_ttnn = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=None,  # Causal mask handled internally
        position_ids=None,
        past_key_values=past_key_values,
        use_cache=True,
    )

    tt_output = tt2torch_tensor(hidden_states_ttnn)
    ref_outputs = outputs.last_hidden_state
    passing, pcc_message = check_with_pcc(tt_output, ref_outputs, 0.90)
    logger.info(pcc_message)
    if passing:
        logger.info("TTNN ChatTTS Decoder forward pass passed!")
    else:
        logger.warning("TTNN ChatTTS Decoder forward pass failed!")
        logger.warning(pcc_message)
        pytest.fail(pcc_message)
