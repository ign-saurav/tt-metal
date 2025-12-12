import ttnn
import pytest
import torch
import shutil
import os

from models.common.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc

from loguru import logger
from transformers import AutoModel
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from models.experimental.miniCPMo.tt.common import torch_to_ttnn, get_activations_memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_tts_decoder_prefill(device, input_dtype, weight_dtype):
    """
    Test TTS decoder in prefill mode comparing TTNN implementation against PyTorch reference.

    Prefill mode processes multiple tokens at once without past_key_values.
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

    # Generate random input embeddings for prefill
    position_ids = torch.load("position_ids_for_prefill_text.pt")
    inputs_embeds = torch.load("inputs_embeds_for_prefill_text.pt")
    past_key_values_for_prefill = torch.load("past_key_values_for_prefill.pt")

    # Run PyTorch reference (prefill mode: no past_key_values)
    outputs = model.tts.model(
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past_key_values_for_prefill,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=False,
    )

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
    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )
    # Run TTNN forward pass (prefill mode: no past_key_values)
    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past_key_values_for_prefill,
        use_cache=True,
    )

    # Compare outputs
    tt_output = tt2torch_tensor(hidden_states_ttnn)
    ref_outputs = outputs.last_hidden_state

    # Handle shape mismatch if any
    if tt_output.shape != ref_outputs.shape:
        slices = tuple(slice(0, s) for s in ref_outputs.shape)
        tt_output = tt_output[slices]

    passing, pcc_message = check_with_pcc(tt_output, ref_outputs, 0.90)
    logger.info(f"Prefill mode PCC: {pcc_message}")
    assert passing, f"Prefill mode PCC check failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_tts_decoder_decode(device, input_dtype, weight_dtype):
    """
    Test TTS decoder in decode mode comparing TTNN implementation against PyTorch reference.

    Decode mode processes a single token with past_key_values (KV cache).
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

    # Load test inputs
    causal_mask = torch.load("tt_decoder_test/causal_mask.pt")
    position_ids = torch.load("tt_decoder_test/position_ids.pt")
    inputs_embeds = torch.load("tt_decoder_test/inputs_embeds.pt")
    past_key_values = torch.load("tt_decoder_test/past_key_values.pt")

    # Run PyTorch reference
    outputs = model.tts.model(
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
    )

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
    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    # Convert to TTNN format
    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )

    # Run TTNN forward pass
    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    # Compare outputs
    tt_output = tt2torch_tensor(hidden_states_ttnn)
    ref_outputs = outputs.last_hidden_state

    # Handle shape mismatch if any
    if tt_output.shape != ref_outputs.shape:
        slices = tuple(slice(0, s) for s in ref_outputs.shape)
        tt_output = tt_output[slices]

    passing, pcc_message = check_with_pcc(tt_output, ref_outputs, 0.90)
    logger.info(f"Final output PCC: {pcc_message}")
    assert passing, f"Final output PCC check failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_tts_decoder_incremental_prefill(device, input_dtype, weight_dtype):
    """
    Test TTS decoder in incremental prefill mode (yyy case).

    This tests prefilling additional text tokens when there's already content in the KV cache.
    position_ids does NOT start at 0 - it continues from where the previous prefill ended.

    This is the "yyy" case from _generate_mel_spec where begin != 0.
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

    position_ids = torch.load("position_ids_for_prefill_text_y.pt")
    inputs_embeds = torch.load("inputs_embeds_for_prefill_text_y.pt")
    past_key_values = torch.load("past_key_values_for_prefill_y.pt")

    # Run reference model for incremental prefill
    with torch.no_grad():
        ref_outputs = model.tts.model(
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,
        )

    # Initialize TTNN decoder
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
    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    # Convert inputs to TTNN
    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )

    # Run TTNN forward pass
    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=position_ids,
    )

    # Compare outputs
    tt_output = tt2torch_tensor(hidden_states_ttnn)
    ref_output = ref_outputs.last_hidden_state

    # Handle shape mismatch if any
    if tt_output.shape != ref_output.shape:
        slices = tuple(slice(0, s) for s in ref_output.shape)
        tt_output = tt_output[slices]

    passing, pcc_message = check_with_pcc(tt_output, ref_output, 0.90)
    logger.info(f"Incremental prefill PCC: {pcc_message}")
    assert passing, f"Incremental prefill PCC check failed: {pcc_message}"
