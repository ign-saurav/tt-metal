import ttnn
import pytest
import json
import torch
import logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig
from tests.ttnn.utils_for_testing import check_with_pcc

# from models.experimental.miniCPMo.tt.tt_modeling_minicpmo import TtnnConditionalChatTTS

# Enable debug logging to see EOS detection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("seed", range(42))
def test_mini_cpm_o_tts_only(device, input_dtype, weight_dtype):
    torch.manual_seed(1)
    # Load config
    config_path = "models/experimental/miniCPMo/reference/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = MiniCPMOConfig.from_dict(
        config_dict,
        init_vision=False,
        init_audio=False,  # skip audio input
        init_tts=True,  # only TTS
        generate_audio=True,
        use_tts_template=True,
        temperature=0.3,
        output_audio_path="result_audio_tts.wav",
    )

    # Initialize model
    with init_empty_weights():
        model = MiniCPMO(config)

    model.init_tts()
    # Note: Don't convert to float32 here as we load weights in bfloat16
    # model.tts.float()  # optional: keep TTS in float32 for stability

    # Move model to CPU before loading checkpoint
    model = model.to_empty(device="cpu")

    # Load checkpoint
    local_checkpoint_path = "models/experimental/miniCPMo/reference/safetensors"
    load_checkpoint_and_dispatch(
        model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Ensure TTS module is in bfloat16 to match loaded weights
    model.tts = model.tts.to(dtype=torch.bfloat16)

    model = model.eval()

    # Compare with TTNN implementation if available
    from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
    from models.experimental.miniCPMo.tt.common import torch_to_ttnn, ttnn_to_torch, get_activations_memory_config

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

    # Get the audio_bos token embedding for testing
    audio_bos_token_id = model.tts.audio_bos_token_id
    audio_bos_input_ids = torch.tensor([[audio_bos_token_id]], dtype=torch.long, device=model.tts.device)
    inputs_embeds_ref = model.tts.emb_text(audio_bos_input_ids)  # [1, 1, hidden_size]

    logger.info(f"Input embeddings shape for comparison: {inputs_embeds_ref.shape}")

    # Prepare inputs for reference model forward pass WITHOUT past_key_values
    # This tests the core transformer layers without KV cache to match TTNN decoder
    # TODO: Once TTNN decoder supports past_key_values, we should test with KV cache

    # For now, test without past_key_values to verify core transformer layers work correctly
    # Position IDs for a single token (seq_len=1)
    position_ids_ref = torch.tensor([0], dtype=torch.long, device=model.tts.device).unsqueeze(0)

    # Create simple causal mask for seq_len=1 (no past tokens)
    # Shape: [batch_size=1, 1, seq_len=1, seq_len=1]
    causal_mask_ref = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=model.tts.device)

    # Run reference model forward pass WITHOUT past_key_values
    logger.info("Running reference model forward pass (without past_key_values)...")
    outputs_ref_model = model.tts.model(
        attention_mask=causal_mask_ref,
        position_ids=position_ids_ref,
        past_key_values=None,  # No KV cache for this test
        inputs_embeds=inputs_embeds_ref,
        use_cache=False,
        output_attentions=False,
    )
    hidden_states_ref = outputs_ref_model.last_hidden_state  # [1, 1, hidden_size]

    logger.info(f"Reference hidden states shape: {hidden_states_ref.shape}")

    # Convert to TTNN format
    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds_ref,
        device,
        memory_config=get_activations_memory_config(),
    )

    # Run TTNN forward pass
    logger.info("Running TTNN decoder forward pass...")
    hidden_states_ttnn = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=None,  # Causal mask handled internally
        position_ids=None,
    )

    # Get logits from TTNN
    logits_ttnn_list = ttnn_decoder.get_logits(hidden_states_ttnn)

    # Convert back to torch
    hidden_states_ttnn_torch = ttnn_to_torch(hidden_states_ttnn)
    logits_ttnn_torch = [ttnn_to_torch(logit) for logit in logits_ttnn_list]

    logger.info(f"TTNN hidden states shape: {hidden_states_ttnn_torch.shape}")

    # Compare hidden states
    # NOTE: Both reference and TTNN are tested WITHOUT past_key_values to verify
    # the core transformer layers work correctly. Once TTNN decoder supports past_key_values,
    # we should add a separate test that compares with KV cache.
    logger.info(
        f"Comparing reference (without past_key_values) vs TTNN (without past_key_values). "
        f"Shapes: ref={hidden_states_ref.shape}, ttnn={hidden_states_ttnn_torch.shape}"
    )

    # Check shapes match
    assert (
        hidden_states_ref.shape == hidden_states_ttnn_torch.shape
    ), f"Shape mismatch: ref={hidden_states_ref.shape}, ttnn={hidden_states_ttnn_torch.shape}"

    does_pass, pcc_message = check_with_pcc(hidden_states_ref, hidden_states_ttnn_torch, pcc=0.90)

    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
