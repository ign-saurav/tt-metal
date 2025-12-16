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
        cache_position=position_ids,
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
        cache_position=position_ids,
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

    # Load test inputs (saved from _generate_mel_spec decode loop)
    causal_mask = torch.load("_debug_decode_inputs/causal_mask.pt")
    position_ids = torch.load("_debug_decode_inputs/position_ids.pt")
    inputs_embeds = torch.load("_debug_decode_inputs/inputs_embeds.pt")
    past_key_values = torch.load("_debug_decode_inputs/past_key_values.pt")
    cache_position = torch.load("_debug_decode_inputs/cache_position.pt")

    logger.info(f"Loaded inputs:")
    logger.info(f"  causal_mask shape: {causal_mask.shape}")
    logger.info(f"  causal_mask values: min={causal_mask.min():.4f}, max={causal_mask.max():.4f}")
    logger.info(f"  causal_mask sample: {causal_mask[0, 0, 0, :5]} ... {causal_mask[0, 0, 0, -5:]}")
    logger.info(f"  position_ids: {position_ids}")
    logger.info(f"  inputs_embeds shape: {inputs_embeds.shape}")
    logger.info(f"  past_key_values: {len(past_key_values)} layers, K shape: {past_key_values[0][0].shape}")
    logger.info(f"  cache_position: {cache_position}")

    # Debug: Log input stats
    logger.info(
        f"Input embeds stats: min={inputs_embeds.min():.4f}, max={inputs_embeds.max():.4f}, mean={inputs_embeds.mean():.4f}"
    )

    # Run PyTorch reference with causal_mask (matching TT streaming mask behavior)
    logger.info(f"Running PyTorch with causal_mask")
    outputs = model.tts.model(
        attention_mask=causal_mask,  # Use streaming TTS chunk mask
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        cache_position=cache_position,
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

    # Run TTNN forward pass with causal_mask for streaming TTS chunk masking
    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=causal_mask,  # Pass streaming TTS chunk mask
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )

    # Compare outputs
    tt_output = tt2torch_tensor(hidden_states_ttnn).float()
    ref_outputs = outputs.last_hidden_state.float()

    logger.info(f"Output comparison:")
    logger.info(f"  TT output shape: {tt_output.shape}, PT output shape: {ref_outputs.shape}")
    logger.info(f"  TT stats: min={tt_output.min():.4f}, max={tt_output.max():.4f}, mean={tt_output.mean():.4f}")
    logger.info(f"  PT stats: min={ref_outputs.min():.4f}, max={ref_outputs.max():.4f}, mean={ref_outputs.mean():.4f}")

    # Handle shape mismatch if any
    if tt_output.shape != ref_outputs.shape:
        logger.warning(f"Shape mismatch! TT: {tt_output.shape}, PT: {ref_outputs.shape}")
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_native_rope_in_decoder(device):
    """
    Test TTNN native RoPE as a drop-in replacement for PyTorch RoPE in decode mode.

    This validates that ttnn.experimental.rotary_embedding can be used to eliminate
    the device-to-host-to-device roundtrip for RoPE in the decode hot path.
    """
    torch.manual_seed(42)

    from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import _compute_rotary_cos_sin, _apply_rotary_pos_emb

    # ChatTTS decoder parameters
    batch_size = 1
    num_heads = 12
    head_dim = 64
    hidden_size = num_heads * head_dim  # 768

    # Simulate decode mode: single token at various positions
    test_positions = [0, 50, 100, 200, 300]

    # Precompute cos/sin cache (matching decoder initialization)
    rotary_cos, rotary_sin = _compute_rotary_cos_sin(head_dim, max_position_embeddings=4096)

    # Prepare TTNN cos/sin cache
    cos_cache_ttnn = rotary_cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, max_pos, head_dim]
    sin_cache_ttnn = rotary_sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    cos_tt = ttnn.from_torch(cos_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_tt = ttnn.from_torch(sin_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    for position in test_positions:
        # Create random Q, K tensors simulating decode mode
        # Shape after nlp_create_qkv_heads_decode and permute: [batch, heads, seq=1, head_dim]
        q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.bfloat16)

        position_ids = torch.tensor([[position]], dtype=torch.long)

        # === PyTorch Reference RoPE (current implementation) ===
        q_pt, k_pt = _apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin, position_ids)

        # === TTNN Native RoPE ===
        q_results = []
        k_results = []

        for head_idx in range(num_heads):
            # Extract single head: [batch, 1, seq=1, head_dim]
            q_head = q[:, head_idx : head_idx + 1, :, :]
            k_head = k[:, head_idx : head_idx + 1, :, :]

            # Permute for TTNN: [batch, 1, seq, dim] -> [seq, 1, batch, dim]
            q_transposed = q_head.permute(2, 1, 0, 3)  # [1, 1, 1, 64]
            k_transposed = k_head.permute(2, 1, 0, 3)

            # Convert to TTNN
            q_tt = ttnn.from_torch(q_transposed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            k_tt = ttnn.from_torch(k_transposed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            # Apply TTNN RoPE
            q_rope_tt = ttnn.experimental.rotary_embedding(q_tt, cos_tt, sin_tt, position)
            k_rope_tt = ttnn.experimental.rotary_embedding(k_tt, cos_tt, sin_tt, position)

            # Convert back and slice (handle tile padding)
            q_rope = ttnn.to_torch(q_rope_tt)[:1, :1, :1, :head_dim].permute(2, 1, 0, 3)
            k_rope = ttnn.to_torch(k_rope_tt)[:1, :1, :1, :head_dim].permute(2, 1, 0, 3)

            q_results.append(q_rope)
            k_results.append(k_rope)

        # Concatenate heads: [batch, heads, seq=1, head_dim]
        q_ttnn = torch.cat(q_results, dim=1).float()
        k_ttnn = torch.cat(k_results, dim=1).float()

        # Compare
        q_pt_float = q_pt.float()
        k_pt_float = k_pt.float()

        passing_q, pcc_q = check_with_pcc(q_ttnn, q_pt_float, 0.99)
        passing_k, pcc_k = check_with_pcc(k_ttnn, k_pt_float, 0.99)

        logger.info(f"Position {position}: Q PCC={pcc_q}, K PCC={pcc_k}")

        assert passing_q, f"Q PCC failed at position {position}: {pcc_q}"
        assert passing_k, f"K PCC failed at position {position}: {pcc_k}"

    logger.info("✓ TTNN native RoPE matches PyTorch RoPE for all test positions")
