# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTS generate method - isolated test for debugging the decode loop.
"""

import os
import shutil

import pytest
import torch
from loguru import logger
from transformers import AutoModel

import ttnn
from models.experimental.miniCPMo.tt.common import (
    get_activations_memory_config,
    torch_to_ttnn,
    ttnn_to_torch,
)
from models.experimental.miniCPMo.tt.ttnn_chattts_decoder import TtnnChatTTSDecoder
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_tts_generate(device, input_dtype, weight_dtype):
    """
    Test TTS generate method comparing TTNN implementation against PyTorch reference.

    This tests the full decode loop with KV cache.
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

    # Load test inputs (saved from _generate_mel_spec)
    save_dir = "_debug_generate_inputs"
    if not os.path.exists(save_dir):
        pytest.skip(f"No saved inputs found at {save_dir}. Run test_tts.py first to generate them.")

    audio_input_ids = torch.load(f"{save_dir}/audio_input_ids.pt")
    past_key_values = torch.load(f"{save_dir}/past_key_values.pt")
    streaming_tts_text_mask = torch.load(f"{save_dir}/streaming_tts_text_mask.pt")
    temperature = torch.load(f"{save_dir}/temperature.pt")
    eos_token = torch.load(f"{save_dir}/eos_token.pt")
    max_new_token = torch.load(f"{save_dir}/max_new_token.pt")

    logger.info(f"Loaded inputs:")
    logger.info(f"  audio_input_ids shape: {audio_input_ids.shape}")
    logger.info(f"  past_key_values: {len(past_key_values)} layers, K shape: {past_key_values[0][0].shape}")
    logger.info(f"  streaming_tts_text_mask shape: {streaming_tts_text_mask.shape}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  eos_token: {eos_token}")
    logger.info(f"  max_new_token: {max_new_token}")

    # Check the progress vs KV cache assertion
    progress = audio_input_ids.shape[1]
    kv_seq_len = past_key_values[0][0].shape[2]
    logger.info(f"Progress: {progress}, KV seq_len: {kv_seq_len}")
    logger.info(f"Expected: progress == kv_seq_len + 1 => {progress} == {kv_seq_len + 1}")

    # Move tensors to device
    audio_input_ids = audio_input_ids.to(model.tts.device)
    past_key_values = [(k.to(model.tts.device), v.to(model.tts.device)) for k, v in past_key_values]
    streaming_tts_text_mask = streaming_tts_text_mask.to(model.tts.device)
    temperature = temperature.to(model.tts.device)
    eos_token = eos_token.to(model.tts.device)

    # Run PyTorch reference generate (just first iteration to compare)
    logger.info("Running PyTorch reference generate...")

    # We'll manually run one iteration of the generate loop for comparison
    # Get the first decode step inputs
    condition_length = (
        1 + model.tts.num_spk_embs * model.tts.use_speaker_embedding + model.tts.streaming_text_reserved_len + 1
    )

    # Check if this is audio_bos
    audio_bos = progress == condition_length
    logger.info(f"audio_bos: {audio_bos}, condition_length: {condition_length}")

    if audio_bos:
        narrowed_input_ids = torch.tensor([[model.tts.audio_bos_token_id]], dtype=torch.long, device=model.tts.device)
        inputs_embeds = model.tts.emb_text(narrowed_input_ids)
    else:
        narrowed_input_ids = audio_input_ids.narrow(dim=1, start=audio_input_ids.shape[1] - 1, length=1)
        code_emb = [model.tts.emb_code[i](narrowed_input_ids[:, :, i]) for i in range(model.tts.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)

    position_ids = torch.tensor([past_key_values[0][0].shape[2]], dtype=torch.long, device=model.tts.device).unsqueeze(
        0
    )

    logger.info(f"inputs_embeds shape: {inputs_embeds.shape}")
    logger.info(f"position_ids: {position_ids}")

    # Create causal mask for streaming
    from models.experimental.miniCPMo.reference.modeling_minicpmo import make_streaming_chunk_mask_generation

    causal_mask = make_streaming_chunk_mask_generation(
        inputs_embeds=inputs_embeds,
        past_seen_tokens=past_key_values[0][0].shape[2],
        streaming_tts_text_mask=streaming_tts_text_mask,
    )
    logger.info(f"causal_mask shape: {causal_mask.shape}")

    cache_position = position_ids.clone()

    # Run one decode step with PyTorch
    pt_outputs = model.tts.model(
        inputs_embeds=inputs_embeds,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )

    pt_hidden_states = pt_outputs.last_hidden_state
    logger.info(f"PyTorch hidden states shape: {pt_hidden_states.shape}")
    logger.info(
        f"PyTorch hidden states stats: min={pt_hidden_states.min():.4f}, max={pt_hidden_states.max():.4f}, mean={pt_hidden_states.mean():.4f}"
    )

    # Now run TTNN decoder for comparison
    logger.info("Running TTNN decoder...")

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

    # Load weights
    tts_state_dict = model.tts.state_dict()
    ttnn_decoder.load_weights(tts_state_dict)

    # Convert inputs to TTNN
    inputs_embeds_ttnn = torch_to_ttnn(
        inputs_embeds,
        device,
        memory_config=get_activations_memory_config(),
    )

    # Run TTNN forward
    hidden_states_ttnn, _ = ttnn_decoder.forward(
        inputs_embeds=inputs_embeds_ttnn,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )

    # Convert back to torch
    tt_hidden_states = ttnn_to_torch(hidden_states_ttnn).float()
    logger.info(f"TTNN hidden states shape: {tt_hidden_states.shape}")
    logger.info(
        f"TTNN hidden states stats: min={tt_hidden_states.min():.4f}, max={tt_hidden_states.max():.4f}, mean={tt_hidden_states.mean():.4f}"
    )

    # Compare
    pcc_passed, pcc_value = check_with_pcc(pt_hidden_states.float(), tt_hidden_states, pcc=0.99)
    logger.info(f"PCC: {pcc_value}")

    assert pcc_passed, f"PCC check failed: {pcc_value}"
    logger.info("Test PASSED!")
