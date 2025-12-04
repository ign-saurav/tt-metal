import ttnn
import pytest
import json
import torch
import logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

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

    # Prepare a simple TTS prompt
    tts_prompt = "Hello! This is a test of the TTS system."

    # Prepare TTS text (adds special tokens and padding)
    tts_text, tts_token_lens = model.prepare_tts_text(tts_prompt)

    # Tokenize the text
    tts_inputs = model.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
    tts_input_ids = torch.tensor(tts_inputs, dtype=torch.long, device=model.device).unsqueeze(0)

    # Build streaming mask
    streaming_tts_text_mask = model._build_streaming_mask(tts_token_lens).to(device=model.tts.device)

    # Setup logits processors
    from models.experimental.miniCPMo.reference.modeling_minicpmo import gen_logits

    logits_warpers, logits_processors = gen_logits(num_code=626, top_P=0.7, top_K=20, repetition_penalty=1.0)

    # Initialize past_key_values and audio_input_ids
    condition_length = (
        1 + model.tts.use_speaker_embedding * model.tts.num_spk_embs + model.tts.streaming_text_reserved_len + 1
    )
    dtype = model.tts.emb_text.weight.dtype
    device = model.tts.device

    past_key_values = [
        (
            torch.zeros(
                1,
                model.tts.config.num_attention_heads,
                condition_length - 1,
                model.tts.config.hidden_size // model.tts.config.num_attention_heads,
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                1,
                model.tts.config.num_attention_heads,
                condition_length - 1,
                model.tts.config.hidden_size // model.tts.config.num_attention_heads,
                dtype=dtype,
                device=device,
            ),
        )
        for _ in range(model.tts.config.num_hidden_layers)
    ]

    audio_input_ids = torch.zeros(1, condition_length, model.tts.num_vq, dtype=torch.long, device=device)

    # Create dummy speaker embeddings (required when use_speaker_embedding=True)
    # Shape: [batch_size, num_spk_embs, llm_hidden_size]
    llm_hidden_size = model.embed_dim  # LLM hidden size (e.g., 3584)
    if model.tts.use_speaker_embedding:
        dummy_spk_embeds = torch.zeros(
            1, model.tts.num_spk_embs, llm_hidden_size, dtype=torch.bfloat16, device=model.device
        )
    else:
        dummy_spk_embeds = None

    # Prefill text tokens (in chunks)
    output_chunk_size = 25
    for chunk_idx in range(1):  # Simplified: just do first chunk
        begin = chunk_idx * model.tts.streaming_text_chunk_size + 0
        end = min(
            (chunk_idx + 1) * model.tts.streaming_text_chunk_size
            + 1
            + model.tts.use_speaker_embedding * model.tts.num_spk_embs,
            condition_length - 1,
        )

        if end - begin > 0:
            text_input_ids = tts_input_ids[:, begin:end]
            position_ids = torch.arange(begin, end, dtype=torch.long, device=device).unsqueeze(0)

            # Prefill text with speaker embeddings
            if chunk_idx == 0 and model.tts.use_speaker_embedding:
                past_key_values = model.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=dummy_spk_embeds,
                )
            else:
                past_key_values = model.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )

    # Generate audio tokens
    outputs = model.tts.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=output_chunk_size,
        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=device),
        eos_token=torch.tensor([625], dtype=torch.long, device=device),
        logits_warpers=logits_warpers,
        logits_processors=logits_processors,
    )

    # Decode audio codes to mel spectrograms
    mel_spec = model.tts.decode_to_mel_specs([outputs.new_ids[0]])

    # Convert mel spectrograms to audio waveform
    wav, sr = model.decode_mel_to_audio(mel_spec, output_path="result_audio_tts_test.wav")

    print(f"TTS audio generated successfully! Audio shape: {wav.shape}, Sample rate: {sr}")
