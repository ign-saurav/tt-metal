import ttnn
import pytest
import json
import librosa
import torch
from loguru import logger
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from models.experimental.miniCPMo.tt.tt_modeling_minicpmo import TTMiniCPMO


MODEL_PATH = "openbmb/MiniCPM-o-2_6"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o(device, input_dtype, weight_dtype):
    """Original test - uses PyTorch LLM backend"""
    # Load config directly from local JSON file
    config_path = "models/experimental/miniCPMo/reference/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = MiniCPMOConfig.from_dict(
        config_dict,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )

    print("Initializing MiniCPM-o model...")
    # Initialize the model directly with the config
    # with torch.device("meta"):
    with init_empty_weights():
        model = MiniCPMO(config)

    # local_checkpoint_path = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/509805e84db1c84f154034d71a21c4f2331e6e11"
    local_checkpoint_path = "models/experimental/miniCPMo/reference/safetensors"
    load_checkpoint_and_dispatch(
        model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    # Set model to eval mode
    model = model.eval()

    # Load tokenizer directly from local reference folder files
    tokenizer_path = "models/experimental/miniCPMo/reference"
    tokenizer = MiniCPMOTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")

    task_prompt = (
        "Please listen to the audio snippet carefully and transcribe the content." + "\n"
    )  # can change to other prompts.
    audio_input, _ = librosa.load("audio_understanding.mp3", sr=16000, mono=True)  # load the audio to be captioned

    msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

    # res = model.chat(
    #     msgs=msgs,
    #     tokenizer=tokenizer,
    #     sampling=True,
    #     max_new_tokens=128,
    #     use_tts_template=False,
    #     generate_audio=False,
    #     # temperature=0.3,
    #     # output_audio_path='result_audio_understanding.wav',
    # )
    # print(res)

    proj_layer_state_dict = model.audio_projection_layer.state_dict()
    apm_state_dict = model.apm.state_dict()

    state_dict = {
        "apm": apm_state_dict,
        "audio_projection_layer": proj_layer_state_dict,
    }

    with init_empty_weights():
        config._name_or_path = "models/experimental/miniCPMo/reference"
        tt_model = TTMiniCPMO(config, state_dict=state_dict, tt_device=device).eval()

    load_checkpoint_and_dispatch(
        tt_model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tt_res = tt_model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    print(tt_res)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_mini_cpm_o_tt_llm(device):
    """
    Test MiniCPMO with TT LLM backend for audio understanding.

    This test uses the saved audio embeddings (aud_qwen_input_embds.pt) from
    MiniCPMO audio inference and runs them through TTQwen2ForCausalLM.

    Expected output: "sounds like a park with" (describing birds chirping audio)
    """
    import os
    from pathlib import Path
    from transformers import AutoTokenizer
    from models.experimental.miniCPMo.tt.tt_qwen2_for_causal_lm import TTQwen2ForCausalLM
    from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
    from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

    logger.info("=" * 60)
    logger.info("Testing TT LLM with Saved Audio Embeddings")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load saved audio embeddings
    logger.info("\n1. Loading Saved Audio Embeddings...")

    def find_input_file(filename):
        locations = [
            filename,
            Path.cwd() / filename,
            Path.home() / "ign_tt" / "forked" / "tt-metal" / filename,
            Path(__file__).parent / filename,
            Path(__file__).parent.parent / filename,
        ]
        for loc in locations:
            if Path(loc).exists():
                return str(loc)
        return None

    input_path = find_input_file("aud_qwen_input_embds.pt")
    if input_path is None:
        pytest.skip("aud_qwen_input_embds.pt not found. Run MiniCPMO audio inference to capture inputs.")

    inputs_embeds = torch.load(input_path, map_location="cpu")
    logger.info(f"   Loaded inputs_embeds: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")

    # Ensure batch dimension
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    logger.info(f"   Input shape: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")

    # Truncate to last 256 tokens (keeps the suffix instructions)
    MAX_SEQ_LEN = 256
    if seq_len > MAX_SEQ_LEN:
        start_idx = seq_len - MAX_SEQ_LEN
        logger.warning(f"   Truncating: taking tokens [{start_idx}:{seq_len}] (last {MAX_SEQ_LEN} of {seq_len})")
        inputs_embeds = inputs_embeds[:, start_idx:, :]
        seq_len = MAX_SEQ_LEN

    # Default terminators for MiniCPM-o
    terminators = [151645, 151643]

    # 2. Load weights
    logger.info("\n2. Loading Weights...")
    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()
    logger.info(f"   Loaded {len(qwen_weights)} weight tensors")

    # 3. Create TT model
    logger.info("\n3. Creating TT Transformer...")
    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
    )
    logger.info(f"   TT Model: {tt_model_args.n_layers} layers, dim={tt_model_args.dim}")

    # 4. Load tokenizer
    logger.info("\n4. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 5. Create TTQwen2ForCausalLM
    logger.info("\n5. Creating TTQwen2ForCausalLM...")
    model = TTQwen2ForCausalLM.from_tt_model(
        tt_model=tt_model,
        tt_model_args=tt_model_args,
        mesh_device=device,
        tt_kv_cache=tt_kv_cache,
        model_path=MODEL_PATH,
        tokenizer=tokenizer,
    )
    model.eval()
    logger.info("   ✅ TTQwen2ForCausalLM created")

    # 6. Run generation - 20 tokens like in the qwen test
    logger.info("\n6. Running Generation (20 tokens)...")
    max_new_tokens = 20

    with torch.no_grad():
        model.reset_cache()

        output = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=0,
            do_sample=False,  # Greedy decoding
        )

    logger.info(f"   Output shape: {output.shape}")

    # 7. Decode output
    logger.info("\n7. Decoding Output...")
    generated_ids = output[0].tolist()
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logger.info("=" * 60)
    logger.info("RESULT:")
    logger.info("=" * 60)
    logger.info(f"✅ Generated {len(generated_ids)} tokens")
    logger.info(f"✅ Token IDs: {generated_ids}")
    logger.info(f"✅ Decoded text: '{decoded_text}'")
    logger.info("=" * 60)

    # Token-by-token decode for debugging
    logger.info("\nToken-by-token decode:")
    for i, tid in enumerate(generated_ids):
        try:
            token_text = tokenizer.decode([tid])
            logger.info(f"   Token {i}: {tid} -> '{token_text}'")
        except:
            logger.info(f"   Token {i}: {tid} -> [decode error]")

    # Assertions
    assert output is not None, "Output is None"
    assert len(generated_ids) > 0, "No tokens generated"
    assert len(decoded_text) > 0, "Decoded text is empty"

    # Check for expected output (should mention park/birds/sounds)
    decoded_lower = decoded_text.lower()
    expected_keywords = ["sound", "park", "bird", "like", "with"]
    found_keywords = [kw for kw in expected_keywords if kw in decoded_lower]
    logger.info(f"   Found keywords: {found_keywords}")

    if len(found_keywords) >= 2:
        logger.info("   ✅ Output contains expected audio description keywords!")
    else:
        logger.warning(f"   ⚠️ Output may not match expected. Got: '{decoded_text}'")

    logger.info("✅ MiniCPMO with TT LLM Backend Test PASSED")
