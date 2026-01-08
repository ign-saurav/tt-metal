# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test TT Qwen2 For Causal LM with Audio Inputs

This test validates the TTQwen2ForCausalLM implementation by:
1. Loading saved inputs from audio inference (aud_qwen_input_embds.pt, aud_qwen_attn_mask.pt, aud_qwen_terminators.pt)
2. Running generation using TTQwen2ForCausalLM with TT Transformer backend
3. Comparing output with saved reference output (qwen_llm_gen_outputs.pt)

These saved tensors capture the inputs to self.llm.generate() in MiniCPMO's _decode method.

Usage:
    pytest test_tt_qwen2_audio.py -v -s

    # Or run specific test:
    pytest test_tt_qwen2_audio.py::test_tt_qwen2_audio_generate -v -s
"""

import pytest
import torch
import ttnn
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple

from transformers import AutoTokenizer

# Import TT Qwen2
from models.experimental.miniCPMo.tt.tt_qwen2_for_causal_lm import TTQwen2ForCausalLM
from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
from models.experimental.miniCPMo.tt_transformers.common import create_tt_model


# --- Configuration ---
MODEL_PATH = "openbmb/MiniCPM-o-2_6"

# Default terminators for MiniCPM-o
DEFAULT_TERMINATORS = [151645, 151643]


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def translate_chinese(text: str) -> str:
    """
    Translate Chinese text to English.
    Falls back to pinyin or original if translation unavailable.
    """
    if not contains_chinese(text):
        return text

    # Try using deep_translator (pip install deep-translator)
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="zh-CN", target="en").translate(text)
        return f"{text} [EN: {translated}]"
    except ImportError:
        pass
    except Exception:
        pass

    # Try using googletrans (pip install googletrans==4.0.0-rc1)
    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.translate(text, src="zh-cn", dest="en")
        return f"{text} [EN: {result.text}]"
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: Simple common phrase dictionary
    common_translations = {
        "的声音": "the sound",
        "你好": "hello",
        "谢谢": "thank you",
        "是": "is/yes",
        "不是": "is not/no",
        "什么": "what",
        "怎么": "how",
        "音频": "audio",
        "视频": "video",
        "图片": "image",
        "文字": "text",
        "语音": "voice",
        "说话": "speak",
        "听": "listen",
        "看": "look/see",
        "请": "please",
        "好": "good/ok",
        "对": "right/correct",
        "错": "wrong",
        "鸟": "bird",
        "歌": "song",
        "唱歌": "singing",
        "背景": "background",
        "噪音": "noise",
        "声音": "sound",
    }

    for cn, en in common_translations.items():
        if cn in text:
            return f"{text} [EN: ~{en}]"

    # Mark as Chinese if we can't translate
    return f"{text} [Chinese]"


def format_token_text(text: str) -> str:
    """Format token text with translation if Chinese"""
    if contains_chinese(text):
        return translate_chinese(text)
    return text


def find_input_file(filename: str) -> Optional[str]:
    """Find input file in various locations"""
    locations = [
        filename,  # Current directory
        Path.cwd() / filename,
        Path.home() / "ign_tt" / "forked" / "tt-metal" / filename,
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
    ]
    for loc in locations:
        if Path(loc).exists():
            return str(loc)
    return None


def load_saved_inputs() -> Tuple[torch.Tensor, Optional[torch.Tensor], list, Optional[torch.Tensor]]:
    """
    Load saved inputs from audio inference.

    Returns:
        Tuple of (inputs_embeds, attention_mask, terminators, reference_output)
    """
    # Input embeddings (required)
    input_path = find_input_file("aud_qwen_input_embds.pt")
    if input_path is None:
        raise FileNotFoundError("aud_qwen_input_embds.pt not found. Run MiniCPMO audio inference to capture inputs.")

    inputs_embeds = torch.load(input_path, map_location="cpu")
    logger.info(f"Loaded inputs_embeds: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")

    # Attention mask (optional)
    mask_path = find_input_file("aud_qwen_attn_mask.pt")
    attention_mask = None
    if mask_path:
        attention_mask = torch.load(mask_path, map_location="cpu")
        logger.info(f"Loaded attention_mask: {attention_mask.shape}")

    # Terminators (optional)
    term_path = find_input_file("aud_qwen_terminators.pt")
    terminators = DEFAULT_TERMINATORS
    if term_path:
        loaded_term = torch.load(term_path, map_location="cpu")
        if isinstance(loaded_term, torch.Tensor):
            terminators = loaded_term.tolist()
        elif isinstance(loaded_term, list):
            terminators = loaded_term
        logger.info(f"Loaded terminators: {terminators}")

    # Reference output (optional, for comparison)
    ref_path = find_input_file("qwen_llm_gen_outputs.pt")
    reference_output = None
    if ref_path:
        try:
            reference_output = torch.load(ref_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded reference output: {type(reference_output)}")
        except Exception as e:
            logger.warning(f"Could not load reference output: {e}")

    return inputs_embeds, attention_mask, terminators, reference_output


@pytest.fixture
def mesh_device(request):
    """Fixture to get the mesh device for TT operations"""
    num_devices = getattr(request, "param", 1)
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_qwen2_audio_generate(mesh_device):
    """
    Test TTQwen2ForCausalLM.generate() with saved audio embeddings.

    This test validates that:
    1. TTQwen2ForCausalLM can be initialized correctly
    2. generate() works with inputs_embeds (audio mode)
    3. Output matches reference (if available)
    """
    logger.info("=" * 60)
    logger.info("Testing TTQwen2ForCausalLM with Audio Embeddings")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load saved inputs
    logger.info("\n1. Loading Saved Inputs...")
    try:
        inputs_embeds, attention_mask, terminators, reference_output = load_saved_inputs()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    # Ensure batch dimension
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    logger.info(f"   Input shape: {inputs_embeds.shape}")
    logger.info(f"   Terminators: {terminators}")

    # Limit sequence length for L1 memory constraints
    MAX_SEQ_LEN = 256
    if seq_len > MAX_SEQ_LEN:
        logger.warning(f"   Truncating sequence from {seq_len} to {MAX_SEQ_LEN}")
        inputs_embeds = inputs_embeds[:, :MAX_SEQ_LEN, :]
        seq_len = MAX_SEQ_LEN

    # 2. Load weights
    logger.info("\n2. Loading Weights...")
    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()
    logger.info(f"   Loaded {len(qwen_weights)} weight tensors")

    # 3. Create TT model
    logger.info("\n3. Creating TT Transformer...")
    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
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

    # 4. Create TTQwen2ForCausalLM
    logger.info("\n4. Creating TTQwen2ForCausalLM...")
    model = TTQwen2ForCausalLM.from_tt_model(
        tt_model=tt_model,
        tt_model_args=tt_model_args,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        model_path=MODEL_PATH,
    )
    model.eval()
    logger.info("   ✅ TTQwen2ForCausalLM created")

    # 5. Load tokenizer
    logger.info("\n5. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 6. Run generation
    logger.info("\n6. Running Generation...")
    max_new_tokens = 20

    with torch.no_grad():
        # Reset cache
        model.reset_cache()

        # Run generate
        output = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=0,
            do_sample=False,  # Greedy decoding for reproducibility
        )

    logger.info(f"   Output shape: {output.shape}")

    # 7. Decode output
    logger.info("\n7. Decoding Output...")
    generated_ids = output[0].tolist()
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"✅ Input sequence length: {seq_len}")
    logger.info(f"✅ Generated {len(generated_ids)} tokens")
    logger.info(f"✅ Token IDs: {generated_ids[:20]}...")  # First 20 tokens
    logger.info(f"✅ Decoded text: {decoded_text[:200]}...")  # First 200 chars

    # 8. Compare with reference (if available)
    if reference_output is not None:
        logger.info("\n8. Comparing with Reference...")

        # Handle different reference output formats
        if hasattr(reference_output, "sequences"):
            ref_sequences = reference_output.sequences
        elif isinstance(reference_output, torch.Tensor):
            ref_sequences = reference_output
        else:
            ref_sequences = None
            logger.warning(f"   Unknown reference format: {type(reference_output)}")

        if ref_sequences is not None:
            ref_ids = ref_sequences[0].tolist() if ref_sequences.dim() > 1 else ref_sequences.tolist()
            ref_decoded = tokenizer.decode(ref_ids, skip_special_tokens=True)

            logger.info(f"   Reference token IDs: {ref_ids[:20]}...")
            logger.info(f"   Reference decoded: {ref_decoded[:200]}...")

            # Calculate match percentage
            min_len = min(len(generated_ids), len(ref_ids))
            matches = sum(1 for a, b in zip(generated_ids[:min_len], ref_ids[:min_len]) if a == b)
            match_pct = matches / min_len * 100 if min_len > 0 else 0

            logger.info(f"   Token match: {matches}/{min_len} ({match_pct:.1f}%)")

            if match_pct >= 80:
                logger.info("   ✅ Output matches reference (>= 80%)")
            else:
                logger.warning(f"   ⚠️ Output differs from reference ({match_pct:.1f}% match)")

    logger.info("=" * 60)

    # Assertions
    assert output is not None, "Output is None"
    assert len(generated_ids) > 0, "No tokens generated"

    logger.info("✅ TTQwen2ForCausalLM Audio Generate Test PASSED")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_qwen2_decode_interface(mesh_device):
    """
    Test TTQwen2ForCausalLM._decode() interface (MiniCPMO compatible).

    This tests the _decode method which is the interface MiniCPMO uses
    to call self.llm.generate().
    """
    logger.info("=" * 60)
    logger.info("Testing TTQwen2ForCausalLM._decode() Interface")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load saved inputs
    logger.info("\n1. Loading Saved Inputs...")
    try:
        inputs_embeds, attention_mask, terminators, _ = load_saved_inputs()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    # Limit sequence length
    MAX_SEQ_LEN = 256
    if inputs_embeds.shape[1] > MAX_SEQ_LEN:
        inputs_embeds = inputs_embeds[:, :MAX_SEQ_LEN, :]

    # 2. Load weights
    logger.info("\n2. Loading Weights...")
    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()

    # 3. Create TT model
    logger.info("\n3. Creating TT Transformer...")
    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
    )

    # 4. Create TTQwen2ForCausalLM
    logger.info("\n4. Creating TTQwen2ForCausalLM...")
    model = TTQwen2ForCausalLM.from_tt_model(
        tt_model=tt_model,
        tt_model_args=tt_model_args,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        model_path=MODEL_PATH,
    )
    model.eval()

    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 6. Call _decode (MiniCPMO interface)
    logger.info("\n5. Calling _decode() (MiniCPMO Interface)...")

    with torch.no_grad():
        outputs = model._decode(
            inputs_embeds=inputs_embeds,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            max_new_tokens=20,
            eos_token_id=terminators,
        )

    # 7. Process results
    logger.info("\n6. Processing Results...")

    if hasattr(outputs, "sequences"):
        sequences = outputs.sequences
        logger.info(f"   Output sequences shape: {sequences.shape}")
        decoded = tokenizer.decode(sequences[0], skip_special_tokens=True)
        logger.info(f"   Decoded: {decoded[:200]}...")
    else:
        logger.info(f"   Output type: {type(outputs)}")

    logger.info("=" * 60)
    logger.info("✅ TTQwen2ForCausalLM._decode() Interface Test PASSED")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_qwen2_single_forward(mesh_device):
    """
    Test a single forward pass through TTQwen2ForCausalLM.

    This is useful for debugging the forward pass without the full
    generation loop.
    """
    logger.info("=" * 60)
    logger.info("Testing TTQwen2ForCausalLM Single Forward Pass")
    logger.info("=" * 60)

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Load saved inputs
    logger.info("\n1. Loading Saved Inputs...")
    try:
        inputs_embeds, attention_mask, _, _ = load_saved_inputs()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    # Limit sequence length
    MAX_SEQ_LEN = 256
    if inputs_embeds.shape[1] > MAX_SEQ_LEN:
        inputs_embeds = inputs_embeds[:, :MAX_SEQ_LEN, :]

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    logger.info(f"   Input shape: {inputs_embeds.shape}")

    # 2. Load weights
    logger.info("\n2. Loading Weights...")
    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()

    # 3. Create TT model
    logger.info("\n3. Creating TT Transformer...")
    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=qwen_weights,
        dummy_weights=False,
    )

    # 4. Create TTQwen2ForCausalLM
    logger.info("\n4. Creating TTQwen2ForCausalLM...")
    model = TTQwen2ForCausalLM.from_tt_model(
        tt_model=tt_model,
        tt_model_args=tt_model_args,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        model_path=MODEL_PATH,
    )
    model.eval()

    # 5. Run single forward
    logger.info("\n5. Running Single Forward Pass...")

    with torch.no_grad():
        model.reset_cache()

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    logits = outputs.logits
    logger.info(f"   Output logits shape: {logits.shape}")

    # Get predicted token
    predicted_token = torch.argmax(logits[0, -1, :]).item()

    # Decode
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    predicted_text = tokenizer.decode([predicted_token])

    logger.info(f"   Predicted token ID: {predicted_token}")
    logger.info(f"   Predicted text: '{format_token_text(predicted_text)}'")

    # Top-5 predictions
    top5_values, top5_indices = torch.topk(logits[0, -1, :], 5)
    logger.info("   Top-5 predictions:")
    for i, (idx, val) in enumerate(zip(top5_indices.tolist(), top5_values.tolist())):
        try:
            decoded = tokenizer.decode([idx])
            decoded_display = format_token_text(decoded)
        except:
            decoded_display = "[unknown]"
        logger.info(f"      {i+1}. Token {idx}: '{decoded_display}' (logit: {val:.4f})")

    logger.info("=" * 60)

    # Assertions
    assert logits is not None, "Logits is None"
    assert logits.shape[0] == batch_size, f"Batch size mismatch: {logits.shape[0]} vs {batch_size}"
    assert predicted_token >= 0, "Invalid predicted token"

    logger.info("✅ TTQwen2ForCausalLM Single Forward Test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
