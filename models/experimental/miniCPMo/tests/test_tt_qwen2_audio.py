# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test TT Qwen2 For Causal LM with Audio-like Inputs

This test validates the TTQwen2ForCausalLM implementation by:
1. Generating random embeddings with the correct shape (simulating audio embeddings)
2. Running generation using TTQwen2ForCausalLM with TT Transformer backend
3. Verifying the model can process embeddings and generate tokens

Usage:
    pytest test_tt_qwen2_audio.py -v -s

    # Or run specific test:
    pytest test_tt_qwen2_audio.py::test_tt_qwen2_audio_generate -v -s
"""

import pytest
import torch
import ttnn
import os
from loguru import logger
from typing import Optional, Tuple

from transformers import AutoTokenizer

# Import TT Qwen2
from models.experimental.miniCPMo.tt.tt_qwen2_for_causal_lm import TTQwen2ForCausalLM
from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
from models.experimental.miniCPMo.tt_transformers.common import create_tt_model
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR


# --- Configuration ---
# Use local REFERENCE_DIR to avoid flash_attn dependency from HuggingFace
MODEL_PATH = str(REFERENCE_DIR)

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


def load_saved_inputs() -> Tuple[torch.Tensor, Optional[torch.Tensor], list, Optional[torch.Tensor]]:
    """
    Generate random inputs for audio inference testing.

    Uses random tensors with the correct shape for MiniCPM-o Qwen2:
    - inputs_embeds: [batch_size, seq_len, hidden_dim] where hidden_dim=3584
    - attention_mask: [batch_size, seq_len] all ones
    - terminators: default EOS token IDs
    - reference_output: None (no reference for random inputs)

    Returns:
        Tuple of (inputs_embeds, attention_mask, terminators, reference_output)
    """
    # Generate random embeddings with correct shape
    # MiniCPM-o Qwen2: hidden_dim=3584
    batch_size = 1
    seq_len = 256  # Reasonable test sequence length
    hidden_dim = 3584

    torch.manual_seed(42)  # For reproducibility
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    logger.info(f"Generated inputs_embeds: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")

    # Generate attention mask (all ones - all tokens are valid)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    logger.info(f"Generated attention_mask: {attention_mask.shape}")

    # Use default terminators
    terminators = DEFAULT_TERMINATORS
    logger.info(f"Using default terminators: {terminators}")

    # No reference output for random inputs
    reference_output = None

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

    # Ensure model files are downloaded to local reference folder
    ensure_model_files()

    # Set HF_MODEL environment variable
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = MODEL_PATH

    # 1. Generate random inputs
    logger.info("\n1. Generating Random Inputs...")
    inputs_embeds, attention_mask, terminators, reference_output = load_saved_inputs()

    # Ensure batch dimension
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    logger.info(f"   Input shape: {inputs_embeds.shape}")
    logger.info(f"   Terminators: {terminators}")

    # Limit sequence length for L1 memory constraints
    # IMPORTANT: Take the LAST 256 tokens to preserve the suffix instructions
    # The embeddings are: [text prefix] + [audio] + [text suffix]
    # Truncating from the START preserves the suffix that tells the model what to do
    MAX_SEQ_LEN = 256
    if seq_len > MAX_SEQ_LEN:
        start_idx = seq_len - MAX_SEQ_LEN
        logger.warning(f"   Truncating: taking tokens [{start_idx}:{seq_len}] (last {MAX_SEQ_LEN} of {seq_len})")
        inputs_embeds = inputs_embeds[:, start_idx:, :]
        seq_len = MAX_SEQ_LEN
    else:
        logger.info(f"   Using full sequence length: {seq_len} tokens")

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

    # 4. Load tokenizer
    logger.info("\n4. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 5. Create TTQwen2ForCausalLM with Generator
    logger.info("\n5. Creating TTQwen2ForCausalLM with upstream Generator...")
    model = TTQwen2ForCausalLM.from_tt_model(
        tt_model=tt_model,
        tt_model_args=tt_model_args,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        model_path=MODEL_PATH,
        tokenizer=tokenizer,
    )
    model.eval()
    logger.info("   ✅ TTQwen2ForCausalLM created with upstream Generator")

    # 6. Run generation
    logger.info("\n6. Running Generation...")
    max_new_tokens = 20  # Generate more tokens for meaningful output

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
