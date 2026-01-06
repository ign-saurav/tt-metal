# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
from loguru import logger
import types


def enable_tt_acceleration(
    model,
    device,
    components: Optional[List[str]] = None,
):
    """
    Replace model components with TT implementations.
    """
    # Import existing wrappers + the new Qwen wrapper
    from models.experimental.miniCPMo.tt.drop_in_replacements import (
        DropInChatTTSDecoder,
        DropInAudioEncoder,
        DropInVisionEncoder,
        DropInQwenModel,  # <--- Added this import
    )

    # Auto-detect components if None
    if components is None:
        components = []
        if hasattr(model, "vpm") and getattr(model.config, "init_vision", False):
            components.append("vision")
        if hasattr(model, "apm") and getattr(model.config, "init_audio", False):
            components.append("audio")
        if hasattr(model, "tts") and getattr(model.config, "init_tts", False):
            components.append("tts")
        logger.info(f"Auto-detected components to accelerate: {components}")

    # --- Vision ---
    if "vision" in components:
        if hasattr(model, "vpm") and model.vpm is not None:
            logger.info("Replacing vision encoder (vpm) with TT implementation...")
            model.vpm = DropInVisionEncoder(
                reference_model=model.vpm,
                device=device,
                config=model.config.vision_config,
            )

    # --- Audio ---
    if "audio" in components:
        if hasattr(model, "apm") and model.apm is not None:
            logger.info("Replacing audio encoder (apm) with TT implementation...")
            model.apm = DropInAudioEncoder(
                reference_model=model.apm,
                device=device,
                config=model.config.audio_config,
            )

    # --- TTS ---
    if "tts" in components:
        if hasattr(model, "tts") and model.tts is not None:
            embed_dim = model.embed_dim if hasattr(model, "embed_dim") else model.llm.config.hidden_size
            logger.info("Replacing TTS decoder with TT implementation...")
            model.tts = DropInChatTTSDecoder(
                reference_model=model.tts,
                device=device,
                llm_embed_dim=embed_dim,
            )

    # --- LLM (Qwen) ---
    # Check for 'llm' or 'qwen' in components list
    if "llm" in components or "qwen" in components:
        logger.info("Replacing LLM (Qwen) with TT implementation...")

        # Initialize our new DropInQwenModel
        tt_qwen_model = DropInQwenModel(device=device, config=model.config)

        # Monkey-patch the model.chat method
        # This redirects the high-level chat call to our TT implementation
        def tt_chat_wrapper(self, msgs, tokenizer, **kwargs):
            return tt_qwen_model.chat(msgs, tokenizer, **kwargs)

        model.chat = types.MethodType(tt_chat_wrapper, model)

        # Keep a reference to the TT object on the model just in case
        model.tt_llm = tt_qwen_model

    logger.info(f"✅ TT acceleration enabled for components: {components}")
    return model


# ... rest of the file (load_minicpmo_with_tt) remains the same ...
