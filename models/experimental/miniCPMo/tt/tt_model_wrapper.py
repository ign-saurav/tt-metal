# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT Model Wrapper utilities for MiniCPM-o.

Provides helper functions to enable TT acceleration on models loaded from
the local reference folder by replacing PyTorch components with TT implementations.

Usage:
    from transformers import AutoModel
    from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration
    from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR

    # Ensure model files are downloaded to local reference folder
    ensure_model_files()

    # Load model from local reference folder
    model = AutoModel.from_pretrained(str(REFERENCE_DIR), trust_remote_code=True, ...)
    model = model.eval()

    # Enable TT acceleration for specific components (model_path is required)
    model = enable_tt_acceleration(model, device, components=['audio', 'tts'], model_path=str(REFERENCE_DIR))

    # Now use model as normal - TT components are drop-in replacements
    result = model.chat(msgs=msgs, tokenizer=tokenizer, ...)
"""

from typing import List, Optional
from loguru import logger


def enable_tt_acceleration(
    model,
    device,
    components: Optional[List[str]] = None,
    model_path: str = None,
):
    """
    Replace model components with TT implementations.

    This function takes a MiniCPM-o model loaded from HuggingFace and replaces
    specified components with TT-accelerated versions. The replaced components
    have the same interface as the originals, so no code changes are needed.

    Args:
        model: MiniCPM-o model loaded from HuggingFace (via AutoModel.from_pretrained)
        device: TT device (ttnn.Device or mesh device)
        components: List of components to accelerate. Options:
                   - 'llm': Replace llm (Qwen2ForCausalLM) - MAIN BOTTLENECK
                   - 'vision': Replace vpm (SigLip vision encoder)
                   - 'audio': Replace apm (Whisper audio encoder)
                   - 'tts': Replace tts (ChatTTS decoder)
                   - 'dvae': Replace tts.dvae (DVAE decoder for mel spectrogram)
                   Default: all available components
        model_path: Path to the local reference folder containing model files (required).
                   Use str(REFERENCE_DIR) from model_setup.py.

    Returns:
        The same model with components replaced by TT implementations.

    Example:
        >>> import ttnn
        >>> from transformers import AutoModel
        >>> from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration
        >>> from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR
        >>>
        >>> # Ensure model files are downloaded to local reference folder
        >>> ensure_model_files()
        >>>
        >>> # Load from local reference folder
        >>> model = AutoModel.from_pretrained(
        ...     str(REFERENCE_DIR),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ...     init_audio=True,
        ...     init_tts=True,
        ... )
        >>> model = model.eval()
        >>> model.init_tts()
        >>>
        >>> # Enable TT acceleration for LLM (main bottleneck)
        >>> device = ttnn.open_device(device_id=0)
        >>> model = enable_tt_acceleration(model, device, ['llm'], model_path=str(REFERENCE_DIR))
        >>>
        >>> # Use model normally - LLM runs on TT hardware
        >>> result = model.chat(msgs=[...], tokenizer=tokenizer, ...)
    """
    from models.experimental.miniCPMo.tt.drop_in_replacements import (
        DropInChatTTSDecoder,
        DropInAudioEncoder,
        DropInVisionEncoder,
        DropInQwen2LLM,
        DropInDVAE,
    )

    # model_path is required to ensure we use local reference folder, not HuggingFace cache
    if model_path is None:
        raise ValueError(
            "model_path is required. Use str(REFERENCE_DIR) from model_setup.py, e.g.:\n"
            "  from models.experimental.miniCPMo.tt.model_setup import REFERENCE_DIR\n"
            "  enable_tt_acceleration(model, device, components=[...], model_path=str(REFERENCE_DIR))"
        )

    # Auto-detect available components if not specified
    if components is None:
        components = []
        if hasattr(model, "vpm") and model.vpm is not None and getattr(model.config, "init_vision", False):
            components.append("vision")
        if hasattr(model, "apm") and model.apm is not None and getattr(model.config, "init_audio", False):
            components.append("audio")
        if hasattr(model, "tts") and model.tts is not None and getattr(model.config, "init_tts", False):
            components.append("tts")
        logger.info(f"Auto-detected components to accelerate: {components}")

    # Get LLM embed dimension
    embed_dim = model.embed_dim if hasattr(model, "embed_dim") else model.llm.config.hidden_size

    # Replace LLM (Qwen2ForCausalLM) - main computational bottleneck
    if "llm" in components:
        if hasattr(model, "llm") and model.llm is not None:
            logger.info("Replacing LLM (Qwen2ForCausalLM) with TT implementation...")
            model.llm = DropInQwen2LLM(
                reference_model=model.llm,
                device=device,
                model_path=model_path,
            )
        else:
            logger.warning("LLM component requested but llm not found or None")

    # Replace vision encoder
    if "vision" in components:
        if hasattr(model, "vpm") and model.vpm is not None:
            logger.info("Replacing vision encoder (vpm) with TT implementation...")
            model.vpm = DropInVisionEncoder(
                reference_model=model.vpm,
                device=device,
                config=model.config.vision_config,
            )
        else:
            logger.warning("Vision component requested but vpm not found or None")

    # Replace audio encoder
    if "audio" in components:
        if hasattr(model, "apm") and model.apm is not None:
            logger.info("Replacing audio encoder (apm) with TT implementation...")
            model.apm = DropInAudioEncoder(
                reference_model=model.apm,
                device=device,
                config=model.config.audio_config,
            )
        else:
            logger.warning("Audio component requested but apm not found or None")

    # Replace TTS decoder
    if "tts" in components:
        if hasattr(model, "tts") and model.tts is not None:
            logger.info("Replacing TTS decoder with TT implementation...")
            model.tts = DropInChatTTSDecoder(
                reference_model=model.tts,
                device=device,
                llm_embed_dim=embed_dim,
            )
        else:
            logger.warning("TTS component requested but tts not found or None")

    # Replace DVAE (inside TTS module)
    if "dvae" in components:
        if hasattr(model, "tts") and model.tts is not None and hasattr(model.tts, "dvae"):
            logger.info("Replacing DVAE with TT implementation...")
            model.tts.dvae = DropInDVAE(
                reference_model=model.tts.dvae,
                device=device,
            )
        else:
            logger.warning("DVAE component requested but tts.dvae not found or None")

    logger.info(f"✅ TT acceleration enabled for components: {components}")
    return model


def load_minicpmo_with_tt(
    model_path: str = None,
    device=None,
    init_vision: bool = False,
    init_audio: bool = True,
    init_tts: bool = True,
    tt_components: Optional[List[str]] = None,
    torch_dtype=None,
    **kwargs,
):
    """
    Convenience function to load MiniCPM-o with TT acceleration in one call.

    Args:
        model_path: Path to local reference folder (required). Use str(REFERENCE_DIR) from model_setup.py.
        device: TT device
        init_vision: Whether to initialize vision module
        init_audio: Whether to initialize audio module
        init_tts: Whether to initialize TTS module
        tt_components: Components to accelerate (default: auto-detect)
        torch_dtype: PyTorch dtype for model weights
        **kwargs: Additional arguments passed to AutoModel.from_pretrained

    Returns:
        Tuple of (model, tokenizer) with TT acceleration enabled.

    Example:
        >>> from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR
        >>> ensure_model_files()
        >>> device = ttnn.open_device(device_id=0)
        >>> model, tokenizer = load_minicpmo_with_tt(
        ...     model_path=str(REFERENCE_DIR),
        ...     device=device,
        ...     init_audio=True,
        ...     init_tts=True,
        ...     tt_components=['audio', 'tts'],
        ... )
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if model_path is None:
        raise ValueError(
            "model_path is required. Use str(REFERENCE_DIR) from model_setup.py, e.g.:\n"
            "  from models.experimental.miniCPMo.tt.model_setup import REFERENCE_DIR\n"
            "  load_minicpmo_with_tt(model_path=str(REFERENCE_DIR), device=device, ...)"
        )

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    logger.info(f"Loading MiniCPM-o from {model_path}...")

    # Load model from local reference folder
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        init_vision=init_vision,
        init_audio=init_audio,
        init_tts=init_tts,
        **kwargs,
    )
    model = model.eval()

    # Initialize TTS components if needed
    if init_tts:
        logger.info("Initializing TTS components (vocos, tokenizer)...")
        model.init_tts()

    # Enable TT acceleration
    if device is not None:
        model = enable_tt_acceleration(model, device, tt_components, model_path=model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("✅ MiniCPM-o loaded with TT acceleration")
    return model, tokenizer
