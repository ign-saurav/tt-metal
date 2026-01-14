# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test TTNN DVAE Encoder against PyTorch reference using actual model modules.
"""

import torch
import pytest
import ttnn
from loguru import logger

from transformers import AutoModel

from models.experimental.miniCPMo.tt.ttnn_dvae import TtnnDVAE
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.fixture(scope="module")
def device():
    device_id = 0
    device = ttnn.open_device(device_id=device_id, l1_small_size=24576, trace_region_size=10000000)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def dvae_model():
    """Load MiniCPM-o model from local reference folder and extract DVAE."""
    # Ensure all required model files are downloaded
    ensure_model_files()

    # Load from local reference folder (patched, no flash_attn needed)
    model = AutoModel.from_pretrained(
        str(REFERENCE_DIR),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,
        init_tts=True,
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    model.tts.dvae.float()

    dvae = model.tts.dvae
    dvae_state_dict = {k: v.float() for k, v in dvae.state_dict().items()}

    yield dvae, dvae_state_dict

    del model


def test_dvae_encoder(device, dvae_model):
    """
    Test TTNN encoder against PyTorch encoder using actual model modules.

    PyTorch path: mel -> coef_div -> downsample_conv -> encoder -> output
    TTNN path: mel -> coef_div -> _encode -> output
    """
    dvae, dvae_state_dict = dvae_model
    coef = dvae_state_dict["coef"]

    time_steps = 64
    num_mel_bins = 100

    logger.info("Testing TTNN encoder against PyTorch encoder (actual model)...")

    # Create TTNN model
    ttnn_model = TtnnDVAE(mesh_device=device)
    ttnn_model.load_weights(dvae_state_dict)

    # Generate test input
    torch.manual_seed(42)
    mel_spectrogram = torch.randn(num_mel_bins, time_steps)  # [100, 64]

    # ========== PYTORCH PATH ==========
    logger.info("Running PyTorch encoder...")

    with torch.no_grad():
        pt_after_coef = torch.div(mel_spectrogram, coef.view(100, 1).expand(mel_spectrogram.shape))
        pt_after_downsample = dvae.downsample_conv(pt_after_coef).unsqueeze(0)  # [1, 512, 32]
        pt_encoder_output = dvae.encoder(pt_after_downsample)  # [1, 1024, 32]

    logger.info(f"PyTorch encoder output: {pt_encoder_output.shape}")
    logger.info(f"PyTorch encoder stats: mean={pt_encoder_output.mean():.4f}, std={pt_encoder_output.std():.4f}")

    # ========== TTNN PATH ==========
    logger.info("Running TTNN encoder...")

    mel_spectrogram_nhwc = mel_spectrogram.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)  # [1, 1, 64, 100]
    tt_coef = ttnn.from_torch(coef.view(1, 1, 1, 100), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = ttnn.from_torch(
        mel_spectrogram_nhwc,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_after_coef = ttnn.div(tt_input, tt_coef)

    tt_encoder_output = ttnn_model._encode(tt_after_coef)
    tt_encoder_torch = ttnn.to_torch(tt_encoder_output).float()
    tt_encoder_nchw = tt_encoder_torch.squeeze(1).permute(0, 2, 1)  # [1, 1024, 32]

    logger.info(f"TTNN encoder output: {tt_encoder_nchw.shape}")
    logger.info(f"TTNN encoder stats: mean={tt_encoder_nchw.mean():.4f}, std={tt_encoder_nchw.std():.4f}")

    pcc_passed, pcc_msg = check_with_pcc(pt_encoder_output, tt_encoder_nchw, pcc=0.99)
    logger.info(f"Encoder PCC: {pcc_msg} - {'✅' if pcc_passed else '❌'}")

    assert pcc_passed, f"Encoder PCC failed: {pcc_msg}"
    logger.info("✅ Encoder test passed!")
