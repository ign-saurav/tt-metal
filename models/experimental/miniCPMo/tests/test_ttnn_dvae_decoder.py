# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test TTNN DVAE Decoder against PyTorch reference using actual model modules.
"""

import torch
import pytest
import ttnn
from loguru import logger

from transformers import AutoModel

from models.experimental.miniCPMo.tt.ttnn_dvae import TtnnDVAE
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.fixture(scope="module")
def device():
    device_id = 0
    device = ttnn.open_device(device_id=device_id, l1_small_size=24576, trace_region_size=10000000)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def dvae_model():
    """Load MiniCPM-o model and extract DVAE."""
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-o-2_6",
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


def test_dvae_decoder(device, dvae_model):
    """
    Test TTNN decoder against PyTorch decoder using actual model modules.

    PyTorch path: decoder_input -> decoder -> out_conv -> coef_mul -> output
    TTNN path: decoder_input -> _decode -> coef_mul -> output
    """
    dvae, dvae_state_dict = dvae_model
    coef = dvae_state_dict["coef"]

    batch_size = 1
    time_steps = 64  # After GFSQ reshape

    logger.info("Testing TTNN decoder against PyTorch decoder (actual model)...")

    # Create TTNN model
    ttnn_model = TtnnDVAE(mesh_device=device)
    ttnn_model.load_weights(dvae_state_dict)

    # Generate test input (decoder input shape: [B, 512, T])
    torch.manual_seed(42)
    decoder_input = torch.randn(batch_size, 512, time_steps)  # [1, 512, 64]

    # ========== PyTorch Decoder (actual model modules) ==========
    with torch.no_grad():
        # Decoder (DVAEDecoder)
        pt_dec = dvae.decoder(decoder_input)  # [1, 512, 64]
        # Out conv
        pt_dec = dvae.out_conv(pt_dec)  # [1, 100, 64]
        # Coef multiplication
        pt_decoder_output = pt_dec * coef.view(100, 1)

    logger.info(f"PyTorch decoder output shape: {pt_decoder_output.shape}")
    logger.info(f"PyTorch decoder output stats: mean={pt_decoder_output.mean():.4f}, std={pt_decoder_output.std():.4f}")

    # ========== TTNN Decoder ==========
    # Convert to NHWC: [1, 512, 64] -> [1, 1, 64, 512]
    decoder_input_nhwc = decoder_input.permute(0, 2, 1).unsqueeze(1)
    tt_dec_input = ttnn.from_torch(
        decoder_input_nhwc,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_decoder_output = ttnn_model._decode(tt_dec_input)
    tt_decoder_torch = ttnn.to_torch(tt_decoder_output).float()
    tt_decoder_nchw = tt_decoder_torch.squeeze(1).permute(0, 2, 1)  # [1, 100, 64]
    tt_decoder_final = tt_decoder_nchw * coef.view(100, 1)

    logger.info(f"TTNN decoder output shape: {tt_decoder_final.shape}")
    logger.info(f"TTNN decoder output stats: mean={tt_decoder_final.mean():.4f}, std={tt_decoder_final.std():.4f}")

    # Compare
    pcc_passed, pcc_msg = check_with_pcc(pt_decoder_output, tt_decoder_final, pcc=0.99)
    logger.info(f"Decoder PCC: {pcc_msg} - {'✅' if pcc_passed else '❌'}")

    assert pcc_passed, f"Decoder PCC failed: {pcc_msg}"
    logger.info("✅ Decoder test passed!")
