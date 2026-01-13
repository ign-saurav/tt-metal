# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test TTNN DVAE against PyTorch reference using actual model from HuggingFace.
Tests full encode-decode pipeline with GFSQ quantization.
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


def test_dvae_decode(device, dvae_model):
    """
    Test TTNN DVAE decoder against PyTorch reference.

    Matches actual usage in decode_to_mel_specs: decoder(batch_result)
    """
    dvae, dvae_state_dict = dvae_model

    logger.info("Testing TTNN DVAE decoder against PyTorch reference...")

    # Create TTNN model
    ttnn_model = TtnnDVAE(mesh_device=device)
    ttnn_model.load_weights(dvae_state_dict)

    # Generate test indices (simulating output from generate)
    # Shape matches what ConditionalChatTTS.generate() produces: [batch, num_vq, seq_len]
    # GFSQ uses 5 levels per group, 4 groups = 5^4 = 625 possible codes
    torch.manual_seed(42)
    indices = torch.randint(0, 625, (1, 4, 32))  # [batch=1, num_vq=4, seq=32]
    logger.info(f"Input indices shape: {indices.shape}")

    logger.info("Running PyTorch: dvae(indices)")
    with torch.no_grad():
        pt_output = dvae(indices)  # defaults to mode="decode"
    logger.info(f"PyTorch output shape: {pt_output.shape}")

    logger.info("Running TTNN: ttnn_model(indices, vq_layer)")
    tt_output = ttnn_model(indices, vq_layer=dvae.vq_layer)  # defaults to mode="decode"
    logger.info(f"TTNN output shape: {tt_output.shape}")

    pcc_passed, pcc_msg = check_with_pcc(pt_output, tt_output, pcc=0.99)
    logger.info(f"DVAE Decode PCC: {pcc_msg} - {'✅' if pcc_passed else '❌'}")

    assert pcc_passed, f"DVAE Decode PCC failed: {pcc_msg}"
    logger.info("✅ DVAE decode test passed!")
