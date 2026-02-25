# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

# from models.experimental.deepseek_ocr.tt.tt_sam import run_tt_sam

from models.experimental.tt_symbiote.utils.device_management import set_device

from models.experimental.tt_symbiote.modules.moe import TTNNDeepseekOCRMoEGate

import torch.profiler


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model (HuggingFace); SAM is ocr_model.model.sam_model."""
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_deepseek_ocr_moe(device, ocr_model):
    """Run torch SAM and TT SAM with same input; assert PCC >= PCC_THRESHOLD."""

    torch.manual_seed(42)
    model = ocr_model.model.layers[1].mlp.gate
    # import pdb; pdb.set_trace()
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 1, 128
    inputs = torch.randn((batch_size, seq_len, ocr_model.config.hidden_size), dtype=torch.bfloat16)
    ref_out = model(inputs)

    ttnn_model = TTNNDeepseekOCRMoEGate.from_torch(model)

    set_device(ttnn_model, device)
    ttnn_model.init_parameters()
    ttnn_model.move_weights_to_device_impl()

    tt_out = ttnn_model(inputs)

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT MOE PCC: {message}")
    assert passed, f"TT MOE PCC check failed: {message}"
