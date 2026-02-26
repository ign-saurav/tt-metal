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
    ref_out = model(inputs)  # HF returns (topk_idx, topk_weight, aux_loss)

    ttnn_model = TTNNDeepseekOCRMoEGate.from_torch(model)
    set_device(ttnn_model, device)
    ttnn_model.init_parameters()
    ttnn_model.move_weights_to_device_impl()

    tt_out = ttnn_model(inputs)  # TTNN returns (topk_idx, topk_weight, aux_loss)

    ref_idx, ref_weight = ref_out[0], ref_out[1].float()
    tt_idx = tt_out[0].to_torch if hasattr(tt_out[0], "to_torch") else tt_out[0]
    tt_weight = tt_out[1].to_torch if hasattr(tt_out[1], "to_torch") else tt_out[1]
    tt_weight = tt_weight.float() if hasattr(tt_weight, "float") else tt_weight.to(torch.float32)
    tt_idx = tt_idx.long()

    # Flatten to (num_tokens, top_k)
    ref_weight = ref_weight.reshape(-1, ref_weight.shape[-1])
    ref_idx = ref_idx.reshape(-1, ref_idx.shape[-1])
    tt_weight = tt_weight.reshape(-1, tt_weight.shape[-1])
    tt_idx = tt_idx.reshape(-1, tt_idx.shape[-1])

    # Sort by expert index so we compare same experts (handles topk tie-breaking order)
    ref_perm = torch.argsort(ref_idx, dim=-1)
    ref_idx_sorted = torch.gather(ref_idx, -1, ref_perm)
    ref_weight_sorted = torch.gather(ref_weight, -1, ref_perm)
    tt_perm = torch.argsort(tt_idx, dim=-1)
    tt_idx_sorted = torch.gather(tt_idx, -1, tt_perm)
    tt_weight_sorted = torch.gather(tt_weight, -1, tt_perm)

    idx_match = (ref_idx_sorted == tt_idx_sorted).all(dim=-1)
    idx_match_rate = idx_match.float().mean().item()
    logger.info(f"TT MOE topk_idx match rate: {idx_match_rate:.4f}")
    assert idx_match_rate >= 0.90, f"topk_idx match rate {idx_match_rate} < 0.90 (float32 vs bfloat16 can differ)"

    # Compare weights by value (desc) so alignment is independent of expert order / tie-breaking
    ref_weight_by_val = torch.sort(ref_weight, dim=-1, descending=True).values
    tt_weight_by_val = torch.sort(tt_weight, dim=-1, descending=True).values
    passed, message = check_with_pcc(ref_weight_by_val, tt_weight_by_val, pcc=0.99)
    logger.info(f"TT MOE PCC: {message}")
    assert passed, f"TT MOE PCC check failed: {message}"
