# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR MoE gate test. TTNN gate implements the same topk methods as the reference
(modeling_deepseekv2.py MoEGate): greedy, group_limited_greedy, noaux_tc. We parametrize over
topk_method; the OCR model is loaded with greedy, so we set config and gate in place so this run
uses the chosen method (and TTNN from_torch sees it).
"""
import pytest
import torch
from torch import nn
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.tt_symbiote.utils.device_management import set_device

from models.experimental.tt_symbiote.modules.moe import TTNNDeepseekOCRMoEGate


# OCR model is loaded with topk_method=greedy; for group_limited_greedy and noaux_tc we must set n_group/topk_group
# (reference requires n_routed_experts % n_group == 0; 64 % 4 == 0).
N_GROUP = 4
TOPK_GROUP = 2


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model (HuggingFace); gate is at model.model.layers[1].mlp.gate."""
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


@pytest.mark.parametrize("topk_method", ["greedy", "group_limited_greedy", "noaux_tc"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_deepseek_ocr_moe(device, ocr_model, topk_method):
    """Torch gate vs TTNN gate for each topk method; compare indices and weights (PCC)."""
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    config = ocr_model.config
    gate = ocr_model.model.layers[1].mlp.gate
    # OCR is loaded with greedy; set config and gate so this run uses topk_method (TTNN reads config in from_torch).
    config.topk_method = topk_method
    gate.topk_method = topk_method
    if topk_method in ("group_limited_greedy", "noaux_tc"):
        config.n_group = N_GROUP
        config.topk_group = TOPK_GROUP
        gate.n_group = N_GROUP
        gate.topk_group = TOPK_GROUP
        assert config.n_routed_experts % N_GROUP == 0
    if topk_method == "noaux_tc":
        if not hasattr(gate, "e_score_correction_bias"):
            gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(gate.n_routed_experts, device=gate.weight.device, dtype=gate.weight.dtype)
            )
        gate.e_score_correction_bias.data.zero_()

    batch_size, seq_len = 1, 128
    inputs = torch.randn((batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16)

    ref_out = gate(inputs)
    ttnn_model = TTNNDeepseekOCRMoEGate.from_torch(gate)
    set_device(ttnn_model, device)
    ttnn_model.init_parameters()
    ttnn_model.move_weights_to_device_impl()
    tt_out = ttnn_model(inputs)

    ref_idx = ref_out[0].reshape(-1, ref_out[0].shape[-1])
    ref_weight = ref_out[1].float().reshape(-1, ref_out[1].shape[-1])
    tt_idx = (tt_out[0].to_torch if hasattr(tt_out[0], "to_torch") else tt_out[0]).long().reshape(-1, ref_idx.shape[-1])
    tt_weight = tt_out[1].to_torch if hasattr(tt_out[1], "to_torch") else tt_out[1]
    tt_weight = (tt_weight.float() if hasattr(tt_weight, "float") else tt_weight.to(torch.float32)).reshape(
        -1, ref_weight.shape[-1]
    )

    ref_perm = torch.argsort(ref_idx, dim=-1)
    ref_idx_sorted = torch.gather(ref_idx, -1, ref_perm)
    tt_perm = torch.argsort(tt_idx, dim=-1)
    tt_idx_sorted = torch.gather(tt_idx, -1, tt_perm)

    idx_match_rate = (ref_idx_sorted == tt_idx_sorted).all(dim=-1).float().mean().item()
    logger.info(f"TT MOE topk_idx match rate: {idx_match_rate:.4f}")
    idx_ok = 0.90 if topk_method == "greedy" else 0.85
    assert idx_match_rate >= idx_ok, f"topk_idx match rate {idx_match_rate} < {idx_ok}"

    ref_weight_by_val = torch.sort(ref_weight, dim=-1, descending=True).values
    tt_weight_by_val = torch.sort(tt_weight, dim=-1, descending=True).values
    pcc = 0.99 if topk_method == "greedy" else 0.98
    passed, message = check_with_pcc(ref_weight_by_val, tt_weight_by_val, pcc=pcc)
    logger.info(f"TT MOE PCC: {message}")
    assert passed, f"TT MOE PCC check failed: {message}"
