# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeConfig,
    Glm4MoeMoE,
    TTNNMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
import ttnn


def _to_float32(t) -> torch.Tensor:
    """Return a plain float32 torch.Tensor from any supported tensor type."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(t, TorchTTNNTensor):
        t = t.to_torch
    if not isinstance(t, torch.Tensor):
        t = ttnn.to_torch(t)
    return t.to(torch.float32)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two float tensors."""
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1]
    return float(pcc.item())


@pytest.fixture
def default_moe_config():
    """Default MoE configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "real_weights",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_glm4_moe_full(mesh_device, default_moe_config, real_weights):
    """Test full Glm4MoeMoE module with TTNN tensor-parallel acceleration."""
    if real_weights:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash").model.layers[1].mlp
        # Convert to bfloat16 so the reference matches TTNN's bfloat16 compute;
        # otherwise PyTorch runs float32 gate while TTNN uses bfloat16 weights,
        # causing systematic routing divergence unrelated to implementation quality.
        model = model.to(dtype=torch.bfloat16)
    else:
        model = Glm4MoeMoE(default_moe_config).to(dtype=torch.bfloat16)

    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 1, 115
    inputs = torch.randn((batch_size, seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16)

    outputs_torch = model(inputs)

    import os

    if os.environ.get("MOE_DEBUG"):
        router_logits_torch = model.gate(inputs)
        topk_i_torch, topk_w_torch = model.route_tokens_to_experts(router_logits_torch)

        router_logits_ttnn = ttnn.from_torch(router_logits_torch.to(torch.bfloat16))
        ttnn_model = TTNNMoE.from_torch(model)
        set_device(ttnn_model, mesh_device)
        topk_i_ttnn, topk_w_ttnn = ttnn_model.route_tokens_to_experts(router_logits_ttnn)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _to_torch_any(t):
            if isinstance(t, TorchTTNNTensor):
                return t.to_torch
            if not isinstance(t, torch.Tensor):
                return ttnn.to_torch(t)
            return t

        topk_i_ttnn_t = _to_torch_any(topk_i_ttnn).to(torch.int32)
        topk_w_ttnn_t = _to_torch_any(topk_w_ttnn).to(torch.float32)

        mismatches = (topk_i_ttnn_t != topk_i_torch).any(dim=-1).sum().item()
        max_w_diff = (topk_w_ttnn_t - topk_w_torch).abs().max().item()
        print("MOE_DEBUG: topk_index_mismatches={}, max_weight_diff={:.6f}".format(mismatches, max_w_diff))

    ttnn_model = TTNNMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)
    outputs_ttnn = ttnn_model(inputs)

    ref = _to_float32(outputs_torch)
    out = _to_float32(outputs_ttnn)

    assert ref.shape == out.shape, f"Shape mismatch: PyTorch {ref.shape} vs TTNN {out.shape}"

    pcc = _pcc(ref, out)
    max_diff = float((ref - out).abs().max().item())
    mean_diff = float((ref - out).abs().mean().item())

    print(
        f"\nGlm4MoeMoE {'(real_weights)' if real_weights else '(random_weights)'}: "
        f"PCC={pcc:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}"
    )

    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")

    assert pcc >= PCC_THRESHOLD, (
        f"PCC {pcc:.6f} is below the required threshold {PCC_THRESHOLD} for "
        f"Glm4MoeMoE ({'real_weights' if real_weights else 'random_weights'}). "
        f"Check TTNNMoE expert computation for correctness."
    )
