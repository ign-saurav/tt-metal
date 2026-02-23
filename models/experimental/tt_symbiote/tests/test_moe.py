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
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
import ttnn


def compute_pcc(torch_tensor, ttnn_tensor, name, num_devices=1):
    """Compute and print PCC between torch and TTNN tensors.

    Args:
        torch_tensor: Reference torch tensor
        ttnn_tensor: TTNN tensor to compare
        name: Name for logging
        num_devices: Number of devices (for handling multi-device output slicing)
    """
    if isinstance(ttnn_tensor, TorchTTNNTensor):
        ttnn_tensor = ttnn_tensor.to_torch
    elif isinstance(ttnn_tensor, ttnn.Tensor):
        ttnn_tensor = ttnn.to_torch(ttnn_tensor)

    if isinstance(torch_tensor, TorchTTNNTensor):
        torch_tensor = torch_tensor.to_torch

    # Handle multi-device replication: ttnn output may be N*num_devices in last dim
    if torch_tensor.shape != ttnn_tensor.shape:
        torch_last_dim = torch_tensor.shape[-1]
        ttnn_last_dim = ttnn_tensor.shape[-1]

        # Check if TTNN is exactly num_devices times larger in last dim (due to replicated all_gather)
        if ttnn_last_dim == torch_last_dim * num_devices:
            print(
                f"[{name}] Multi-device replication detected ({num_devices}x), slicing first {torch_last_dim} columns"
            )
            ttnn_tensor = ttnn_tensor[..., :torch_last_dim]
        elif torch_tensor.shape[:-1] == ttnn_tensor.shape[:-1]:
            print(f"[{name}] Shape mismatch in last dim: torch={torch_tensor.shape}, ttnn={ttnn_tensor.shape}")
            print(f"[{name}] Slicing TTNN to match torch shape")
            ttnn_tensor = ttnn_tensor[..., :torch_last_dim]
        else:
            print(f"[{name}] Shape mismatch: torch={torch_tensor.shape}, ttnn={ttnn_tensor.shape}")
            return None

    t = torch_tensor.to(torch.float32).flatten()
    n = ttnn_tensor.to(torch.float32).flatten()

    if t.shape != n.shape:
        print(f"[{name}] Shape mismatch after adjustment: torch={t.shape}, ttnn={n.shape}")
        return None

    pcc = torch.corrcoef(torch.stack([t, n]))[0, 1].item()
    diff = torch.abs(t - n)
    print(f"[{name}] PCC: {pcc:.6f}, Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}")
    return pcc


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


@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Use real weights
        # False,  # Use random weights
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_glm4_moe_full(mesh_device, default_moe_config, real_weights):
    """Test full Glm4MoeMoE module with TTNN acceleration."""
    if real_weights:
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                "zai-org/GLM-4.7-Flash", trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .model.layers[1]
            .mlp
        )
        print(model)
    else:
        model = Glm4MoeMoE(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 1, 115
    hidden_size = model.config.hidden_size if hasattr(model, "config") else default_moe_config.hidden_size
    inputs = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)

    # Create TTNN model and set device
    ttnn_model = TTNNMoE.from_torch(model)
    print(ttnn_model)
    set_device(ttnn_model, mesh_device)

    num_devices = mesh_device.get_num_devices()
    print(f"\nRunning on MeshDevice with {num_devices} devices")

    print("\n" + "=" * 60)
    print("PCC CHECK PER LAYER")
    print("=" * 60)

    # ===== Layer 1: Gate (Router Logits) =====
    print("\n--- Layer 1: Gate (Router Logits) ---")
    inputs_flat = inputs.view(-1, hidden_size)
    router_logits_torch = model.gate(inputs)
    router_logits_ttnn = ttnn_model.gate(inputs)
    compute_pcc(router_logits_torch, router_logits_ttnn, "Gate/Router Logits", num_devices)

    # ===== Layer 2: Route Tokens to Experts =====
    # NOTE: Isolated test is invalid - torch tensor gets sharded when passed to TTNN
    # We'll compare routing in the Full Forward path instead
    print("\n--- Layer 2: Route Tokens to Experts (torch reference) ---")
    topk_indices_torch, topk_weights_torch = model.route_tokens_to_experts(router_logits_torch)
    print(f"[Torch] TopK Indices shape: {topk_indices_torch.shape}")
    print(f"[Torch] TopK Weights shape: {topk_weights_torch.shape}")
    print(f"[Torch] Sample indices (first 3 tokens): {topk_indices_torch[:3]}")
    print(f"[Torch] Sample weights (first 3 tokens): {topk_weights_torch[:3]}")
    print(f"[Torch] e_score_correction_bias (first 10): {model.gate.e_score_correction_bias[:10]}")
    print(
        f"[Torch] e_score_correction_bias range: min={model.gate.e_score_correction_bias.min():.4f}, max={model.gate.e_score_correction_bias.max():.4f}"
    )

    # ===== Layer 3: Shared Experts =====
    print("\n--- Layer 3: Shared Experts ---")
    shared_output_torch = model.shared_experts(inputs)
    shared_output_ttnn = ttnn_model.shared_experts(inputs)
    compute_pcc(shared_output_torch, shared_output_ttnn, "Shared Experts", num_devices)

    # ===== Layer 4: Routed Experts =====
    print("\n--- Layer 4: Routed Experts ---")
    hidden_states_flat = inputs.view(-1, hidden_size)
    experts_output_torch = model.experts(hidden_states_flat, topk_indices_torch, topk_weights_torch)
    print(f"Torch experts output shape: {experts_output_torch.shape}")
    # Note: TTNN experts require specific tensor format, comparing via full forward

    # ===== Full Forward Comparison =====
    print("\n--- Full Forward Pass ---")
    outputs_torch = model(inputs)
    outputs_ttnn = ttnn_model(inputs)
    compute_pcc(outputs_torch, outputs_ttnn, "Full MoE Output", num_devices)

    print("\n" + "=" * 60)
    print("PCC CHECK COMPLETE")
    print("=" * 60 + "\n")

    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")
