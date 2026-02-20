# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Focused diagnostic: test full pipeline with FORCED identical routing.

Forces the TTNN model to use PyTorch's exact routing decisions, so we can
isolate how much PCC loss comes from:
  A) routing differences (bfloat16 topk ties) vs.
  B) expert computation differences (sparse_matmul precision).
"""

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeConfig,
    Glm4MoeMoE,
    TTNNMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


def _to_f32(t) -> torch.Tensor:
    if isinstance(t, TorchTTNNTensor):
        t = t.to_torch
    if not isinstance(t, torch.Tensor):
        try:
            t = ttnn.to_torch(t)
        except RuntimeError:
            t = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=-1))
    return t.to(torch.float32)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


@pytest.fixture
def default_moe_config():
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
        True,  # load from pretrained checkpoint
        False,  # random initialisation
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_experts_with_forced_routing(mesh_device, default_moe_config, real_weights):
    """Force TTNN to use PyTorch's exact routing, isolating expert compute PCC."""
    if real_weights:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash").model.layers[1].mlp
        # Convert to bfloat16 so the reference matches TTNN's bfloat16 compute;
        # otherwise PyTorch runs float32 gate while TTNN uses bfloat16 weights,
        # causing systematic routing divergence unrelated to implementation quality.
        model = model.to(dtype=torch.bfloat16)
    else:
        torch.manual_seed(42)
        model = Glm4MoeMoE(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    B, T, H = 1, 115, default_moe_config.hidden_size
    inputs = torch.randn(B, T, H, dtype=torch.bfloat16)

    # PyTorch reference
    pt_logits = model.gate(inputs)
    pt_idx, pt_w = model.route_tokens_to_experts(pt_logits)
    pt_full = model(inputs)

    # Also compute PyTorch routed-only and shared-only
    pt_hidden = inputs.view(-1, H)
    pt_routed = model.experts(pt_hidden, pt_idx, pt_w).view(B, T, H)
    pt_shared = model.shared_experts(inputs)

    print(f"\n=== Expert Computation Diagnostic ({'real_weights' if real_weights else 'random_weights'}) ===")

    # Build TTNN model
    ttnn_model = TTNNMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)

    # --- Test A: Normal TTNN forward (TTNN routing) ---
    outputs_normal = ttnn_model(inputs)
    pcc_normal = _pcc(_to_f32(pt_full), _to_f32(outputs_normal))
    print(f"\n(A) Normal TTNN pipeline:    PCC = {pcc_normal:.6f}")

    # Compare routing decisions between PT and TTNN
    # Use the same router logits for fair comparison (PyTorch logits converted to TTNN)
    try:
        # Reshape to match expected format: (T, n_routed_experts)
        pt_logits_reshaped = pt_logits.view(-1, pt_logits.shape[-1]).to(torch.bfloat16)
        # Create TTNN tensor directly on the device with replication
        router_logits_ttnn = ttnn.from_torch(
            pt_logits_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn_model.route_tokens_to_experts.preprocess_weights()
        ttnn_model.route_tokens_to_experts.move_weights_to_device()
        ttnn_idx, ttnn_w = ttnn_model.route_tokens_to_experts.forward(router_logits_ttnn)
        # Convert back to torch with mesh composer for multi-device tensors
        ttnn_idx_torch = ttnn.to_torch(ttnn_idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        ttnn_w_torch = ttnn.to_torch(ttnn_w, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        # Take only first device's results since input was replicated
        num_tokens = pt_idx.shape[0]
        ttnn_idx_torch = ttnn_idx_torch[:num_tokens].to(torch.int32)
        ttnn_w_torch = ttnn_w_torch[:num_tokens].to(torch.float32)
        routing_mismatches = (ttnn_idx_torch != pt_idx).any(dim=-1).sum().item()
        routing_match_pct = (1.0 - routing_mismatches / pt_idx.shape[0]) * 100.0
        max_weight_diff = (ttnn_w_torch - pt_w).abs().max().item()
        print(f"    Routing match: {routing_match_pct:.2f}% ({routing_mismatches}/{pt_idx.shape[0]} mismatches)")
        print(f"    Max weight diff: {max_weight_diff:.6f}")
    except Exception as e:
        print(f"    Routing comparison skipped due to error: {e}")

    # --- Test B: Force TTNN to use PT routing ---
    # We monkey-patch the forward to inject PT routing decisions
    original_forward = TTNNMoE.forward

    def forced_routing_forward(self, x):
        residual = x

        x = ttnn.experimental.all_gather_async(
            x,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Inject PyTorch routing decisions (replicated to all devices)
        topk_idx_tt = ttnn.from_torch(pt_idx.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        topk_idx_tt = ttnn.to_device(topk_idx_tt, x.device())
        topk_w_tt = ttnn.from_torch(pt_w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        topk_w_tt = ttnn.to_device(topk_w_tt, x.device())

        x = ttnn.unsqueeze(x, 1)
        routed_output = self.experts(x, topk_idx_tt, topk_w_tt)

        routed_output = ttnn.experimental.reduce_scatter_minimal_async(
            routed_output.to_ttnn,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        shared_output = self.shared_experts(residual)
        output = ttnn.add(routed_output, shared_output.to_ttnn)
        output = ttnn.squeeze(output, 1)
        return output

    TTNNMoE.forward = forced_routing_forward
    try:
        # Need a fresh model to avoid stale state
        ttnn_model2 = TTNNMoE.from_torch(model)
        set_device(ttnn_model2, mesh_device)
        outputs_forced = ttnn_model2(inputs)
    finally:
        TTNNMoE.forward = original_forward

    ref = _to_f32(pt_full)
    out_forced = _to_f32(outputs_forced)
    if ref.shape != out_forced.shape:
        n = min(ref.numel(), out_forced.numel())
        ref_f = ref.reshape(-1)[:n]
        out_f = out_forced.reshape(-1)[:n]
    else:
        ref_f = ref
        out_f = out_forced
    pcc_forced = _pcc(ref_f, out_f)
    print(f"(B) TTNN with PT routing:   PCC = {pcc_forced:.6f}")

    # Shared experts PCC (should be high)
    ttnn_shared = _to_f32(ttnn_model.shared_experts(inputs))
    pt_shared_f = _to_f32(pt_shared)
    if ttnn_shared.shape != pt_shared_f.shape:
        n = min(ttnn_shared.numel(), pt_shared_f.numel())
        ttnn_shared = ttnn_shared.reshape(-1)[:n]
        pt_shared_f = pt_shared_f.reshape(-1)[:n]
    pcc_shared = _pcc(pt_shared_f, ttnn_shared)
    print(f"(C) Shared experts only:    PCC = {pcc_shared:.6f}")

    # Routed-only PCC (derived from full - shared)
    # This isolates the expert matmul precision
    # Flatten all tensors to ensure shape compatibility
    out_f_flat = out_f.reshape(-1)
    ref_f_flat = ref_f.reshape(-1)
    ttnn_shared_flat = ttnn_shared.reshape(-1)
    pt_shared_f_flat = pt_shared_f.reshape(-1)

    n_out = min(out_f_flat.numel(), ttnn_shared_flat.numel())
    n_ref = min(ref_f_flat.numel(), pt_shared_f_flat.numel())

    routed_ttnn_forced = out_f_flat[:n_out] - ttnn_shared_flat[:n_out] if n_out > 0 else out_f_flat
    routed_pt = ref_f_flat[:n_ref] - pt_shared_f_flat[:n_ref] if n_ref > 0 else ref_f_flat
    pcc_routed = _pcc(routed_pt, routed_ttnn_forced)
    print(f"(D) Routed experts (forced): PCC = {pcc_routed:.6f}")

    max_diff = float((ref_f - out_f).abs().max().item())
    mean_diff = float((ref_f - out_f).abs().mean().item())
    print(f"\nForced-routing: max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")
    print("\n=== Diagnostic complete ===\n")
