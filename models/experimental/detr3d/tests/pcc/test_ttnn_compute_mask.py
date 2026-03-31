# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.detr3d.ttnn.masked_transformer_encoder import TtnnMaskedTransformerEncoder


class _MaskEncoderStub:
    """Minimal wrapper to call compute_mask and compute_mask_ttnn without building full encoder."""

    def __init__(self, device):
        self.device = device


# Bind encoder methods so we can call them with only device set
_MaskEncoderStub.compute_mask = TtnnMaskedTransformerEncoder.compute_mask
_MaskEncoderStub.compute_mask_ttnn = TtnnMaskedTransformerEncoder.compute_mask_ttnn


@pytest.mark.parametrize(
    "batch_size, seq_len, radius",
    [
        (1, 512, 0.4),
        (1, 1024, 0.64),
        (1, 2048, 1.2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_compute_mask_vs_compute_mask_ttnn(
    batch_size,
    seq_len,
    radius,
    device,
):
    """Compare compute_mask() and compute_mask_ttnn() mask outputs element-wise and report mismatches."""
    torch.manual_seed(0)
    xyz = torch.randn((batch_size, seq_len, 3)) * 0.5 + 1.0

    stub = _MaskEncoderStub(device)

    # Reference: torch-based compute_mask (returns mask_ttnn, dist)
    mask_ref_ttnn, dist_ref = stub.compute_mask(xyz, radius, dist=None)
    mask_ref = ttnn.to_torch(mask_ref_ttnn)
    ttnn.deallocate(mask_ref_ttnn)

    # TTNN implementation

    tt_xyz = ttnn.from_torch(
        xyz,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask_ttnn_out, dist_ttnn_out = stub.compute_mask_ttnn(tt_xyz, radius, dist=None)
    mask_tt = ttnn.to_torch(mask_ttnn_out)
    ttnn.deallocate(mask_ttnn_out)
    ttnn.deallocate(dist_ttnn_out)

    # Trim to (batch_size, seq_len, seq_len) in case of tiling padding
    target_shape = (batch_size, seq_len, seq_len)
    if mask_ref.shape != target_shape:
        mask_ref = mask_ref.reshape(target_shape)
    if mask_tt.shape != target_shape:
        mask_tt = mask_tt.reshape(target_shape)

    ref = mask_ref.cpu().float().numpy()
    tt = mask_tt.cpu().float().numpy()

    total_elements = ref.size
    mismatched = (ref != tt).sum()
    match_pct = 100.0 * (1.0 - mismatched / total_elements) if total_elements else 100.0

    assert (ref == tt).all(), (
        f"compute_mask vs compute_mask_ttnn differ:\n"
        f"  Shape: {target_shape}\n"
        f"  total elements: {total_elements}\n"
        f"  Mismatched: {mismatched}\n"
        f"  Match: {match_pct:.4f}%\n"
        f"  Ref (sample): {ref.ravel()[:20]}\n"
        f"  TTNN (sample): {tt.ravel()[:20]}"
    )
