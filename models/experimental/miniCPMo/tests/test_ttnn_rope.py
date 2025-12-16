# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for TTNN native RoPE vs PyTorch reference implementation.
Tests that ttnn.experimental.rotary_embedding matches the current PyTorch RoPE.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (PyTorch reference)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_cos_sin_cache(head_dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
    """Compute cos/sin cache - returns [max_pos, head_dim] tensors."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()  # [max_pos, head_dim]
    sin = emb.sin()  # [max_pos, head_dim]
    return cos, sin


def apply_rotary_pos_emb_pytorch(q, k, cos, sin, position_ids):
    """PyTorch reference RoPE implementation (current implementation)."""
    if position_ids.dim() == 1:
        positions = position_ids
    else:
        positions = position_ids.squeeze(0)

    cos_pos = cos[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_pos = sin[positions].unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos_pos) + (_rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (_rotate_half(k) * sin_pos)
    return q_embed, k_embed


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_rope_single_head(device):
    """Test TTNN RoPE on a single head to understand shape transformations."""
    torch.manual_seed(42)

    batch_size = 1
    head_dim = 64
    seq_len = 1
    position = 150

    logger.info(f"Testing single head RoPE at position {position}")

    # Single head Q tensor: [batch, 1, seq_len, head_dim]
    q = torch.randn(batch_size, 1, seq_len, head_dim, dtype=torch.bfloat16)

    cos_cache, sin_cache = compute_cos_sin_cache(head_dim)

    # PyTorch reference
    cos_pos = cos_cache[position].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
    sin_pos = sin_cache[position].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    q_pt = (q * cos_pos) + (_rotate_half(q) * sin_pos)
    logger.info(f"PyTorch output shape: {q_pt.shape}")

    # TTNN expects [seq_len, 1, batch, head_dim] with token_idx
    q_transposed = q.permute(2, 1, 0, 3)  # [seq_len=1, 1, batch=1, head_dim=64]
    logger.info(f"TTNN input shape (transposed): {q_transposed.shape}")

    # Cos/sin cache: [1, 1, max_pos, head_dim]
    cos_cache_ttnn = cos_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin_cache_ttnn = sin_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    logger.info(f"Cos cache shape: {cos_cache_ttnn.shape}")

    cos_tt = ttnn.from_torch(cos_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_tt = ttnn.from_torch(sin_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    q_tt = ttnn.from_torch(q_transposed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    logger.info(f"TTNN tensor shape: {q_tt.shape}")

    # Apply RoPE
    q_rope_tt = ttnn.experimental.rotary_embedding(q_tt, cos_tt, sin_tt, position)
    logger.info(f"TTNN output shape (on device): {q_rope_tt.shape}")

    # Convert back to torch
    q_rope_torch = ttnn.to_torch(q_rope_tt)
    logger.info(f"TTNN output shape (to_torch): {q_rope_torch.shape}")

    # Permute back: [seq_len, 1, batch, head_dim] -> [batch, 1, seq_len, head_dim]
    # Need to handle potential tile padding
    # Slice to original dimensions first
    q_rope_sliced = q_rope_torch[:seq_len, :1, :batch_size, :head_dim]
    logger.info(f"TTNN output shape (sliced): {q_rope_sliced.shape}")

    q_tt_result = q_rope_sliced.permute(2, 1, 0, 3).float()
    logger.info(f"TTNN output shape (permuted back): {q_tt_result.shape}")

    # Compare
    pcc = comp_pcc(q_pt.float(), q_tt_result)
    logger.info(f"PCC: {pcc}")

    assert pcc[0], f"PCC failed: {pcc[1]}"
    logger.info(f"✓ Single head TTNN RoPE matches PyTorch with PCC={pcc[1]:.6f}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_rope_decode_all_heads(device):
    """Test TTNN RoPE on all heads for decode mode."""
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 12
    head_dim = 64
    seq_len = 1
    position = 150

    logger.info(f"Testing all heads RoPE at position {position}")

    # Q tensor: [batch, heads, seq_len, head_dim]
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    cos_cache, sin_cache = compute_cos_sin_cache(head_dim)
    position_ids = torch.tensor([position], dtype=torch.long)

    # PyTorch reference
    q_pt, _ = apply_rotary_pos_emb_pytorch(q, q, cos_cache, sin_cache, position_ids)
    logger.info(f"PyTorch output shape: {q_pt.shape}")

    # TTNN: Prepare cos/sin cache
    cos_cache_ttnn = cos_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin_cache_ttnn = sin_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    cos_tt = ttnn.from_torch(cos_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_tt = ttnn.from_torch(sin_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Process each head
    q_results = []

    for head_idx in range(num_heads):
        # Extract single head: [batch, 1, seq_len, head_dim]
        q_head = q[:, head_idx : head_idx + 1, :, :]

        # Transpose for TTNN: [seq_len, 1, batch, head_dim]
        q_transposed = q_head.permute(2, 1, 0, 3)

        q_tt = ttnn.from_torch(q_transposed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        q_rope_tt = ttnn.experimental.rotary_embedding(q_tt, cos_tt, sin_tt, position)

        # Convert back and slice to handle tile padding
        q_rope_torch = ttnn.to_torch(q_rope_tt)
        q_rope_sliced = q_rope_torch[:seq_len, :1, :batch_size, :head_dim]

        # Permute back: [batch, 1, seq_len, head_dim]
        q_result = q_rope_sliced.permute(2, 1, 0, 3)
        q_results.append(q_result)

    # Concatenate all heads: [batch, heads, seq_len, head_dim]
    q_tt_result = torch.cat(q_results, dim=1).float()
    logger.info(f"TTNN combined output shape: {q_tt_result.shape}")

    # Compare
    pcc = comp_pcc(q_pt.float(), q_tt_result)
    logger.info(f"PCC: {pcc}")

    assert pcc[0], f"PCC failed: {pcc[1]}"
    logger.info(f"✓ All heads TTNN RoPE matches PyTorch with PCC={pcc[1]:.6f}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("position", [0, 50, 100, 200, 500])  # Test various decode positions
def test_ttnn_rope_decode_various_positions(device, position):
    """
    Test TTNN RoPE at various decode positions.
    This validates the optimization can work throughout the decode loop.
    """
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 12
    head_dim = 64
    seq_len = 1

    logger.info(f"Testing decode RoPE at position {position}")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    cos_cache, sin_cache = compute_cos_sin_cache(head_dim)
    position_ids = torch.tensor([position], dtype=torch.long)

    # PyTorch reference
    q_pt, k_pt = apply_rotary_pos_emb_pytorch(q, k, cos_cache, sin_cache, position_ids)

    # TTNN
    cos_cache_ttnn = cos_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin_cache_ttnn = sin_cache.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    cos_tt = ttnn.from_torch(cos_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_tt = ttnn.from_torch(sin_cache_ttnn, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    q_results = []
    k_results = []

    for head_idx in range(num_heads):
        q_head = q[:, head_idx : head_idx + 1, :, :].permute(2, 1, 0, 3)
        k_head = k[:, head_idx : head_idx + 1, :, :].permute(2, 1, 0, 3)

        q_tt = ttnn.from_torch(q_head, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        k_tt = ttnn.from_torch(k_head, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        q_rope_tt = ttnn.experimental.rotary_embedding(q_tt, cos_tt, sin_tt, position)
        k_rope_tt = ttnn.experimental.rotary_embedding(k_tt, cos_tt, sin_tt, position)

        q_rope = ttnn.to_torch(q_rope_tt)[:seq_len, :1, :batch_size, :head_dim].permute(2, 1, 0, 3)
        k_rope = ttnn.to_torch(k_rope_tt)[:seq_len, :1, :batch_size, :head_dim].permute(2, 1, 0, 3)

        q_results.append(q_rope)
        k_results.append(k_rope)

    q_tt_result = torch.cat(q_results, dim=1).float()
    k_tt_result = torch.cat(k_results, dim=1).float()

    pcc_q = comp_pcc(q_pt.float(), q_tt_result)
    pcc_k = comp_pcc(k_pt.float(), k_tt_result)

    logger.info(f"Position {position} - PCC Q={pcc_q[1]:.6f}, K={pcc_k[1]:.6f}")

    assert pcc_q[0], f"Q PCC failed at position {position}: {pcc_q[1]}"
    assert pcc_k[0], f"K PCC failed at position {position}: {pcc_k[1]}"


def get_rot_transformation_mat(dhead=32):
    """Generate transformation matrix for TTNN rotary_embedding_llama."""
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def precompute_freqs_llama(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute frequencies for Llama-style RoPE (interleaved pairs).
    Returns cos/sin of shape [seq_len, dim//2].
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def gather_cos_sin_llama(position_ids, cos, sin):
    """
    Gather cos/sin for specific positions, formatted for rotary_embedding_llama.
    The Llama convention stacks [cos, cos] to create interleaved pattern.
    """
    if isinstance(position_ids, int):
        position_ids = torch.tensor([position_ids])
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    # Stack to create interleaved pattern: [cos0, cos0, cos1, cos1, ...]
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def apply_rotary_pos_emb_llama(q, k, cos, sin):
    """
    Apply RoPE using Llama convention (interleaved rotation).
    q, k: [batch, heads, seq, dim]
    cos, sin: [1, 1, seq, dim] (already gathered and stacked)
    """

    def rotate_interleaved(x, cos, sin):
        # Interleaved rotation: pairs of adjacent elements
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        cos_half = cos[..., 0::2]
        sin_half = sin[..., 0::2]

        # Apply rotation to pairs
        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x1 * sin_half + x2 * cos_half

        # Interleave back
        result = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        return result

    q_embed = rotate_interleaved(q, cos, sin)
    k_embed = rotate_interleaved(k, cos, sin)
    return q_embed, k_embed


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_rotary_embedding_llama_decode(device):
    """
    Test ttnn.experimental.rotary_embedding_llama with HEIGHT_SHARDED inputs.

    This is the optimized approach that:
    1. Uses nlp_create_qkv_heads_decode to create HEIGHT_SHARDED Q/K
    2. Uses embedding lookup for cos/sin at specific positions
    3. Applies rotary_embedding_llama without any host transfers
    """
    torch.manual_seed(42)

    # ChatTTS decoder parameters
    batch_size = 1
    num_heads = 12
    head_dim = 64
    hidden_size = num_heads * head_dim  # 768
    position = 150  # Decode at position 150

    logger.info(f"Testing rotary_embedding_llama at position {position}")

    # Create random QKV fused tensor (simulating after QKV projection)
    # Shape for nlp_create_qkv_heads_decode: [1, 1, seq_len, 3 * hidden_size]
    qkv_fused = torch.randn(1, 1, 1, 3 * hidden_size, dtype=torch.bfloat16)

    # Split into Q, K, V for PyTorch reference
    q_pt = qkv_fused[..., :hidden_size].view(batch_size, 1, num_heads, head_dim)
    k_pt = qkv_fused[..., hidden_size : 2 * hidden_size].view(batch_size, 1, num_heads, head_dim)
    # Permute to [batch, heads, seq, dim] for PyTorch RoPE
    q_pt = q_pt.permute(0, 2, 1, 3)  # [1, 12, 1, 64]
    k_pt = k_pt.permute(0, 2, 1, 3)

    # Compute cos/sin cache using Llama convention (interleaved pairs)
    cos_cache_llama, sin_cache_llama = precompute_freqs_llama(head_dim, end=4096)

    # PyTorch reference RoPE using Llama convention
    cos_gathered, sin_gathered = gather_cos_sin_llama(position, cos_cache_llama, sin_cache_llama)
    q_pt_rope, k_pt_rope = apply_rotary_pos_emb_llama(q_pt.float(), k_pt.float(), cos_gathered, sin_gathered)

    # === TTNN rotary_embedding_llama setup ===
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y

    # 1. Create transformation matrix (HEIGHT_SHARDED)
    trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, num_cores, 1)
    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, ttnn.TILE_SIZE * num_cores, ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=core_grid.y, x=core_grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mat_tt = ttnn.from_torch(
        trans_mat, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=trans_mat_mem_config
    )

    # 2. Create cos/sin matrices for embedding lookup (Llama format: [max_pos, head_dim])
    # The Llama cos/sin cache uses stacked format for interleaved rotation
    # Stack [cos, cos] to create [max_pos, head_dim] with pattern [c0, c0, c1, c1, ...]
    cos_stacked = torch.stack([cos_cache_llama, cos_cache_llama], dim=-1).flatten(-2).to(torch.bfloat16)
    sin_stacked = torch.stack([sin_cache_llama, sin_cache_llama], dim=-1).flatten(-2).to(torch.bfloat16)

    cos_matrix_tt = ttnn.from_torch(cos_stacked, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    sin_matrix_tt = ttnn.from_torch(sin_stacked, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    # 3. Create position indices for embedding lookup
    rot_idxs = torch.tensor([[position]], dtype=torch.int32)  # [1, batch]
    rot_idxs_tt = ttnn.from_torch(rot_idxs, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    # 4. Lookup cos/sin for this position
    cos_pos = ttnn.embedding(rot_idxs_tt, cos_matrix_tt, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, head_dim]
    sin_pos = ttnn.embedding(rot_idxs_tt, sin_matrix_tt, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Reshape for rotary_embedding_llama: [1, batch, 1[32], head_dim]
    cos_pos = ttnn.unsqueeze_to_4D(cos_pos)  # [1, 1, 1, head_dim]
    sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
    cos_pos = ttnn.transpose(cos_pos, 1, 2)  # [1, 1, 1[32], head_dim]
    sin_pos = ttnn.transpose(sin_pos, 1, 2)
    cos_pos = ttnn.to_layout(cos_pos, ttnn.TILE_LAYOUT)
    sin_pos = ttnn.to_layout(sin_pos, ttnn.TILE_LAYOUT)

    # Create sharded memory config for cos/sin
    grid = ttnn.num_cores_to_corerangeset(batch_size, core_grid, row_wise=True).bounding_box().grid_size()
    cos_sin_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, batch_size, ttnn.TILE_SIZE, head_dim),
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    cos_pos = ttnn.interleaved_to_sharded(cos_pos, cos_sin_mem_config)
    sin_pos = ttnn.interleaved_to_sharded(sin_pos, cos_sin_mem_config)

    # 5. Create QKV heads using nlp_create_qkv_heads_decode (HEIGHT_SHARDED output)
    qkv_fused_tt = ttnn.from_torch(qkv_fused, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    try:
        q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_fused_tt,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        logger.info(f"Q shape after nlp_create_qkv_heads_decode: {q_tt.shape}")
        logger.info(f"Q memory config: {q_tt.memory_config()}")

        # 6. Apply rotary_embedding_llama (all on device, no host transfers!)
        q_rope_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_pos, sin_pos, trans_mat_tt, is_decode_mode=True)
        k_rope_tt = ttnn.experimental.rotary_embedding_llama(k_tt, cos_pos, sin_pos, trans_mat_tt, is_decode_mode=True)

        logger.info(f"Q after RoPE shape: {q_rope_tt.shape}")

        # Convert back to interleaved for comparison
        q_rope_tt = ttnn.to_memory_config(q_rope_tt, ttnn.DRAM_MEMORY_CONFIG)
        k_rope_tt = ttnn.to_memory_config(k_rope_tt, ttnn.DRAM_MEMORY_CONFIG)

        # Convert to torch for comparison
        # nlp_create_qkv_heads_decode output is [batch, seq=1, heads, dim]
        q_tt_result = ttnn.to_torch(q_rope_tt).float()
        k_tt_result = ttnn.to_torch(k_rope_tt).float()

        logger.info(f"Q result shape: {q_tt_result.shape}")

        # Permute to match PyTorch: [batch, heads, seq, dim]
        if q_tt_result.shape[1] == 1:  # [batch, seq=1, heads, dim]
            q_tt_result = q_tt_result.permute(0, 2, 1, 3)
            k_tt_result = k_tt_result.permute(0, 2, 1, 3)

        # Slice to actual dimensions (remove tile padding)
        q_tt_result = q_tt_result[:batch_size, :num_heads, :1, :head_dim]
        k_tt_result = k_tt_result[:batch_size, :num_heads, :1, :head_dim]

        logger.info(f"Q result shape after permute/slice: {q_tt_result.shape}")
        logger.info(f"Q_pt_rope shape: {q_pt_rope.shape}")

        # Compare
        pcc_q = comp_pcc(q_pt_rope.float(), q_tt_result)
        pcc_k = comp_pcc(k_pt_rope.float(), k_tt_result)

        logger.info(f"rotary_embedding_llama PCC: Q={pcc_q[1]:.6f}, K={pcc_k[1]:.6f}")

        assert pcc_q[0], f"Q PCC failed: {pcc_q[1]}"
        assert pcc_k[0], f"K PCC failed: {pcc_k[1]}"

        logger.info("✓ rotary_embedding_llama with HEIGHT_SHARDED works!")

    except Exception as e:
        logger.error(f"rotary_embedding_llama failed: {e}")
        raise
