# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests self-attention, GLM4 Flash attention, and paged attention KV cache."""

import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.attention import (
    SelfAttention,
    SelfAttentionConfig,
    TTNNSelfAttention,
    TTNNGlm4MoeLiteAttention,
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_self_attention(device):
    config = SelfAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )
    model = SelfAttention(config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)
    inputs = TorchTTNNTensor(torch.randn((1, 5, 768), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNSelfAttention.from_torch(model)
    set_device(ttnn_model, device)
    ttnn_model.preprocess_weights()
    ttnn_model.move_weights_to_device()
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "SelfAttention")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_flash_attention(device):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.kv_lora_rank = 128
    config.q_lora_rank = 192
    config.qk_nope_head_dim = 64
    config.qk_rope_head_dim = 64
    config.qk_head_dim = 128
    config.v_head_dim = 128
    config.moe_intermediate_size = 384
    config.num_local_experts = 4
    config.num_experts_per_tok = 2

    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    torch_attn = model.model.layers[0].self_attn

    hidden = torch.randn(1, 5, 512, dtype=torch.bfloat16)
    pos = torch.arange(5).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, pos)
    torch_out = torch_attn(hidden, attention_mask=None, position_embeddings=(cos, sin))

    ttnn_attn = TTNNGlm4MoeLiteAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    inputs = TorchTTNNTensor(hidden)
    ttnn_out = ttnn_attn(inputs, position_embeddings=(cos, sin))

    compare_fn_outputs(torch_out, ttnn_out, "Glm4MoeLiteAttention")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_flash_attention_with_paged_kv_cache(device):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.kv_lora_rank = 128
    config.q_lora_rank = 192
    config.qk_nope_head_dim = 64
    config.qk_rope_head_dim = 64
    config.qk_head_dim = 128
    config.v_head_dim = 128
    config.moe_intermediate_size = 384
    config.num_local_experts = 4
    config.num_experts_per_tok = 2

    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    torch_attn = model.model.layers[0].self_attn

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=64)
    paged_cache = TTNNPagedAttentionKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim_k=128,
        head_dim_v=128,
        paged_config=paged_config,
        device=None,
        batch_size=1,
        dtype=torch.bfloat16,
    ).to_device(device)

    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()

    hidden = torch.randn(1, 5, 512, dtype=torch.bfloat16)
    pos = torch.arange(5).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, pos)

    torch_out_prefill = torch_attn(
        hidden,
        attention_mask=None,
        position_embeddings=(cos, sin),
        past_key_values=dynamic_cache,
    )

    ttnn_attn = TTNNGlm4MoeLiteAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    inputs = TorchTTNNTensor(hidden)
    cache_position = torch.arange(5).unsqueeze(0)
    ttnn_out_prefill = ttnn_attn(
        inputs,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache,
        cache_position=cache_position,
    )

    compare_fn_outputs(torch_out_prefill, ttnn_out_prefill, "Glm4MoeLiteAttention_PagedPrefill")

    assert paged_cache.get_seq_length(0) == 5
    assert dynamic_cache.get_seq_length(0) == 5


def test_paged_kv_cache_update_accuracy():
    torch.manual_seed(42)
    batch_size = 1
    num_heads = 8
    head_dim_k = 64
    head_dim_v = 48
    num_layers = 2
    block_size = 32
    max_num_blocks = 64

    paged_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    cache = TTNNPagedAttentionKVCache(
        num_layers=num_layers,
        num_kv_heads=num_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        paged_config=paged_config,
        device=None,
        batch_size=batch_size,
        dtype=torch.bfloat16,
    )

    ref_keys = [[] for _ in range(num_layers)]
    ref_vals = [[] for _ in range(num_layers)]

    prefill_len = 17
    k_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim_k, dtype=torch.bfloat16)
    v_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim_v, dtype=torch.bfloat16)

    for layer_idx in range(num_layers):
        cached_k, cached_v = cache.update(k_prefill, v_prefill, layer_idx)
        ref_keys[layer_idx].append(k_prefill)
        ref_vals[layer_idx].append(v_prefill)

        expected_k = torch.cat(ref_keys[layer_idx], dim=2)
        expected_v = torch.cat(ref_vals[layer_idx], dim=2)

        assert (
            cached_k.shape == expected_k.shape
        ), f"Layer {layer_idx} prefill K shape mismatch: {cached_k.shape} vs {expected_k.shape}"
        assert cached_v.shape == expected_v.shape
        assert torch.allclose(cached_k.float(), expected_k.float(), atol=1e-3)
        assert torch.allclose(cached_v.float(), expected_v.float(), atol=1e-3)

    for step in range(5):
        k_decode = torch.randn(batch_size, num_heads, 1, head_dim_k, dtype=torch.bfloat16)
        v_decode = torch.randn(batch_size, num_heads, 1, head_dim_v, dtype=torch.bfloat16)

        for layer_idx in range(num_layers):
            cached_k, cached_v = cache.update(k_decode, v_decode, layer_idx)
            ref_keys[layer_idx].append(k_decode)
            ref_vals[layer_idx].append(v_decode)

            expected_k = torch.cat(ref_keys[layer_idx], dim=2)
            expected_v = torch.cat(ref_vals[layer_idx], dim=2)

            assert cached_k.shape == expected_k.shape
            assert cached_v.shape == expected_v.shape
            assert torch.allclose(cached_k.float(), expected_k.float(), atol=1e-3)
            assert torch.allclose(cached_v.float(), expected_v.float(), atol=1e-3)

    assert cache.get_seq_length(0) == prefill_len + 5
    assert cache.get_seq_length(1) == prefill_len + 5


def test_paged_kv_cache_cross_block_boundary():
    torch.manual_seed(123)
    batch_size = 1
    num_heads = 4
    head_dim_k = 32
    head_dim_v = 32
    block_size = 8
    max_num_blocks = 32

    paged_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    cache = TTNNPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        paged_config=paged_config,
        device=None,
        batch_size=batch_size,
        dtype=torch.float32,
    )

    all_k = []
    all_v = []

    seq_len = block_size * 3 + 5
    k = torch.randn(batch_size, num_heads, seq_len, head_dim_k)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim_v)
    cached_k, cached_v = cache.update(k, v, layer_idx=0)
    all_k.append(k)
    all_v.append(v)

    expected_k = torch.cat(all_k, dim=2)
    expected_v = torch.cat(all_v, dim=2)
    assert torch.allclose(cached_k, expected_k, atol=1e-6)
    assert torch.allclose(cached_v, expected_v, atol=1e-6)

    for _ in range(3):
        k_step = torch.randn(batch_size, num_heads, 1, head_dim_k)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim_v)
        cached_k, cached_v = cache.update(k_step, v_step, layer_idx=0)
        all_k.append(k_step)
        all_v.append(v_step)

    expected_k = torch.cat(all_k, dim=2)
    expected_v = torch.cat(all_v, dim=2)
    assert torch.allclose(cached_k, expected_k, atol=1e-6)
    assert torch.allclose(cached_v, expected_v, atol=1e-6)
    assert cache.get_seq_length(0) == seq_len + 3


def test_paged_kv_cache_matches_dynamic_cache():
    torch.manual_seed(7)
    from transformers.cache_utils import DynamicCache

    batch_size = 1
    num_heads = 4
    head_dim = 64
    num_layers = 2

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=64)
    paged_cache = TTNNPagedAttentionKVCache(
        num_layers=num_layers,
        num_kv_heads=num_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        paged_config=paged_config,
        device=None,
        batch_size=batch_size,
        dtype=torch.float32,
    )
    dynamic_cache = DynamicCache()

    prefill_len = 20
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        v = torch.randn(batch_size, num_heads, prefill_len, head_dim)

        pk, pv = paged_cache.update(k.clone(), v.clone(), layer_idx)
        dk, dv = dynamic_cache.update(k.clone(), v.clone(), layer_idx)

        assert torch.allclose(pk.float(), dk.float(), atol=1e-5)
        assert torch.allclose(pv.float(), dv.float(), atol=1e-5)

    for step in range(10):
        for layer_idx in range(num_layers):
            k = torch.randn(batch_size, num_heads, 1, head_dim)
            v = torch.randn(batch_size, num_heads, 1, head_dim)

            pk, pv = paged_cache.update(k.clone(), v.clone(), layer_idx)
            dk, dv = dynamic_cache.update(k.clone(), v.clone(), layer_idx)

            assert torch.allclose(pk.float(), dk.float(), atol=1e-5)
            assert torch.allclose(pv.float(), dv.float(), atol=1e-5)

    assert paged_cache.get_seq_length(0) == dynamic_cache.get_seq_length(0)
    assert paged_cache.get_seq_length(1) == dynamic_cache.get_seq_length(1)
