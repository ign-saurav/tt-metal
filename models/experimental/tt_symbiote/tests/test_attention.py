# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests self-attention with TTNN acceleration."""

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.attention import SelfAttention, SelfAttentionConfig, TTNNSelfAttention
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.tt_transformers.tt.common import PagedAttentionConfig
from transformers import AutoModelForCausalLM
from models.experimental.tt_symbiote.modules.attention import TTNNGlm4MoeLiteAttention
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_self_attention(device):
    """Test SELF Attention with TTNN acceleration."""
    config = SelfAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )
    model = SelfAttention(config).to(dtype=torch.bfloat16)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    inputs = TorchTTNNTensor(torch.randn((1, 5, 768), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNSelfAttention.from_torch(model)
    set_device(ttnn_model, device)
    ttnn_model.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "SelfAttention")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_moe_lite_attention_paged(device):
    """Test GLM-4.7-Flash Attention with paged attention using real HF model."""
    # os.environ["MESH_DEVICE"] = "N150"  # Required for TTNN modules

    # Load real GLM-4.7-Flash model from Hugging Face
    model_name = "zai-org/GLM-4.7-Flash"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Keep on CPU for weight extraction
    )

    # Get the first attention layer
    torch_attn = hf_model.model.layers[0].self_attn
    print(f"Attention layer: {torch_attn}")

    # Create paged attention config
    paged_config = PagedAttentionConfig(
        block_size=32,
        max_num_blocks=1024,
    )

    # Create TTNN attention with paged config
    ttnn_attn = TTNNGlm4MoeLiteAttention.from_torch(torch_attn, paged_attention_config=paged_config)
    print(f"TTNN attention: {ttnn_attn}")
    set_device(ttnn_attn, device)

    # Initialize paged cache
    ttnn_attn.init_paged_cache(device)
    print(f"Paged cache initialized")
    # Create page table
    page_table = TTNNGlm4MoeLiteAttention.create_page_table(
        paged_config,
        device,
        batch_size=1,
    )

    # Create position indices for decode
    position_idxs = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=device,
    )

    # Test inputs
    batch_size, seq_len, hidden_size = 1, 1, torch_attn.hidden_size
    hidden_states = TorchTTNNTensor(torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16))

    # Create position embeddings
    head_dim = torch_attn.hidden_size // torch_attn.num_heads
    cos = ttnn.from_torch(
        torch.randn(1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
    )
    sin = ttnn.from_torch(
        torch.randn(1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
    )
    position_embeddings = (cos, sin)

    # Run torch reference
    torch_attn.eval()
    with torch.no_grad():
        # Create torch inputs for reference
        torch_hidden = hidden_states.to_torch
        torch_cos = cos.to_torch
        torch_sin = sin.to_torch

        # Simulate torch attention forward (simplified)
        outputs_torch = torch_attn(torch_hidden, position_embeddings=(torch_cos, torch_sin), use_cache=False)[0]

    # Run TTNN with paged attention
    outputs_ttnn = ttnn_attn.forward(
        hidden_states,
        position_embeddings,
        page_table=page_table,
        position_idxs=position_idxs,
    )

    # Compare outputs
    compare_fn_outputs(outputs_torch, outputs_ttnn, "GLM-4.7-Flash Attention with Paged Attention")
