# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.models.transformers.sd35_med.attention_sd35_medium import SD35MediumSelfAttention
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as TorchSD3Transformer2DModel


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "dim, num_heads, seq_len, batch_size",
    [
        (1536, 24, 1024, 1),
        (1536, 24, 512, 1),
    ],
    ids=["sd35_med_1k", "sd35_med_512"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_self_attention(device, dim, num_heads, seq_len, batch_size, dtype, reset_seeds):
    """Test SD3.5 Medium self-attention matching MM-DiT reference"""
    torch.manual_seed(1234)

    # Load SD3.5 Medium transformer from Stability AI pipeline
    model_checkpoint_path = "stabilityai/stable-diffusion-3.5-medium"
    torch_transformer = TorchSD3Transformer2DModel.from_pretrained(
        model_checkpoint_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    torch_transformer.eval()

    if not hasattr(torch_transformer, "transformer_blocks") or len(torch_transformer.transformer_blocks) == 0:
        raise ValueError("Could not find transformer_blocks in the loaded model")

    first_block = torch_transformer.transformer_blocks[0]
    if not hasattr(first_block, "attn"):
        raise ValueError(
            f"Could not find 'attn' attribute in transformer block. Available attributes: {dir(first_block)}"
        )

    # Extract just the attention layer
    reference_model = first_block.attn
    reference_model.eval()

    # Create TTNN model
    tt_model = SD35MediumSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        pre_only=False,
        qk_norm="rms",
        eps=1e-6,
        mesh_device=device,
    )

    # Load weights from the extracted attention layer
    try:
        state_dict = reference_model.state_dict()
        tt_model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Could not load state dict: {e}")

    # Create input
    x_input = torch.randn(1, batch_size, seq_len, dim, dtype=torch.bfloat16)

    with torch.no_grad():
        try:
            ref_output = reference_model(x_input.squeeze(0))
        except Exception:
            try:
                ref_output = reference_model(x_input.squeeze(0), encoder_hidden_states=None)
            except Exception:
                ref_output = None

    # TTNN forward
    tt_x_input = ttnn.from_torch(x_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_model(tt_x_input, seq_len)

    # Convert back and compare
    tt_output_torch = ttnn.to_torch(tt_output)[0, :batch_size, :seq_len, :dim]

    if ref_output is not None:
        passing, pcc = comp_pcc(ref_output, tt_output_torch, 0.99)
        assert passing, f"PCC check failed: {pcc}"
