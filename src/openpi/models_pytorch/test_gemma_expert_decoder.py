"""Test script for gemma_expert decoder only.

This script tests the gemma_expert decoder part of the PaliGemmaWithExpertModel
independently, without the vision tower or language model components.
Saves outputs in the same format as pi0_inference.py for PCC comparison with TT hardware.
"""

import numpy as np
import torch
from safetensors.torch import load_file

import openpi.models.gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.pi0_config_jax import Pi0Config


def load_checkpoint(model, ckpt_path):
    """Load checkpoint into model."""
    print(f"Loading checkpoint from {ckpt_path}")
    sd = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("Checkpoint loaded.\n")
    return model


def test_gemma_expert_decoder_only(ckpt_path=None, save_outputs=True):
    """Test the gemma_expert decoder model in isolation and save outputs for PCC comparison.
    
    Args:
        ckpt_path: Optional path to checkpoint file. If None, uses random weights.
        save_outputs: If True, save outputs to numpy file like pi0_inference.py
    """
    print("=" * 80)
    print("Testing gemma_expert decoder only")
    print("=" * 80)
    
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")
    
    # Get configs (same as pi0_inference.py)
    config = Pi0Config()
    paligemma_config = _gemma.get_config(config.paligemma_variant)
    action_expert_config = _gemma.get_config(config.action_expert_variant)
    
    print(f"\nAction expert config:")
    print(f"  width (hidden_size): {action_expert_config.width}")
    print(f"  depth: {action_expert_config.depth}")
    print(f"  mlp_dim: {action_expert_config.mlp_dim}")
    print(f"  num_heads: {action_expert_config.num_heads}")
    print(f"  num_kv_heads: {action_expert_config.num_kv_heads}")
    print(f"  head_dim: {action_expert_config.head_dim}")
    
    # Create full PI0Pytorch model to get the embedding layers
    print("\nCreating PI0Pytorch model...")
    pi0_model = PI0Pytorch(config)
    pi0_model.eval()
    
    # Load checkpoint if provided
    if ckpt_path:
        load_checkpoint(pi0_model, ckpt_path)
    
    pi0_model.to(device)
    
    # Extract gemma_expert from the model
    gemma_expert = pi0_model.paligemma_with_expert.gemma_expert
    gemma_expert_decoder = gemma_expert.model
    gemma_expert_decoder.eval()
    
    print(f"\nGemma expert decoder structure:")
    print(f"  Hidden size: {gemma_expert_decoder.config.hidden_size}")
    print(f"  Number of layers: {len(gemma_expert_decoder.layers)}")
    print(f"  Embed tokens: {gemma_expert_decoder.embed_tokens}")  # Should be None
    
    # Create test inputs matching pi0_inference.py format
    batch_size = 1
    action_horizon = config.action_horizon
    action_dim = config.action_dim
    
    print(f"\nCreating test inputs (matching pi0_inference.py format)...")
    print(f"  Batch size: {batch_size}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  Action dim: {action_dim}")
    
    # Create dummy state (same format as pi0_inference.py)
    dummy_state_np = [
        # arm 1 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # arm 2 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # grippers / extra (18)
        *([0.0] * 18)
    ]
    state = torch.tensor(dummy_state_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Create dummy noisy actions (same seed as pi0_inference.py for reproducibility)
    np.random.seed(0)
    noise_np = np.random.randn(
        batch_size,
        action_horizon,
        action_dim,
    ).astype(np.float32)
    noisy_actions = torch.from_numpy(noise_np).to(device)
    
    # Create timestep
    timestep = torch.tensor(0.5, dtype=torch.float32, device=device).expand(batch_size)
    
    print(f"\nInput shapes:")
    print(f"  state: {state.shape}")
    print(f"  noisy_actions: {noisy_actions.shape}")
    print(f"  timestep: {timestep.shape}")
    
    # Embed suffix using the same logic as PI0Pytorch.embed_suffix
    print("\nEmbedding suffix (state + actions + timestep)...")
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pi0_model.embed_suffix(
        state, noisy_actions, timestep
    )
    
    # Convert dtype if needed
    if (
        pi0_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
    
    print(f"\nSuffix embedding shapes:")
    print(f"  suffix_embs: {suffix_embs.shape}")
    print(f"  suffix_pad_masks: {suffix_pad_masks.shape}")
    print(f"  suffix_att_masks: {suffix_att_masks.shape}")
    
    # Create attention masks for gemma_expert only (no prefix)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    position_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1
    
    # Prepare 4D attention mask
    att_mask_4d = suffix_att_2d_masks[:, None, :, :].float()
    att_mask_4d = torch.where(att_mask_4d == 0, float("-inf"), 0.0)
    
    print(f"\nAttention mask shapes:")
    print(f"  att_mask_4d: {att_mask_4d.shape}")
    print(f"  position_ids: {position_ids.shape}")
    
    # Forward pass through gemma_expert decoder only
    print("\n" + "=" * 80)
    print("Running forward pass through gemma_expert decoder...")
    print("=" * 80)
    
    with torch.no_grad():
        output = gemma_expert_decoder.forward(
            inputs_embeds=suffix_embs,
            attention_mask=att_mask_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
    
    decoder_output = output.last_hidden_state
    
    print(f"\nDecoder output shape: {decoder_output.shape}")
    print(f"Expected shape: {suffix_embs.shape}")
    assert decoder_output.shape == suffix_embs.shape, \
        f"Output shape mismatch! Got {decoder_output.shape}, expected {suffix_embs.shape}"
    print("✓ Decoder output shape is correct!")
    
    # Save outputs in the same format as pi0_inference.py
    if save_outputs:
        print("\n" + "=" * 80)
        print("Saving gemma_expert decoder outputs...")
        print("=" * 80)
        
        # Convert to numpy and save
        decoder_output_np = decoder_output.cpu().numpy()
        
        print("\n")
        print("shape:", decoder_output_np.shape)
        print(decoder_output_np)
        np.save("gemma_expert_decoder_outputs.npy", decoder_output_np)
        print("\ngemma_expert decoder outputs saved to 'gemma_expert_decoder_outputs.npy'")
        print("Use this file for PCC comparison with TT hardware outputs")
    
    print("\n" + "=" * 80)
    print("Test complete! ✓")
    print("=" * 80)
    
    return decoder_output


if __name__ == "__main__":
    ckpt_path = "checkpoint/model.safetensors"
    test_gemma_expert_decoder_only(ckpt_path=ckpt_path, save_outputs=True)
