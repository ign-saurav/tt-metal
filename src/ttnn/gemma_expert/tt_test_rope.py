import torch
import ttnn
from rms_norm import GemmaRotaryEmbeddingTTNN,apply_rotary_pos_emb_ttnn
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
from transformers.models.gemma.configuration_gemma import GemmaConfig

def pcc(a: torch.Tensor, b: torch.Tensor):
    a = a.flatten()
    b = b.flatten()
    a_mean = a.mean()
    b_mean = b.mean()
    num = torch.sum((a - a_mean) * (b - b_mean))
    den = torch.sqrt(torch.sum((a - a_mean) ** 2) * torch.sum((b - b_mean) ** 2))
    return (num / den).item()
    
def main():    
    # -----------------------
    # Configuration
    # -----------------------
    B = 2      # batch
    H = 4      # heads
    T = 8      # seq length
    D = 256    # head dim (must be even)
    
    config = GemmaConfig(hidden_size=H * D,              # 256
        num_attention_heads=H,           # 4
        max_position_embeddings=2048)
    dtype = torch.float32

    device = ttnn.open_device(device_id=0)

    # -----------------------
    # Inputs
    # -----------------------
    torch.manual_seed(0)

    q_torch = torch.randn(B, H, T, D, dtype=dtype)
    k_torch = torch.randn(B, H, T, D, dtype=dtype)

    position_ids = torch.arange(T).unsqueeze(0).repeat(B, 1)

    # -----------------------
    # PyTorch RoPE
    # -----------------------
    rope_torch = GemmaRotaryEmbedding(config)
    head_dim_torch = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    print(f"Torch inv_freq shape: {rope_torch.inv_freq.shape}")
    print(f"Config hidden_size: {config.hidden_size}, head_dim: {head_dim_torch}")
    
    cos_torch, sin_torch = rope_torch(q_torch, position_ids)

    print("Cosine Tensor Shape (Torch):", cos_torch.shape)
    print("Sine Tensor Shape (Torch):", sin_torch.shape)

    # Slice cos/sin to match head_dim (transformers computes for full hidden_size)
    # cos/sin have shape [B, T, hidden_size], but q/k have shape [B, H, T, head_dim]
    head_dim = D
    cos_torch = cos_torch[:, :, :head_dim]
    sin_torch = sin_torch[:, :, :head_dim]

    q_ref, k_ref = apply_rotary_pos_emb(
        q_torch, k_torch, cos_torch, sin_torch
    )

    # -----------------------
    # TTNN inputs
    # -----------------------
    q_ttnn = ttnn.from_torch(q_torch, device=device)
    k_ttnn = ttnn.from_torch(k_torch, device=device)
    pos_ttnn = ttnn.from_torch(position_ids, device=device)

    rope_ttnn = GemmaRotaryEmbeddingTTNN(config, device=device)
    inv_freq_ttnn_torch = ttnn.to_torch(rope_ttnn.inv_freq)
    print(f"TTNN inv_freq shape: {inv_freq_ttnn_torch.shape}")
    print(f"inv_freq PCC: {pcc(rope_torch.inv_freq, inv_freq_ttnn_torch)}")
    print(f"inv_freq first 5 - Torch: {rope_torch.inv_freq[:5]}, TTNN: {inv_freq_ttnn_torch[:5]}")

    # Our TTNN implementation computes for head_dim directly, so no slicing needed
    cos_ttnn, sin_ttnn = rope_ttnn.forward(pos_ttnn, x_dtype=ttnn.float32)

    # Debug: Compare cos/sin values
    cos_ttnn_torch = ttnn.to_torch(cos_ttnn)
    sin_ttnn_torch = ttnn.to_torch(sin_ttnn)
    
    # Check if cos values are all the same (causing nan PCC)
    cos_torch_flat = cos_torch.flatten()
    cos_ttnn_flat = cos_ttnn_torch.flatten()
    print(f"Cos std - Torch: {cos_torch_flat.std():.6f}, TTNN: {cos_ttnn_flat.std():.6f}")
    print(f"Sin std - Torch: {sin_torch.flatten().std():.6f}, TTNN: {sin_ttnn_torch.flatten().std():.6f}")
    
    cos_pcc = pcc(cos_torch, cos_ttnn_torch)
    sin_pcc = pcc(sin_torch, sin_ttnn_torch)
    print(f"Cos PCC: {cos_pcc:.6f}")
    print(f"Sin PCC: {sin_pcc:.6f}")
    print(f"Cos shapes - Torch: {cos_torch.shape}, TTNN: {cos_ttnn_torch.shape}")
    print(f"Sin shapes - Torch: {sin_torch.shape}, TTNN: {sin_ttnn_torch.shape}")
    
    # Check values at different positions
    print(f"Cos[0,0,:5] - Torch: {cos_torch[0,0,:5]}, TTNN: {cos_ttnn_torch[0,0,:5]}")
    print(f"Sin[0,0,:5] - Torch: {sin_torch[0,0,:5]}, TTNN: {sin_ttnn_torch[0,0,:5]}")
    print(f"Cos[0,7,:5] - Torch: {cos_torch[0,7,:5]}, TTNN: {cos_ttnn_torch[0,7,:5]}")
    print(f"Sin[0,7,:5] - Torch: {sin_torch[0,7,:5]}, TTNN: {sin_ttnn_torch[0,7,:5]}")
    
    # Check max difference
    cos_diff = torch.abs(cos_torch - cos_ttnn_torch)
    sin_diff = torch.abs(sin_torch - sin_ttnn_torch)
    print(f"Cos max diff: {cos_diff.max():.6f}, mean diff: {cos_diff.mean():.6f}")
    print(f"Sin max diff: {sin_diff.max():.6f}, mean diff: {sin_diff.mean():.6f}")

    q_out_ttnn, k_out_ttnn = apply_rotary_pos_emb_ttnn(
        q_ttnn, k_ttnn, cos_ttnn, sin_ttnn
    )

    # -----------------------
    # Back to Torch
    # -----------------------
    q_out = ttnn.to_torch(q_out_ttnn)
    k_out = ttnn.to_torch(k_out_ttnn)

    # -----------------------
    # PCC checks
    # -----------------------
    q_pcc = pcc(q_ref, q_out)
    k_pcc = pcc(k_ref, k_out)

    print(f"Q PCC: {q_pcc:.6f}")
    print(f"K PCC: {k_pcc:.6f}")

    assert q_pcc > 0.9999, "Q RoPE PCC failed"
    assert k_pcc > 0.9999, "K RoPE PCC failed"

    print("✅ RoPE PCC test PASSED")

if __name__ == "__main__":
    main()