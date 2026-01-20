import torch
import ttnn

# TTNN implementation
from rms_norm import GemmaRMSNormTTNN
import os

from rms_norm import GemmaRMSNorm

# -------------------------
# PCC
# -------------------------
def pcc(a: torch.Tensor, b: torch.Tensor):
    a = a.flatten()
    b = b.flatten()

    a_mean = a.mean()
    b_mean = b.mean()

    num = torch.sum((a - a_mean) * (b - b_mean))
    den = torch.sqrt(
        torch.sum((a - a_mean) ** 2) * torch.sum((b - b_mean) ** 2)
    )

    return (num / den).item()


# -------------------------
# MAIN TEST
# -------------------------
def main():
    B, T, D = 2, 4, 128
    eps = 1e-6

    # -------------------------
    # PyTorch input (GPU)
    # -------------------------
    x_torch = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    # HF Gemma RMSNorm (PyTorch reference)
    torch_rms = GemmaRMSNorm(dim=D, eps=eps).cuda()

    with torch.no_grad():
        out_ref = torch_rms.forward(x_torch)

    # -------------------------
    # TTNN input (CPU)
    # -------------------------
    x_ttnn = ttnn.from_torch(x_torch)

    w_ttnn = ttnn.from_torch(torch_rms.weight)

    ttnn_rms = GemmaRMSNormTTNN(
        dim=D,
        eps=eps,
        weight=w_ttnn,
    )

    out_ttnn, _ = ttnn_rms.forward(x_ttnn)
    out_ttnn_torch = ttnn.to_torch(out_ttnn)

    # -------------------------
    # PCC
    # -------------------------
    corr = pcc(out_ref.cpu(), out_ttnn_torch)
    print("PCC:", corr)


if __name__ == "__main__":
    main()
