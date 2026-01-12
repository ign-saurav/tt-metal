import torch
import ttnn
from rms_norm import GemmaRMSNorm, GemmaRMSNormTTNN


def pcc(a: torch.Tensor, b: torch.Tensor):
    a = a.flatten()
    b = b.flatten()
    a_mean = a.mean()
    b_mean = b.mean()
    num = torch.sum((a - a_mean) * (b - b_mean))
    den = torch.sqrt(torch.sum((a - a_mean) ** 2) * torch.sum((b - b_mean) ** 2))
    return (num / den).item()


def main():
    B, T, D = 2, 4, 128
    eps = 1e-6

    # -------------------------
    # PyTorch reference (CPU)
    # -------------------------
    x_torch = torch.randn(B, T, D)
    torch_rms = GemmaRMSNorm(dim=D, eps=eps)

    with torch.no_grad():
        out_ref,_ = torch_rms(x_torch)

    # -------------------------
    # TTNN
    # -------------------------
    device = ttnn.open_device(device_id=0)

    x_ttnn = ttnn.from_torch(x_torch, device=device)
    w_ttnn = ttnn.from_torch(torch_rms.weight, device=device)

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
    corr = pcc(out_ref, out_ttnn_torch)
    print("PCC:", corr)


if __name__ == "__main__":
    main()
