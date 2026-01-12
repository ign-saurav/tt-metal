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
    cond_dim = 16
    eps = 1e-6

    # -------------------------
    # PyTorch reference
    # -------------------------
    x_torch = torch.randn(B, T, D)
    cond_torch = torch.randn(B, cond_dim)

    torch_rms = GemmaRMSNorm(dim=D, eps=eps, cond_dim=cond_dim)

    with torch.no_grad():
        out_ref, gate_ref = torch_rms(x_torch, cond_torch)

    # -------------------------
    # TTNN
    # -------------------------
    device = ttnn.open_device(device_id=0)

    x_ttnn = ttnn.from_torch(x_torch, device=device)
    cond_ttnn = ttnn.from_torch(cond_torch, device=device)

    w_ttnn = ttnn.from_torch(torch_rms.dense.weight, device=device)
    b_ttnn = ttnn.from_torch(torch_rms.dense.bias, device=device)

    ttnn_rms = GemmaRMSNormTTNN(
        dim=D,
        eps=eps,
        dense_weight=w_ttnn,
        dense_bias=b_ttnn,
    )

    out_ttnn, gate_ttnn = ttnn_rms.forward(x_ttnn, cond_ttnn)

    out_ttnn_torch = ttnn.to_torch(out_ttnn)
    gate_ttnn_torch = ttnn.to_torch(gate_ttnn)

    # -------------------------
    # PCC checks
    # -------------------------
    print("Output PCC:", pcc(out_ref, out_ttnn_torch))
    print("Gate PCC:", pcc(gate_ref, gate_ttnn_torch))


if __name__ == "__main__":
    main()
