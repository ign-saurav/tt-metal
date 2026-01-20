import ttnn
import torch
from torch import nn
from typing import Optional

class GemmaRMSNormTTNN:
    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        weight: ttnn.Tensor | None = None,          # [D]
        dense_weight: ttnn.Tensor | None = None,    # [cond_dim, 3D]
        dense_bias: ttnn.Tensor | None = None,      # [3D]
    ):
        self.dim = dim
        self.eps = eps

        self.weight = weight
        self.dense_weight = dense_weight
        self.dense_bias = dense_bias

        self.is_adaptive = dense_weight is not None

    def _norm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_sq = ttnn.mul(x, x)
        var = ttnn.mean(x_sq, dim=-1, keepdim=True)
        inv_std = ttnn.rsqrt(ttnn.add(var, self.eps))
        return ttnn.mul(x, inv_std)

    def forward(self, x: ttnn.Tensor, cond: ttnn.Tensor | None = None):
        normed = self._norm(x)

        rank = ttnn.get_rank(x)

        # -------------------------
        # Regular RMSNorm
        # -------------------------
        if not self.is_adaptive:
            if cond is not None:
                raise ValueError("cond provided but RMSNorm is not adaptive")

            # reshape weight for broadcasting
            if rank == 3:
                scale = ttnn.reshape(self.weight, (1, 1, self.dim))
            else:
                scale = ttnn.reshape(self.weight, (1, self.dim))

            one = ttnn.ones_like(scale)
            out = ttnn.mul(normed, ttnn.add(scale, one))
            return out, None

        # -------------------------
        # Adaptive RMSNorm
        # -------------------------
        if cond is None:
            raise ValueError("Adaptive RMSNorm requires cond input")

        modulation = ttnn.linear(cond, self.dense_weight, self.dense_bias)
        # [B, 3D]

        if rank == 3:
            modulation = ttnn.unsqueeze(modulation, dim=1)  # [B, 1, 3D]

        scale, shift, gate = ttnn.split(modulation, 3, dim=-1)

        out = ttnn.mul(normed, ttnn.add(scale, 1.0))
        out = ttnn.add(out, shift)

        return out, gate

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        
        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            #self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)
        
        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate
        
        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")
        
        #self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)
        
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        
        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype
        
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str


