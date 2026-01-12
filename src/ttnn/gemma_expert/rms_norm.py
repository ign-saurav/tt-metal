import ttnn
import torch
from torch import nn
from typing import Optional

class GemmaRMSNormTTNN:
    def __init__(self, dim, eps, *, weight=None, dense_weight=None, dense_bias=None):
        self.dim = dim
        self.eps = eps
        self.weight = weight
        self.dense_weight = dense_weight
        self.dense_bias = dense_bias
        self.is_adaptive = dense_weight is not None

    def _norm(self, x):
        assert x.storage_type() == ttnn.StorageType.DEVICE

        x_sq = ttnn.mul(x, x)
        var = ttnn.mean(x_sq, dim=-1, keepdim=True)

        eps_tensor = ttnn.full_like(var, self.eps)
        inv_std = ttnn.rsqrt(ttnn.add(var, eps_tensor))

        return ttnn.mul(x, inv_std)

    def forward(self, x, cond=None):
        normed = self._norm(x)

        shape = x.shape
        rank = len(shape)

        if not self.is_adaptive:
            if cond is not None:
                raise ValueError("cond provided but RMSNorm is not adaptive")

            if rank == 3:
                scale = ttnn.reshape(self.weight, (1, 1, self.dim))
            else:
                scale = ttnn.reshape(self.weight, (1, self.dim))

            one = ttnn.full_like(scale, 1.0)
            scale = ttnn.add(scale, one)

            scale = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)
            normed = ttnn.to_layout(normed, ttnn.TILE_LAYOUT)
            
            out = ttnn.mul(normed, scale)
            return out, None

        # Adaptive RMSNorm
        if cond is None:
            raise ValueError("Adaptive RMSNorm requires cond input")

        cond = ttnn.to_layout(cond, ttnn.TILE_LAYOUT)
        self.dense_weight = ttnn.to_layout(self.dense_weight, ttnn.TILE_LAYOUT)
        self.dense_bias = ttnn.to_layout(self.dense_bias, ttnn.TILE_LAYOUT)

        modulation = ttnn.linear(cond, self.dense_weight, bias=self.dense_bias, transpose_b=True)

        if rank == 3:
            modulation = ttnn.unsqueeze(modulation, dim=1)

        print("Modulation :", modulation)

        B, _, _ = modulation.shape
        scale = ttnn.slice(modulation, [0, 0, 0], [B, 1, self.dim])
        print("Scale :", scale)
        shift = ttnn.slice(modulation, [0, 0, self.dim], [B, 1, 2 * self.dim])
        gate  = ttnn.slice(modulation, [0, 0, 2 * self.dim], [B, 1, 3 * self.dim])

        scale = ttnn.add(scale, ttnn.full_like(scale, 1.0))

        B, T, _ = normed.shape

        scale = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)
        shift = ttnn.to_layout(shift, ttnn.TILE_LAYOUT)
        gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT)
        normed = ttnn.to_layout(normed, ttnn.TILE_LAYOUT)
        
        ones = ttnn.ones([B, T, 1], device=normed.device(), layout=ttnn.TILE_LAYOUT)
        scale = ttnn.matmul(ones, scale)
        
        shift = ttnn.matmul(ones, shift)
        print("Shift :", shift)

        out = ttnn.mul(normed, scale)
        out = ttnn.add(out, shift)

        print("Out :", out)

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


class GemmaRotaryEmbeddingTTNN:
    def __init__(self, config, device):
        # rope type selection (same logic)
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"

        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        # rope init (Torch side)
        rope_init_fn = _compute_default_rope_parameters
        inv_freq_torch, attention_scaling = rope_init_fn(config, device=None)

        # move constants to TTNN
        self.inv_freq = ttnn.from_torch(inv_freq_torch, device=device)
        self.attention_scaling = attention_scaling  # scalar (Python float)

        self.original_inv_freq = self.inv_freq

    def forward(self, position_ids, x_dtype):
        """
        position_ids: TTNN tensor [B, T]
        x_dtype: dtype of attention tensors (bf16/fp16)
        """

        B, T = position_ids.shape
        D_half = self.inv_freq.shape[0]
        device = position_ids.device()  # Save device before reassigning position_ids

        # [1, D/2, 1]
        inv_freq = ttnn.reshape(self.inv_freq, (1, D_half, 1))
        
        # Expand inv_freq to match batch dimension: [1, D/2, 1] -> [B, D/2, 1]
        # Convert to torch, expand, then convert back (TTNN doesn't support batch dimension broadcasting in matmul)
        # Ensure float32 for computation (matching transformers)
        inv_freq_torch = ttnn.to_torch(inv_freq).float()
        inv_freq_torch = inv_freq_torch.expand(B, D_half, 1)
        inv_freq = ttnn.from_torch(inv_freq_torch, device=device, dtype=ttnn.float32)
        inv_freq = ttnn.to_layout(inv_freq, ttnn.TILE_LAYOUT)

        # [B, 1, T] - ensure float32
        position_ids_reshaped = ttnn.reshape(position_ids, (B, 1, T))
        position_ids_torch = ttnn.to_torch(position_ids_reshaped).float()
        position_ids = ttnn.from_torch(position_ids_torch, device=device, dtype=ttnn.float32)
        position_ids = ttnn.to_layout(position_ids, ttnn.TILE_LAYOUT)

        # freqs = inv_freq @ position_ids
        # [B, D/2, T]
        freqs = ttnn.matmul(inv_freq, position_ids)
        freqs = ttnn.to_layout(freqs, ttnn.TILE_LAYOUT)

        # [B, T, D/2]
        freqs = ttnn.transpose(freqs, 1, 2)

        # duplicate frequencies → [B, T, D]
        freqs = ttnn.to_layout(freqs, ttnn.TILE_LAYOUT)
        emb = ttnn.concat([freqs, freqs], dim=-1)
        emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

        # cos / sin
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)

        # attention scaling
        if self.attention_scaling != 1.0:
            scale = ttnn.full_like(cos, self.attention_scaling)
            cos = ttnn.mul(cos, scale)
            sin = ttnn.mul(sin, scale)

        # cast back to model dtype
                # cast back to model dtype (convert to torch, cast, convert back)
        cos_torch = ttnn.to_torch(cos)
        sin_torch = ttnn.to_torch(sin)
        if x_dtype == ttnn.float32:
            torch_dtype = torch.float32
        elif x_dtype == ttnn.bfloat16:
            torch_dtype = torch.bfloat16
        elif x_dtype == ttnn.float16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32  # default
        cos_torch = cos_torch.to(dtype=torch_dtype)
        sin_torch = sin_torch.to(dtype=torch_dtype)
        cos = ttnn.from_torch(cos_torch, device=device)
        sin = ttnn.from_torch(sin_torch, device=device)
        # Ensure final output is in TILE_LAYOUT
        cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)
        return cos, sin

def rotate_half_ttnn(x):
    """
    TTNN equivalent of PyTorch rotate_half:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    Rotates half the hidden dims of the input.
    
    Args:
        x: TTNN tensor [..., D]
    Returns:
        TTNN tensor [..., D] with rotated dimensions
    """
    D = x.shape[-1]
    D_half = D // 2

    # Get shape for slice parameters
    shape = x.shape
    rank = len(shape)
    shape_list = [shape[i] for i in range(rank)]  # Convert TTNN Shape to Python list
    
    # slice last dimension: x1 = x[..., :D_half] (first half)
    start_x1 = [0] * (rank - 1) + [0]
    end_x1 = shape_list[:-1] + [D_half]
    
    # slice last dimension: x2 = x[..., D_half:] (second half)
    start_x2 = [0] * (rank - 1) + [D_half]
    end_x2 = shape_list
    
    x1 = ttnn.slice(x, start_x1, end_x1)
    x2 = ttnn.slice(x, start_x2, end_x2)
    
    # Convert to TILE_LAYOUT (required for unary operations like neg)
    x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
    x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)

    # negate x2 and concatenate: [-x2, x1]
    neg_x2 = ttnn.neg(x2)
    neg_x2 = ttnn.to_layout(neg_x2, ttnn.TILE_LAYOUT)
    result = ttnn.concat([neg_x2, x1], dim=-1)
    return ttnn.to_layout(result, ttnn.TILE_LAYOUT)

def apply_rotary_pos_emb_ttnn(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, k  : TTNN tensors
            shape either [B, H, T, D] or [B, T, H, D]
    cos,sin: TTNN tensors [B, T, D]
    """

    # Ensure all tensors are in TILE_LAYOUT for operations
    q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
    k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
    cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
    sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)

    # unsqueeze cos/sin for broadcasting
    cos = ttnn.unsqueeze(cos, dim=unsqueeze_dim)
    sin = ttnn.unsqueeze(sin, dim=unsqueeze_dim)

    # rotate q
    q_rot = rotate_half_ttnn(q)
    q_embed = ttnn.add(
        ttnn.mul(q, cos),
        ttnn.mul(q_rot, sin),
    )

    # rotate k
    k_rot = rotate_half_ttnn(k)
    k_embed = ttnn.add(
        ttnn.mul(k, cos),
        ttnn.mul(k_rot, sin),
    )

    return q_embed, k_embed

def _compute_default_rope_parameters(
    config = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,  # noqa: ARG001
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
    else:
        raise ValueError(
            "Either `config` or `rope_kwargs` must be provided to `_compute_default_rope_parameters`"
        )

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor