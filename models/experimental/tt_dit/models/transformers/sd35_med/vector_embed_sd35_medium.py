# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
import torch
from models.experimental.tt_dit.layers.linear import Linear


class VectorEmbedder(nn.Module):
    """TTNN implementation of VectorEmbedder"""

    def __init__(self, input_dim: int, hidden_size: int, mesh_device=None, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device

        self.linear1 = Linear(
            in_features=input_dim,
            out_features=hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )
        self.linear2 = Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Ensure input is 3D for TTNN matmul: [B, 1, F]
        if len(x.shape) == 2:
            x = ttnn.unsqueeze(x, 1)

        # Linear → SiLU → Linear
        x = self.linear1(x)
        x = ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = self.linear2(x)

        return x

    def to_cached_state_dict(self, path_prefix):
        """Convert embedder state to cached state dict"""
        cache_dict = {}
        if hasattr(self.linear1, "to_cached_state_dict"):
            linear1_cache = self.linear1.to_cached_state_dict(path_prefix + "linear1.")
            # linear1_cache already has keys like "weight", "bias" with full paths as values
            # We need to add "linear1." prefix to the keys
            for k, v in linear1_cache.items():
                cache_dict[f"linear1.{k}"] = v
        if hasattr(self.linear2, "to_cached_state_dict"):
            linear2_cache = self.linear2.to_cached_state_dict(path_prefix + "linear2.")
            for k, v in linear2_cache.items():
                cache_dict[f"linear2.{k}"] = v
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        """Load embedder state from cached state dict"""

        def substate(state, key):
            prefix = f"{key}."
            result = {}
            for k, v in state.items():
                if k.startswith(prefix):
                    # Remove the prefix from the key
                    new_key = k[len(prefix) :]
                    result[new_key] = v
            return result

        if hasattr(self.linear1, "from_cached_state_dict"):
            linear1_dict = substate(cache_dict, "linear1")
            if linear1_dict:  # Only call if we have keys
                self.linear1.from_cached_state_dict(linear1_dict)
        if hasattr(self.linear2, "from_cached_state_dict"):
            linear2_dict = substate(cache_dict, "linear2")
            if linear2_dict:  # Only call if we have keys
                self.linear2.from_cached_state_dict(linear2_dict)
