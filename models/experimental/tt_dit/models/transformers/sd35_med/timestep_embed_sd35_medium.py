# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium TimeStepEmbedder Implementation

This module implements TimeStepEmbedder for
SD3.5 Medium, matching the reference implementation.
"""

import math
import torch
import ttnn
import torch.nn as nn

from models.experimental.tt_dit.layers.linear import Linear


class TimestepEmbedder(nn.Module):
    """
    TTNN implementation of timestep embedding
    """

    def __init__(self, hidden_size, dtype=torch.bfloat16, frequency_embedding_size=256, mesh_device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mesh_device = mesh_device

        # Two Linear layers same as reference model
        self.linear1 = Linear(
            in_features=frequency_embedding_size,
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

    def forward(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """
        t: TTNN Tensor of shape [B] (bf16)
        """
        # Convert to torch for sinusoidal embedding
        t_torch = ttnn.to_torch(t)
        B = t_torch.shape[0]
        half = self.frequency_embedding_size // 2

        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t_torch.device)

        args = t_torch[:, None].float() * freqs[None]
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embed = embed.to(torch.bfloat16)

        # Convert back into TTNN with TILE layout
        x = ttnn.from_torch(
            embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

        # Ensure padded dims exist for matmul => [B, Freq] → [B, 1, Freq]
        if len(x.shape) == 2:
            x = ttnn.unsqueeze(x, 1)

        # Linear1 → SiLU → Linear2 (all TILE)
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
        from loguru import logger

        def substate(state, key):
            prefix = f"{key}."
            result = {}
            for k, v in state.items():
                if k.startswith(prefix):
                    # Remove the prefix from the key
                    new_key = k[len(prefix) :]
                    result[new_key] = v
            return result

        # Debug: log available keys
        logger.debug(f"TimestepEmbedder.from_cached_state_dict: Available keys: {list(cache_dict.keys())}")

        if hasattr(self.linear1, "from_cached_state_dict"):
            linear1_dict = substate(cache_dict, "linear1")
            logger.debug(f"TimestepEmbedder: linear1_dict keys: {list(linear1_dict.keys())}")
            if linear1_dict:  # Only call if we have keys
                self.linear1.from_cached_state_dict(linear1_dict)
            else:
                error_msg = (
                    "TimestepEmbedder: No keys found for linear1 in cache. "
                    "This may indicate the cache was created before embedder caching was implemented. "
                    "Please delete the cache directory and recreate it."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        if hasattr(self.linear2, "from_cached_state_dict"):
            linear2_dict = substate(cache_dict, "linear2")
            logger.debug(f"TimestepEmbedder: linear2_dict keys: {list(linear2_dict.keys())}")
            if linear2_dict:  # Only call if we have keys
                self.linear2.from_cached_state_dict(linear2_dict)
            else:
                error_msg = (
                    "TimestepEmbedder: No keys found for linear2 in cache. "
                    "This may indicate the cache was created before embedder caching was implemented. "
                    "Please delete the cache directory and recreate it."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
