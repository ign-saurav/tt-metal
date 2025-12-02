# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Final Layer Implementation
Final layer with normalization, modulation, and projection.
"""

import ttnn
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.normalization import LayerNorm
from models.experimental.tt_dit.utils.substate import substate


class SD35MediumFinalLayer:
    """Final layer for SD3.5 Medium with normalization and modulation."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: int = None,
        mesh_device=None,
        parallel_config=None,
    ):
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.mesh_device = mesh_device

        # Final normalization
        self.norm_final = LayerNorm(
            embedding_dim=hidden_size,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

        # Output projection
        if total_out_channels is None:
            output_dim = patch_size * patch_size * out_channels
        else:
            output_dim = total_out_channels

        self.linear = Linear(
            in_features=hidden_size,
            out_features=output_dim,
            bias=True,
            mesh_device=mesh_device,
        )

        # AdaLN modulation: SiLU + Linear
        self.adaLN_modulation_silu = lambda x: ttnn.silu(x)
        self.adaLN_modulation = Linear(
            in_features=hidden_size,
            out_features=2 * hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

        # Compute config for precision
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def modulate(self, x, shift, scale):
        """Apply modulation: x * (1 + scale) + shift"""
        if shift is None:
            shift = ttnn.zeros_like(scale)

        # Ensure proper broadcasting
        # x is [1, B, seq_len, hidden_size]
        # scale/shift are [1, B, hidden_size]
        # Need to unsqueeze at dimension 2 (seq_len) to get [1, B, 1, hidden_size]
        scale_expanded = ttnn.unsqueeze(scale, 2)
        shift_expanded = ttnn.unsqueeze(shift, 2)

        return x * (1 + scale_expanded) + shift_expanded

    def __call__(self, x, c):
        """
        Forward pass for final layer.
        Args:
            x: [1, B, seq_len, hidden_size]
            c: [1, B, hidden_size]
        Returns:
            output: [1, B, seq_len, output_dim]
        """
        # Apply SiLU activation to conditioning
        c_activated = self.adaLN_modulation_silu(c)

        # Get shift and scale from modulation
        modulation = self.adaLN_modulation(c_activated)
        # shift, scale = ttnn.split(modulation, dim=-1)
        shift, scale = ttnn.chunk(modulation, 2, dim=-1)

        # Apply final normalization
        x_norm = self.norm_final(x)

        # Apply modulation
        x_modulated = self.modulate(x_norm, shift, scale)

        # Final linear projection
        output = self.linear(x_modulated)

        return output

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load normalization weights (should be empty for elementwise_affine=False)
        self.norm_final.load_torch_state_dict(substate(state_dict, "norm_final"))

        # Load linear weights
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))

        # Load AdaLN modulation weights
        adaLN_state_dict = substate(state_dict, "adaLN_modulation")
        if adaLN_state_dict:
            # Strip the "1." prefix if present (from Sequential)
            normalized_adaLN_state_dict = {}
            for k, v in adaLN_state_dict.items():
                if k.startswith("1."):
                    normalized_adaLN_state_dict[k[2:]] = v  # Remove "1." prefix
                else:
                    normalized_adaLN_state_dict[k] = v
            self.adaLN_modulation.load_torch_state_dict(normalized_adaLN_state_dict)
        else:
            self.adaLN_modulation.load_torch_state_dict({})

    def to_cached_state_dict(self, path_prefix):
        """Convert final layer state to cached state dict"""
        cache_dict = {}

        # Cache norm_final (if it has parameters)
        if hasattr(self.norm_final, "to_cached_state_dict"):
            norm_cache = self.norm_final.to_cached_state_dict(path_prefix + "norm_final.")
            for key, value in norm_cache.items():
                cache_dict[f"norm_final.{key}"] = value

        # Cache linear
        if hasattr(self.linear, "to_cached_state_dict"):
            linear_cache = self.linear.to_cached_state_dict(path_prefix + "linear.")
            for key, value in linear_cache.items():
                cache_dict[f"linear.{key}"] = value

        # Cache adaLN_modulation
        if hasattr(self.adaLN_modulation, "to_cached_state_dict"):
            adaLN_cache = self.adaLN_modulation.to_cached_state_dict(path_prefix + "adaLN_modulation.")
            for key, value in adaLN_cache.items():
                cache_dict[f"adaLN_modulation.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        """Load final layer state from cached state dict"""
        from loguru import logger

        def substate(state, key):
            prefix = f"{key}."
            result = {}
            for k, v in state.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix) :]
                    result[new_key] = v
            return result

        logger.debug(f"FinalLayer.from_cached_state_dict: Available keys: {list(cache_dict.keys())}")

        # Load norm_final
        if hasattr(self.norm_final, "from_cached_state_dict"):
            norm_dict = substate(cache_dict, "norm_final")
            if norm_dict:
                self.norm_final.from_cached_state_dict(norm_dict)
            else:
                logger.warning("FinalLayer: No keys found for norm_final")

        # Load linear
        if hasattr(self.linear, "from_cached_state_dict"):
            linear_dict = substate(cache_dict, "linear")
            if linear_dict:
                self.linear.from_cached_state_dict(linear_dict)
            else:
                logger.warning("FinalLayer: No keys found for linear")

        # Load adaLN_modulation
        if hasattr(self.adaLN_modulation, "from_cached_state_dict"):
            adaLN_dict = substate(cache_dict, "adaLN_modulation")
            if adaLN_dict:
                self.adaLN_modulation.from_cached_state_dict(adaLN_dict)
            else:
                logger.warning("FinalLayer: No keys found for adaLN_modulation")
