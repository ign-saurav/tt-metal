"""Configuration helpers for the BEVFormerV2 TT backbone.

Provides:
- **Global defaults** for convolution parameters (dtypes, layout, etc.)
- **Per-layer overrides** addressed by a simple string path
  (for example: ``stem.conv1``, ``layer2.0.conv2``, ``layer4.2.conv3``).

The goal is to make the BEVFormerV2 backbone configurable without changing
the existing call‑sites too much. If no config object is provided, the
current hard‑coded behaviour is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import ttnn


@dataclass
class BevFormerV2ConvDefaults:
    """Default settings that apply to all convolutions unless overridden."""

    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    shard_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    deallocate_activation: bool = False
    act_block_h: Optional[int] = None
    # Low-level kernel / conv configuration – kept minimal but overridable.
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True
    math_approx_mode: bool = False
    enable_act_double_buffer: bool = False
    reshard_if_not_optimal: bool = True


@dataclass
class BevFormerV2ModelConfig:
    """Configuration object for BEVFormerV2 TT backbone.

    Example usage:

    .. code-block:: python

        from models.experimental.bevformerv2.tt.model_configs import BevFormerV2ModelConfig

        model_cfg = BevFormerV2ModelConfig()
        # Make layer2 use bfloat8 activations and block sharding on conv3
        model_cfg.register_layer_override(
            "layer2.0.conv3",
            activation_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        backbone = TtResNet50_MMD_C345(conv_args, conv_pth, device, model_configs=model_cfg)
    """

    defaults: BevFormerV2ConvDefaults = field(default_factory=BevFormerV2ConvDefaults)
    # Map from "layer path" -> per-layer overrides
    layer_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register_layer_override(self, layer_path: str, **overrides: Any) -> None:
        """Register overrides for a specific layer.

        Parameters
        ----------
        layer_path:
            A string identifying the logical layer, e.g.:
            - ``"stem.conv1"``
            - ``"layer2.0.conv2"``
            - ``"layer4.2.downsample"``
        overrides:
            Keyword arguments matching fields from :class:`BevFormerV2ConvDefaults`,
            e.g. ``activation_dtype``, ``weights_dtype``, ``shard_layout``,
            ``deallocate_activation``, ``act_block_h``, ``math_fidelity``,
            ``fp32_dest_acc_en``, ``packer_l1_acc``, ``math_approx_mode``,
            ``enable_act_double_buffer``, ``reshard_if_not_optimal``.
        """

        if layer_path not in self.layer_overrides:
            self.layer_overrides[layer_path] = {}
        self.layer_overrides[layer_path].update(overrides)

    # --------------------------------------------------------------------- #
    # Query helpers used by create_conv2d_configuration
    # --------------------------------------------------------------------- #

    def get_effective_conv_settings(self, layer_path: Optional[str]) -> BevFormerV2ConvDefaults:
        """Return the effective convolution settings for a given layer.

        If ``layer_path`` is ``None`` or no overrides are registered, the
        global defaults are returned.
        """

        if not layer_path or layer_path not in self.layer_overrides:
            return self.defaults

        # Create a shallow copy of defaults and apply overrides on top
        overrides = self.layer_overrides[layer_path]
        return BevFormerV2ConvDefaults(
            activation_dtype=overrides.get("activation_dtype", self.defaults.activation_dtype),
            weights_dtype=overrides.get("weights_dtype", self.defaults.weights_dtype),
            shard_layout=overrides.get("shard_layout", self.defaults.shard_layout),
            deallocate_activation=overrides.get(
                "deallocate_activation",
                self.defaults.deallocate_activation,
            ),
            act_block_h=overrides.get("act_block_h", self.defaults.act_block_h),
            math_fidelity=overrides.get("math_fidelity", self.defaults.math_fidelity),
            fp32_dest_acc_en=overrides.get("fp32_dest_acc_en", self.defaults.fp32_dest_acc_en),
            packer_l1_acc=overrides.get("packer_l1_acc", self.defaults.packer_l1_acc),
            math_approx_mode=overrides.get("math_approx_mode", self.defaults.math_approx_mode),
            enable_act_double_buffer=overrides.get(
                "enable_act_double_buffer",
                self.defaults.enable_act_double_buffer,
            ),
            reshard_if_not_optimal=overrides.get(
                "reshard_if_not_optimal",
                self.defaults.reshard_if_not_optimal,
            ),
        )
