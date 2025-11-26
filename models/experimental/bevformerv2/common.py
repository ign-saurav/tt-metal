"""
Utility helpers for working with BEVFormerV2 experiment assets.
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.nn import Module


_BEVFORMERV2_ROOT = Path(__file__).resolve().parent
_DEMO_DIR = _BEVFORMERV2_ROOT / "demo"
_DEFAULT_RESNET50_BACKBONE = _DEMO_DIR / "resnet50_backbone.pth"
_DEFAULT_FPN_WEIGHTS = _DEMO_DIR / "fpn_weights.pth"


def _ensure_checkpoint_path(path: Optional[Path], default_path: Path) -> Path:
    resolved_path = Path(path) if path is not None else default_path
    resolved_path = resolved_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_path}")
    return resolved_path


def _extract_state_dict(
    checkpoint_payload,
    state_dict_key: Optional[str] = None,
) -> Mapping[str, torch.Tensor]:
    if state_dict_key is not None:
        if not isinstance(checkpoint_payload, Mapping) or state_dict_key not in checkpoint_payload:
            raise KeyError(f"Key '{state_dict_key}' not found in checkpoint payload.")
        return checkpoint_payload[state_dict_key]

    if isinstance(checkpoint_payload, Mapping):
        for candidate in ("state_dict", "model", "module"):
            maybe_state_dict = checkpoint_payload.get(candidate)
            if isinstance(maybe_state_dict, Mapping):
                return maybe_state_dict
        return checkpoint_payload

    if hasattr(checkpoint_payload, "state_dict"):
        return checkpoint_payload.state_dict()

    raise ValueError("Unable to extract a state_dict from the provided checkpoint payload.")


def _strip_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Optional[Iterable[str]] = None,
) -> Mapping[str, torch.Tensor]:
    if not prefixes:
        return state_dict

    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if prefix and new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        stripped[new_key] = value
    return stripped


def load_module_weights(
    module: Module,
    *,
    checkpoint_path: Optional[Path] = None,
    default_checkpoint_path: Path = _DEFAULT_RESNET50_BACKBONE,
    map_location: str | torch.device = "cpu",
    state_dict_key: Optional[str] = None,
    strip_prefixes: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Load weights from a checkpoint file into the provided module.

    Returns a tuple of (missing_keys, unexpected_keys) from load_state_dict so
    callers can assert on them if needed.
    """

    checkpoint_file = _ensure_checkpoint_path(checkpoint_path, default_checkpoint_path)
    payload = torch.load(checkpoint_file, map_location=map_location)
    state_dict = _extract_state_dict(payload, state_dict_key=state_dict_key)
    state_dict = _strip_prefixes(state_dict, prefixes=strip_prefixes)
    load_result = module.load_state_dict(state_dict, strict=strict)
    return load_result


def load_resnet50_backbone_weights(
    model: Module,
    *,
    checkpoint_path: Optional[Path] = None,
    map_location: str | torch.device = "cpu",
    state_dict_key: Optional[str] = None,
    strip_prefixes: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Convenience helper to load the demo ResNet-50 backbone weights.
    """

    return load_module_weights(
        model,
        checkpoint_path=checkpoint_path,
        default_checkpoint_path=_DEFAULT_RESNET50_BACKBONE,
        map_location=map_location,
        state_dict_key=state_dict_key,
        strip_prefixes=strip_prefixes,
        strict=strict,
    )


def load_fpn_weights(
    model: Module,
    *,
    checkpoint_path: Optional[Path] = None,
    map_location: str | torch.device = "cpu",
    state_dict_key: Optional[str] = None,
    strip_prefixes: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Convenience helper to load the demo FPN weights.
    """

    return load_module_weights(
        model,
        checkpoint_path=checkpoint_path,
        default_checkpoint_path=_DEFAULT_FPN_WEIGHTS,
        map_location=map_location,
        state_dict_key=state_dict_key,
        strip_prefixes=strip_prefixes,
        strict=strict,
    )
