# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

########################################################
# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc4/mmdet3d/models/builder.py
# and https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/utils/registry.py
# Copyright (c) OpenMMLab. All rights reserved.
########################################################

from typing import Dict, Any


class Registry:
    """A simple registry to map strings to classes."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def __contains__(self, key):
        """Check if a module is registered."""
        return key in self._module_dict

    def get(self, key):
        """Get a module from the registry."""
        return self._module_dict.get(key)

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        Args:
            name (str, optional): Module name. Defaults to class name.
            force (bool): Whether to override existing. Defaults to False.
            module: The module to register. If None, returns a decorator.
        """

        def _register(cls):
            module_name = name if name else cls.__name__
            if module_name in self._module_dict and not force:
                raise KeyError(f"{module_name} already registered")
            self._module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg: Dict[str, Any]):
        """Build a module from config dict."""
        if cfg is None:
            return None
        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"{obj_type} not found. Available: {list(self._module_dict.keys())}")
        obj_cls = self._module_dict[obj_type]
        return obj_cls(**cfg)


# Model registries
MODELS = Registry("models")
TASK_UTILS = Registry("task util")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")

# Layer registries (merged from registry.py)
CONV_LAYERS = Registry("conv layer")
NORM_LAYERS = Registry("norm layer")
ACTIVATION_LAYERS = Registry("activation layer")
PADDING_LAYERS = Registry("padding layer")
UPSAMPLE_LAYERS = Registry("upsample layer")
PLUGIN_LAYERS = Registry("plugin layer")

# Alias for compatibility
MMCV_MODELS = MODELS


def build_backbone(cfg):
    """Build backbone from config dict."""
    if cfg is None:
        return None

    cfg = cfg.copy()
    obj_type = cfg.pop("type")

    # Try BACKBONES first, then MODELS
    if obj_type in BACKBONES._module_dict:
        obj_cls = BACKBONES._module_dict[obj_type]
    elif obj_type in MODELS._module_dict:
        obj_cls = MODELS._module_dict[obj_type]
    else:
        raise KeyError(
            f"{obj_type} not found in BACKBONES or MODELS. "
            f"Available in BACKBONES: {list(BACKBONES._module_dict.keys())}, "
            f"Available in MODELS: {list(MODELS._module_dict.keys())}"
        )

    return obj_cls(**cfg)


def build_neck(cfg):
    """Build neck from config dict."""
    if cfg is None:
        return None

    cfg = cfg.copy()
    obj_type = cfg.pop("type")

    # Try NECKS first, then MODELS
    if obj_type in NECKS._module_dict:
        obj_cls = NECKS._module_dict[obj_type]
    elif obj_type in MODELS._module_dict:
        obj_cls = MODELS._module_dict[obj_type]
    else:
        raise KeyError(
            f"{obj_type} not found in NECKS or MODELS. "
            f"Available in NECKS: {list(NECKS._module_dict.keys())}, "
            f"Available in MODELS: {list(MODELS._module_dict.keys())}"
        )

    return obj_cls(**cfg)


def build_head(cfg):
    """Build head from config dict."""
    if cfg is None:
        return None

    cfg = cfg.copy()
    obj_type = cfg.pop("type")

    # Try HEADS first, then MODELS
    if obj_type in HEADS._module_dict:
        obj_cls = HEADS._module_dict[obj_type]
    elif obj_type in MODELS._module_dict:
        obj_cls = MODELS._module_dict[obj_type]
    else:
        raise KeyError(
            f"{obj_type} not found in HEADS or MODELS. "
            f"Available in HEADS: {list(HEADS._module_dict.keys())}, "
            f"Available in MODELS: {list(MODELS._module_dict.keys())}"
        )

    return obj_cls(**cfg)
