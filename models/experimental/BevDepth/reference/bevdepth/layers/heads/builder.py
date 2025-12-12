# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc4/mmdet3d/models/builder.py
# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Any


# Create simple registries
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self, name=None, force=False, module=None):
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
        if cfg is None:
            return None
        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"{obj_type} not found. Available: {list(self._module_dict.keys())}")
        obj_cls = self._module_dict[obj_type]
        return obj_cls(**cfg)


# Create registries
MODELS = Registry("models")
TASK_UTILS = Registry("task util")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")

# Alias for compatibility
MMCV_MODELS = MODELS


def build_backbone(cfg):
    """Build backbone from config dict"""
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
    """Build neck from config dict"""
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
    """Build head from config dict"""
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
