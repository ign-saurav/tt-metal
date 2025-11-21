# # Copyright (c) OpenMMLab. All rights reserved.
# import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.utils import Registry

# MODELS = Registry('models', parent=MMCV_MODELS)

# BACKBONES = MODELS
# NECKS = MODELS
# ROI_EXTRACTORS = MODELS
# SHARED_HEADS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
# DETECTORS = MODELS


# def build_backbone(cfg):
#     """Build backbone."""
#     return BACKBONES.build(cfg)


# def build_neck(cfg):
#     """Build neck."""
#     return NECKS.build(cfg)


# def build_roi_extractor(cfg):
#     """Build roi extractor."""
#     return ROI_EXTRACTORS.build(cfg)


# def build_shared_head(cfg):
#     """Build shared head."""
#     return SHARED_HEADS.build(cfg)


# def build_head(cfg):
#     """Build head."""
#     return HEADS.build(cfg)


# def build_loss(cfg):
#     """Build loss."""
#     return LOSSES.build(cfg)


# def build_detector(cfg, train_cfg=None, test_cfg=None):
#     """Build detector."""
#     if train_cfg is not None or test_cfg is not None:
#         warnings.warn(
#             'train_cfg and test_cfg is deprecated, '
#             'please specify them in model', UserWarning)
#     assert cfg.get('train_cfg') is None or train_cfg is None, \
#         'train_cfg specified in both outer field and model field '
#     assert cfg.get('test_cfg') is None or test_cfg is None, \
#         'test_cfg specified in both outer field and model field '
#     return DETECTORS.build(
#         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


#############################

"""
Simplified builder without mmcv dependencies
"""
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


def build_model(cfg):
    """Build model from config dict"""
    return MODELS.build(cfg)


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
