# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from .registry import CONV_LAYERS

CONV_LAYERS.register_module("Conv1d", module=nn.Conv1d)
CONV_LAYERS.register_module("Conv2d", module=nn.Conv2d)
CONV_LAYERS.register_module("Conv3d", module=nn.Conv3d)
CONV_LAYERS.register_module("Conv", module=nn.Conv2d)

# Deformable Convolution Support
# Similar to uniad/vadv2: prioritize torchvision (no compiled extensions needed)
# This avoids heavy CUDA dependencies while maintaining functionality
_TORCHVISION_DCN_AVAILABLE = False
_MMCV_DCN_AVAILABLE = False
DeformConv2dPack = None
DeformConv2d = None
ModulatedDeformConv2d = None
mmcv_build_conv_layer = None

# Try torchvision first (no compiled extensions needed, works like uniad/vadv2)
try:
    # Import our standalone DCN implementation that uses torchvision
    from .deform_conv import DeformConv2dPack

    _TORCHVISION_DCN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_DCN_AVAILABLE = False
    DeformConv2dPack = None

# Fallback to MMCV (requires compiled CUDA extensions)
# Only use if torchvision is not available
if not _TORCHVISION_DCN_AVAILABLE:
    try:
        from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
        from mmcv.cnn import build_conv_layer as mmcv_build_conv_layer

        # Test if DCN actually works by checking if the classes are available
        if DeformConv2d is not None and ModulatedDeformConv2d is not None:
            _MMCV_DCN_AVAILABLE = True
    except (ImportError, AssertionError, AttributeError, ModuleNotFoundError):
        # MMCV not installed, extensions not compiled, or incompatible version
        _MMCV_DCN_AVAILABLE = False
        DeformConv2d = None
        ModulatedDeformConv2d = None
        mmcv_build_conv_layer = None


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Handle DCN types - prioritize torchvision (like uniad/vadv2), then MMCV, then fallback to Conv2d
    if layer_type in ("DCN", "DCNv2", "DeformConv2d", "ModulatedDeformConv2d"):
        if _TORCHVISION_DCN_AVAILABLE and DeformConv2dPack is not None:
            # Primary option: Use torchvision's deform_conv2d (no compiled extensions needed)
            # This matches the approach used in uniad/vadv2 models
            # Extract DCN parameters
            deform_groups = cfg_.pop("deform_groups", 1)
            cfg_.pop("fallback_on_stride", None)
            im2col_step = cfg_.pop("im2col_step", 32)
            cfg_.pop("with_modulated_dcn", None)
            # Set bias=False (DCN doesn't support bias)
            cfg_["bias"] = False
            # Create DCN layer using torchvision backend
            return DeformConv2dPack(deform_groups=deform_groups, im2col_step=im2col_step, *args, **kwargs, **cfg_)
        elif _MMCV_DCN_AVAILABLE and mmcv_build_conv_layer is not None:
            # Fallback: Use MMCV's build_conv_layer with compiled CUDA extensions
            # Only used if torchvision is not available
            return mmcv_build_conv_layer(cfg, *args, **kwargs)
        else:
            # Last resort: Fallback to regular Conv2d
            # Note: This will NOT match the original model's behavior exactly
            import warnings

            warnings.warn(
                f"DCN layer type '{layer_type}' requested but neither torchvision deform_conv2d "
                f"nor MMCV DCN is available. Falling back to regular Conv2d. "
                f"This may cause accuracy differences. "
                f"To get correct behavior, install torchvision (pip install torchvision)"
            )
            # Remove DCN-specific parameters
            cfg_.pop("deform_groups", None)
            cfg_.pop("fallback_on_stride", None)
            cfg_.pop("im2col_step", None)
            cfg_.pop("with_modulated_dcn", None)
            # Set bias=False to match typical DCN configuration
            if "bias" not in cfg_:
                cfg_["bias"] = False
            conv_layer = nn.Conv2d

    # Handle regular conv types from registry
    elif layer_type in CONV_LAYERS:
        conv_layer = CONV_LAYERS.get(layer_type)

    # Unknown type
    else:
        raise KeyError(f"Unrecognized layer type {layer_type}")

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
