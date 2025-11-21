# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from .registry import CONV_LAYERS

CONV_LAYERS.register_module("Conv1d", module=nn.Conv1d)
CONV_LAYERS.register_module("Conv2d", module=nn.Conv2d)
CONV_LAYERS.register_module("Conv3d", module=nn.Conv3d)
CONV_LAYERS.register_module("Conv", module=nn.Conv2d)


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

    # Fix Me: Used conv2d for DCN, DCNv2, DeformConv2d, ModulatedDeformConv2d
    # print(f"layer_type: {layer_type}")
    # print(f"CONV_LAYERS: {CONV_LAYERS}")
    # if layer_type in ('DCN', 'DCNv2', 'DeformConv2d', 'ModulatedDeformConv2d'):
    #     # Fallback to regular Conv2d (no CUDA required)
    #     cfg.pop('deform_groups', None)
    #     cfg.pop('fallback_on_stride', None)
    #     layer = nn.Conv2d
    # if layer_type not in CONV_LAYERS:
    #     raise KeyError(f'Unrecognized layer type {layer_type}')
    # else:
    #     conv_layer = CONV_LAYERS.get(layer_type)
    # Handle DCN types (fallback to Conv2d)
    if layer_type in ("DCN", "DCNv2", "DeformConv2d", "ModulatedDeformConv2d"):
        # Remove DCN-specific parameters
        cfg_.pop("deform_groups", None)
        cfg_.pop("fallback_on_stride", None)
        cfg_.pop("im2col_step", None)
        cfg_.pop("with_modulated_dcn", None)
        conv_layer = nn.Conv2d

    # Handle regular conv types from registry
    elif layer_type in CONV_LAYERS:
        conv_layer = CONV_LAYERS.get(layer_type)

    # Unknown type
    else:
        raise KeyError(f"Unrecognized layer type {layer_type}")

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
