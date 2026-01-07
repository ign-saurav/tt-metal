# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
########################################################
# Adapted from: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/conv.py
# Copyright (c) OpenMMLab. All rights reserved.
########################################################

from torch import nn
from models.experimental.BevDepth.reference.registry import CONV_LAYERS

CONV_LAYERS.register_module("Conv1d", module=nn.Conv1d)
CONV_LAYERS.register_module("Conv2d", module=nn.Conv2d)
CONV_LAYERS.register_module("Conv3d", module=nn.Conv3d)
CONV_LAYERS.register_module("Conv", module=nn.Conv2d)
from models.experimental.BevDepth.reference.deform_conv import DeformConv2dPack


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

    # Handle DCN types
    if layer_type in ("DCN", "DCNv2", "DeformConv2d", "ModulatedDeformConv2d"):
        deform_groups = cfg_.pop("deform_groups", 1)
        cfg_.pop("fallback_on_stride", None)
        im2col_step = cfg_.pop("im2col_step", 128)  # BEVDepth uses 128
        cfg_.pop("with_modulated_dcn", None)
        cfg_["bias"] = False
        return DeformConv2dPack(deform_groups=deform_groups, im2col_step=im2col_step, *args, **kwargs, **cfg_)

    # Handle regular conv types from registry
    elif layer_type in CONV_LAYERS:
        conv_layer = CONV_LAYERS.get(layer_type)

    # Unknown type
    else:
        raise KeyError(f"Unrecognized layer type {layer_type}")

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
