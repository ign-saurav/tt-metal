# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
"""MMDetection provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry("model", parent=MMENGINE_MODELS, locations=["mmdet.models"])

CONV_LAYERS = Registry("conv layer")
NORM_LAYERS = Registry("norm layer")
ACTIVATION_LAYERS = Registry("activation layer")
PADDING_LAYERS = Registry("padding layer")
UPSAMPLE_LAYERS = Registry("upsample layer")
PLUGIN_LAYERS = Registry("plugin layer")

DROPOUT_LAYERS = Registry("drop out layers")
POSITIONAL_ENCODING = Registry("position encoding")
ATTENTION = Registry("attention")
FEEDFORWARD_NETWORK = Registry("feed-forward Network")
TRANSFORMER_LAYER = Registry("transformerLayer")
TRANSFORMER_LAYER_SEQUENCE = Registry("transformer-layers sequence")
