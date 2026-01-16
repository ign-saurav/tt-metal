# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

########################################################
# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc4/mmdet3d/models/dense_heads/centerpoint_head.py
# Copyright (c) OpenMMLab. All rights reserved.
########################################################
import copy
from typing import List, Optional, Tuple, Union

from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor, nn
from functools import partial

from models.experimental.BevDepth.reference.builder import MODELS


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@MODELS.register_module()
class SeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels. Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer. Default: 1.
        init_bias (float, optional): Initial bias. Default: -2.19.
        conv_cfg (dict, optional): Config of conv layer. Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer. Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        init_bias=-2.19,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        init_cfg=None,
        **kwargs,
    ):
        assert init_cfg is None, "To prevent abnormal initialization behavior, init_cfg is not allowed to be set"
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]
            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                    )
                )
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == "heatmap":
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains reg, height, dim, rot, vel, heatmap.
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@MODELS.register_module()
class CenterHead(BaseModule):
    """CenterHead for CenterPoint (inference only).

    Args:
        in_channels (list[int] | int, optional): Channels of the input feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number and class names.
        common_heads (dict, optional): Conv information for common heads. Default: dict().
        separate_head (dict, optional): Config of separate head.
        share_conv_channel (int, optional): Output channels for share_conv layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap. Default: 2.
        conv_cfg (dict, optional): Config of conv layer. Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer. Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
        norm_bbox (bool): Whether normalize the bbox predictions. Defaults to True.
    """

    def __init__(
        self,
        in_channels: Union[List[int], int] = [128],
        tasks: Optional[List[dict]] = None,
        bbox_coder: Optional[dict] = None,
        common_heads: dict = dict(),
        loss_cls: dict = dict(type="mmdet.GaussianFocalLoss", reduction="mean"),
        loss_bbox: dict = dict(type="mmdet.L1Loss", reduction="none", loss_weight=0.25),
        separate_head: dict = dict(type="mmdet.SeparateHead", init_bias=-2.19, final_kernel=3),
        share_conv_channel: int = 64,
        num_heatmap_convs: int = 2,
        conv_cfg: dict = dict(type="Conv2d"),
        norm_cfg: dict = dict(type="BN2d"),
        bias: str = "auto",
        norm_bbox: bool = True,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ):
        assert init_cfg is None, "To prevent abnormal initialization behavior, init_cfg is not allowed to be set"
        super(CenterHead, self).__init__(init_cfg=init_cfg, **kwargs)

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.num_anchor_per_locs = [n for n in num_classes]

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels, share_conv_channel, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias
        )

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(MODELS.build(separate_head))

    def forward_single(self, x: Tensor) -> dict:
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g., features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)
