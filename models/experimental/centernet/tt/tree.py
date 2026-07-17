# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
    MaxPool2dConfiguration,
    HeightShardedStrategyConfiguration,
)
from models.experimental.centernet.tt.root import TtRoot
from models.experimental.centernet.tt.basic_block import TtBasicBlock
from models.common.lightweightmodule import LightweightModule


class TtTree(LightweightModule):
    def __init__(
        self,
        levels: int,
        block: TtBasicBlock,
        in_channels: int,
        out_channels: int,
        parameters,
        device,
        layer_args,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
    ):
        super(TtTree, self).__init__()
        self.device = device
        self.levels = levels
        self.level_root = level_root
        self.root_dim = root_dim
        self.stride = stride

        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(
                inplanes=in_channels,
                planes=out_channels,
                stride=stride,
                dilation=dilation,
                parameters=parameters.tree1,
                device=device,
                layer_args=layer_args.tree1,
            )
            self.tree2 = block(
                inplanes=out_channels,
                planes=out_channels,
                stride=1,
                dilation=dilation,
                parameters=parameters.tree2,
                device=device,
                layer_args=layer_args.tree2,
            )
        else:
            self.tree1 = TtTree(
                levels=levels - 1,
                block=block,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                level_root=False,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                parameters=parameters.tree1,
                device=device,
                layer_args=layer_args.tree1,
            )
            self.tree2 = TtTree(
                levels=levels - 1,
                block=block,
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                level_root=False,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                parameters=parameters.tree2,
                device=device,
                layer_args=layer_args.tree2,
            )

        if levels == 1:
            self.root = TtRoot(
                in_channels=root_dim,
                out_channels=out_channels,
                kernel_size=root_kernel_size,
                residual=root_residual,
                parameters=parameters.root,
                layer_args=layer_args.root,
                device=device,
            )

        self.downsample = None
        if stride > 1:
            self.downsample = TtMaxPool2d(
                MaxPool2dConfiguration(
                    input_height=layer_args.downsample.input_height,
                    input_width=layer_args.downsample.input_width,
                    channels=in_channels,
                    batch_size=layer_args.downsample.batch_size,
                    kernel_size=(stride, stride),
                    stride=(stride, stride),
                ),
                device,
            )

        self.project = None
        if in_channels != out_channels:
            self.project = TtConv2d(
                self._make_config(
                    parameters.project,
                    layer_args.project["0"].batch_size,
                    layer_args.project["0"].input_height,
                    layer_args.project["0"].input_width,
                    in_channels,
                    out_channels,
                    stride=1,
                    dilation=1,
                    padding=0,
                    activation=None,
                    is_project=True,
                ),
                device,
            )

    def _make_config(self, params, bs, h, w, in_ch, out_ch, stride, dilation, padding, activation, is_project=False):
        weight = params.weight
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)

        bias = getattr(params, "bias", None)
        if bias is not None and isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
            bias = ttnn.from_device(bias)

        return Conv2dConfiguration(
            input_height=h,
            input_width=w,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=bs,
            kernel_size=(1, 1) if is_project else (3, 3),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            deallocate_activation=False,
        )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children

        bottom = self.downsample(x) if self.downsample else x

        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x
