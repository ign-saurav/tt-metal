# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
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

        # Calculate root_dim if not provided
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        # Create tree branches
        if levels == 1:
            # Leaf level: use BasicBlock
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
            # Recursive level: create sub-trees
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

        # Add root module at leaf level
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

        # Optional downsample layer
        self.downsample = None
        if stride > 1:
            import pdb

            pdb.set_trace()
            self.downsample = TtMaxPool2d(
                MaxPool2dConfiguration(
                    input_height=layer_args.downsample.input_height,
                    input_width=layer_args.downsample.input_width,
                    channels=in_channels,
                    batch_size=layer_args.downsample.batch_size,
                    kernel_size=(stride, stride),
                    stride=(stride, stride),
                    # sharding_strategy=AutoShardedStrategyConfiguration(),
                ),
                device,
            )

        # Optional project layer (1x1 conv + BatchNorm)
        self.project = None
        if in_channels != out_channels:
            import pdb

            pdb.set_trace()
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
                    is_project=True,  # Add this flag
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

        # For project layer, limit core grid to avoid exceeding available cores
        if is_project:
            core_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}  # 8x8 = 64 cores max
            )
            override_sharding_config = True
        else:
            core_grid = None
            override_sharding_config = False

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
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            # core_grid=core_grid,
            # override_sharding_config=override_sharding_config,
        )

    # def forward(self, x):
    #     # Store input for residual connection if needed
    #     if self.level_root:
    #         x_residual = x

    #     # Apply projection if input/output channels differ
    #     if self.project is not None:
    #         x = self.project(x)

    #     # Apply downsampling if stride > 1
    #     if self.downsample is not None:
    #         import pdb; pdb.set_trace()
    #         x = self.downsample(x)

    #     # Forward through tree branches
    #     x1 = self.tree1(x)
    #     x2 = self.tree2(x)

    #     # Apply root at leaf level
    #     if self.levels == 1:
    #         # Concatenate branches and apply root
    #         if self.level_root:
    #             out = self.root(x1, x2, x_residual)
    #         else:
    #             out = self.root(x1, x2)
    #     else:
    #         # For non-leaf levels, just return the second branch output
    #         out = x2

    #     return out

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children

        # Apply downsampling first
        bottom = self.downsample(x) if self.downsample else x

        # Apply projection to the downsampled output
        residual = self.project(bottom) if self.project else bottom

        # Accumulate children if level_root
        if self.level_root:
            children.append(bottom)

        # Forward through tree1 with residual
        import pdb

        pdb.set_trace()
        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x
