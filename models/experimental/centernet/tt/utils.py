# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtConvTranspose2D(LightweightModule):
    def __init__(
        self,
        conv_transpose,
        conv_transpose_pth,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        is_dealloc_act=False,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
    ):
        super().__init__()
        self.conv_transpose = conv_transpose
        self.device = device
        self.in_channels = conv_transpose.in_channels
        self.out_channels = conv_transpose.out_channels
        self.kernel_size = conv_transpose.kernel_size
        self.stride = conv_transpose.stride
        self.padding = conv_transpose.padding
        self.output_padding = conv_transpose.output_padding

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=math_approx_mode,
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=shard_layout,
            deallocate_activation=is_dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )

        if conv_transpose_pth.bias is not None:
            self.bias = ttnn.from_device(conv_transpose_pth.bias)
        else:
            self.bias = None

        self.weight = ttnn.from_device(conv_transpose_pth.weight)
        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.memory_config = memory_config
        self._weights_prepared = False

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_height = shape[1]
            input_width = shape[2]
        else:
            batch_size = x.shape[0]
            input_height = x.shape[1]
            input_width = x.shape[2]

        [x, [_out_height, _out_width], [self.weight, self.bias]] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.conv_transpose.in_channels,
            out_channels=self.conv_transpose.out_channels,
            device=self.device,
            kernel_size=self.conv_transpose.kernel_size,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
            memory_config=self.memory_config,
            mirror_kernel=True,
        )

        shape = (batch_size, _out_height, _out_width, x.shape[-1])
        if self.reshape_output:
            x = ttnn.reshape(x, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.return_dims:
            return x, shape
        else:
            return x
