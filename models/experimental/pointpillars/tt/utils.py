# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtPointPillarsConv2D:
    def __init__(
        self,
        parameters,
        conv,
        device,
        cache={},
        activation=None,
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        groups=1,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
        output_dtype=ttnn.bfloat16,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch_size = 1
        self.conv_params = conv
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.deallocate_activation = deallocate_activation
        self.output_dtype = output_dtype
        self.cache = cache
        self.parameters = parameters
        self.shard_layout = shard_layout
        self.output_layout = output_layout
        self.dilation = dilation
        self.weights_dtype = weights_dtype
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters["weight"], self.parameters["bias"]
        self.output_shape = conv.output_shape

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=True,
        )

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x):
        [x, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            memory_config=None,
            dtype=self.output_dtype,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )

        return x
