# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TTConv2D:
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        parameters: dict | None = None,
        kernel_fidelity: dict | None = None,
        *,
        memory_config=None,
        act_block_h=None,
        act_block_w=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=None,
        activation="",
        groups=1,
        num_cores_nhw=None,
        is_reshape=False,
        enable_split_reader=False,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        slice_config=None,
        dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            ValueError("Invalid config")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            ValueError("Invalid config")
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            ValueError("Invalid config")
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            ValueError("Invalid config")

        self.kernel_fidelity = kernel_fidelity
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.deallocate_activation = deallocate_activation
        self.reallocate_halo_output = reallocate_halo_output
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        self.shard_layout = shard_layout
        self.slice_config = slice_config
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_split_reader = enable_split_reader
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.kernel_fidelity["ACTIVATIONS_DTYPE"]
        if weights_dtype is not None:
            self.weights_dtype = weights_dtype
        else:
            self.weights_dtype = self.kernel_fidelity["WEIGHTS_DTYPE"]
        if math_fidelity is not None:
            self.math_fidelity = math_fidelity
        else:
            self.math_fidelity = self.kernel_fidelity["MATH_FIDELITY"]

    def __call__(self, device, input_tensor, input_shape):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            in_place=True,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.kernel_fidelity["MATH_FIDELITY"],
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        if self.act_block_w is not None:
            conv_config.act_block_w_div = self.act_block_w

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_shape[-1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=input_shape[-4],
            input_height=input_shape[-3],
            input_width=input_shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=self.slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "bilinear",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(
        self,
        device,
        input_tensor,
        input_shape=None,
        reshape_output=False,
        pad_ch_to_32=False,
        sent_to_dram=False,
        dtype=ttnn.bfloat8_b,
    ):
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            output_tensor = ttnn.from_device(output_tensor)
            output_tensor = ttnn.to_dtype(output_tensor, dtype)
            output_tensor = ttnn.to_device(output_tensor, device)

            output_tensor = ttnn.reshape(
                output_tensor,
                [
                    1,
                    1,
                    output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
                    output_tensor.shape[3],
                ],
            )

        return output_tensor
