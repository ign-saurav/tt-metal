# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
import torch


class TtPointPillarsConv2D(LightweightModule):
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        is_dealloc_act=False,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
    ):
        super().__init__()
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            # fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=True,
        )
        self.is_dealloc_act = is_dealloc_act
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.is_dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )
        if conv_pth.bias is not None:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.weight = ttnn.from_device(conv_pth.weight)
        self.memory_config = memory_config

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_height = shape[1]
            input_width = shape[2]
        else:
            batch_size = x.shape[0]
            input_height = x.shape[1]
            input_width = x.shape[2]

        [x, [_out_height, _out_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            device=self.device,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
            memory_config=self.memory_config,
        )
        shape = (batch_size, _out_height, _out_width, x.shape[-1])
        if self.reshape_output:
            x = ttnn.reshape(x, shape)
        if self.return_dims:
            return x, shape
        else:
            return x


class TtPointPillarsConv1D(LightweightModule):
    def __init__(
        self,
        conv,
        parameters,
        device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        fp32_accum=False,
        packer_l1_acc=False,
        activation=None,
        deallocate_activation=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
    ):
        super().__init__()
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        self.stride = conv.stride[0]
        self.groups = conv.groups
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            activation=activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )
        self.weight = ttnn.from_device(parameters.weight)
        self.bias = None
        if "bias" in parameters and parameters["bias"] is not None:
            bias = ttnn.from_device(parameters.bias)
            self.bias = bias
        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.memory_config = memory_config

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_length = shape[1]
        else:
            batch_size = x.shape[0]
            input_length = x.shape[1]

        [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=self.memory_config,
            dtype=self.activation_dtype,
        )
        shape = (batch_size, out_length, tt_output_tensor_on_device.shape[-1])
        if self.reshape_output:
            tt_output_tensor_on_device = ttnn.reshape(tt_output_tensor_on_device, shape)
        if self.return_dims:
            return tt_output_tensor_on_device, shape
        return tt_output_tensor_on_device


class TtPointPillarsConvTranspose2D(LightweightModule):
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
        math_fidelity=ttnn.MathFidelity.LoFi,
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

        if not self._weights_prepared:
            # First call: prepare weights on device and store them
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
            self._weights_prepared = True
        else:
            # Subsequent calls: use prepared weights, no write operation
            [x, [_out_height, _out_width]] = ttnn.conv_transpose2d(
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
                return_weights_and_bias=False,
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


def prepare_split_conv_transpose2d_weights_bias(
    in_channels,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    torch_weight_tensor,
    torch_bias_tensor,
):
    split_output_channels = out_channels // conv_out_channel_split_factor
    split_input_channels = in_channels // conv_in_channel_split_factor

    # Split weights - conv_transpose2d uses IOHW format
    # FIXED: Split output channels first (dimension 0), then input channels (dimension 1)
    if conv_out_channel_split_factor > 1:
        split_weight_tensors = list(torch.split(torch_weight_tensor, split_output_channels, 0))
    else:
        split_weight_tensors = [torch_weight_tensor]

    for i in range(len(split_weight_tensors)):
        split_weight_tensors[i] = torch.split(split_weight_tensors[i], split_input_channels, 1)

    # FIXED: Use correct variable name and consider using float32 for better PCC
    ttnn_split_weights = [
        [
            ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat16,  # Consider float32 for better numerical accuracy
            )
            for weight in output_channel_split_weights  # FIXED: Correct variable name
        ]
        for output_channel_split_weights in split_weight_tensors
    ]

    # Split bias - same as conv2d
    if conv_out_channel_split_factor > 1:
        split_bias_tensors = list(torch.split(torch_bias_tensor, split_output_channels, 3))
    else:
        split_bias_tensors = [torch_bias_tensor]

    ttnn_split_bias = [
        ttnn.from_torch(
            bias,
            dtype=ttnn.bfloat16,  # Match weights dtype
        )
        for bias in split_bias_tensors
    ]

    return ttnn_split_weights, ttnn_split_bias


def split_conv_transpose2d_and_run(
    hidden_states,
    conv_weight,
    conv_bias,
    device,
    in_channels,
    input_height,
    input_width,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    compute_config,
    conv_config,
    conv_output_dtype,
    kernel_size=3,
    padding=1,
    output_padding=0,
    return_weights_and_bias=False,
    stride=1,
):
    split_input_channels = in_channels // conv_in_channel_split_factor
    split_output_channels = out_channels // conv_out_channel_split_factor

    conv_kwargs = {
        "in_channels": split_input_channels,
        "out_channels": split_output_channels,
        "batch_size": 1,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": (padding, padding),
        "output_padding": (output_padding, output_padding),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
        "mirror_kernel": True,  # Required for conv_transpose2d
    }

    outputs = []
    # Pre-initialize device_weights to match conv_weight structure: [in_channel_slice][out_channel_slice]
    device_weights = [[] for _ in range(conv_in_channel_split_factor)]
    device_bias = []
    # First loop goes over output channel slices and saves outputs in a list
    for out_channel_slice_id in range(conv_out_channel_split_factor):
        out_channel_slice_output = None
        # Second loop goes over input channel slices and accumulates the outputs
        for in_channel_slice_id in range(conv_in_channel_split_factor):
            hidden_states_slice = hidden_states[
                :, :, :, in_channel_slice_id * split_input_channels : (in_channel_slice_id + 1) * split_input_channels
            ]
            results = ttnn.conv_transpose2d(
                input_tensor=hidden_states_slice,
                weight_tensor=conv_weight[in_channel_slice_id][out_channel_slice_id],
                bias_tensor=conv_bias[out_channel_slice_id],
                **conv_kwargs,
                compute_config=compute_config,
                return_weights_and_bias=return_weights_and_bias,
                dtype=conv_output_dtype,
            )
            # results = ttnn.to_memory_config(results, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states_slice.deallocate(True)

            if return_weights_and_bias:
                # First time we call this function, weights and biases are passed in on host;
                # Save them so that we can reuse them on the next calls
                in_channel_slice_output, [weights, bias] = results
                device_weights[in_channel_slice_id].append(weights)
                if in_channel_slice_id == 0:
                    device_bias.append(bias)
            else:
                in_channel_slice_output = results

            in_channel_slice_output = ttnn.move(in_channel_slice_output)

            if in_channel_slice_id == 0:
                if in_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    out_channel_slice_output = ttnn.to_memory_config(in_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
                    in_channel_slice_output.deallocate(True)
                else:
                    out_channel_slice_output = in_channel_slice_output
            else:
                out_channel_slice_output = ttnn.add(
                    out_channel_slice_output,
                    in_channel_slice_output,
                    # memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                in_channel_slice_output.deallocate(True)

        if out_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            out_channel_slice_output = ttnn.to_memory_config(out_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(out_channel_slice_output)

    hidden_states.deallocate(True)

    # Concatenate the outputs, if we split by output channels
    if len(outputs) > 1:
        output = ttnn.concat(outputs, dim=-1)
        for output_slice in outputs:
            output_slice.deallocate(True)
    else:
        output = outputs[0]

    if return_weights_and_bias:
        return output, device_weights, device_bias
    return output


class TtPointPillarsConvTranspose2DSplit(LightweightModule):
    def __init__(
        self,
        conv_transpose,
        conv_transpose_pth,
        device=None,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        is_dealloc_act=False,
        return_dims=False,
        memory_config=None,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        conv_in_channel_split_factor=2,
        conv_out_channel_split_factor=2,
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
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor

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
            activation=None,
        )

        conv_weights, conv_bias = prepare_split_conv_transpose2d_weights_bias(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_in_channel_split_factor=self.conv_in_channel_split_factor,  # Split 128 into 2x64
            conv_out_channel_split_factor=self.conv_out_channel_split_factor,  # Split 128 into 2x64
            torch_weight_tensor=conv_transpose_pth.weight,
            torch_bias_tensor=conv_transpose_pth.bias,
        )
        if conv_transpose_pth.bias is not None:
            self.bias = conv_bias
        else:
            self.bias = None

        self.weight = conv_weights
        self.memory_config = memory_config
        self._weights_prepared = False  # Track if weights have been prepared on device

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_height = shape[1]
            input_width = shape[2]
        else:
            batch_size = x.shape[0]
            input_height = x.shape[1]
            input_width = x.shape[2]

        if not self._weights_prepared:
            # First call: prepare weights on device and store them
            output, self.weight, self.bias = split_conv_transpose2d_and_run(
                hidden_states=x,
                conv_weight=self.weight,
                conv_bias=self.bias,
                device=self.device,
                in_channels=self.conv_transpose.in_channels,
                input_height=input_height,
                input_width=input_width,
                out_channels=self.conv_transpose.out_channels,
                conv_in_channel_split_factor=self.conv_in_channel_split_factor,
                conv_out_channel_split_factor=self.conv_out_channel_split_factor,
                compute_config=self.compute_config,
                conv_config=self.conv_config,
                conv_output_dtype=ttnn.bfloat16,
                kernel_size=self.conv_transpose.kernel_size,
                padding=0,
                output_padding=0,
                stride=self.conv_transpose.stride,
                return_weights_and_bias=True,
            )
            self._weights_prepared = True
        else:
            # Subsequent calls: use prepared weights, no write operation
            output = split_conv_transpose2d_and_run(
                hidden_states=x,
                conv_weight=self.weight,
                conv_bias=self.bias,
                device=self.device,
                in_channels=self.conv_transpose.in_channels,
                input_height=input_height,
                input_width=input_width,
                out_channels=self.conv_transpose.out_channels,
                conv_in_channel_split_factor=self.conv_in_channel_split_factor,
                conv_out_channel_split_factor=self.conv_out_channel_split_factor,
                compute_config=self.compute_config,
                conv_config=self.conv_config,
                conv_output_dtype=ttnn.bfloat16,
                kernel_size=self.conv_transpose.kernel_size,
                padding=0,
                output_padding=0,
                stride=self.conv_transpose.stride,
                return_weights_and_bias=False,
            )

        return output
