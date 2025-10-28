# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from typing import Optional

# ---------------------------
# TTNN utility modules
# ---------------------------


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
        activation=None,
        groups=1,
        num_cores_nhw=None,
        is_reshape=True,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
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
        print(input_shape)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            shard_layout=self.shard_layout,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
            in_place=False,
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
            in_channels=self.weights.shape[1],
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
            # output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "nearest",
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
        # Convert a **sharded** tensor (distributed across cores) into a single **interleaved** tensor, choosing the backing memory
        # - DRAM: use when tensors are large or when later ops expect DRAM residency.
        # - L1  : fastest on-chip memory; use when the tensor fits and you’ll run
        #         compute-heavy kernels immediately after.
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Optionally pad channels to a multiple of 32 to match TT tile/channel alignment.
        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Remove channel padding if added.
        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            host = ttnn.from_device(output_tensor)
            host = ttnn.to_dtype(host, dtype)
            B, H, W, C = host.shape
            host = ttnn.reshape(host, [1, 1, B * H * W, C])
            output_tensor = ttnn.to_device(host, device)

        return output_tensor


class Conv2dNormActivation:
    """
    TTNN implementation of Conv2d + GroupNorm + ReLU block.

    Encapsulates the pattern used in RetinaNet regression head:
    - Conv2d with DRAM slicing
    - GroupNorm with tile alignment padding
    - ReLU activation
    """

    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        num_groups: int = 32,
        grid_size: Optional[ttnn.CoreGrid] = None,
        input_mask: Optional[ttnn.Tensor] = None,
        layer_optimisations=None,
        model_config=None,
    ):
        """
        Args:
            parameters: Dict with keys 'weight', 'norm_weight', 'norm_bias'
            device: TTNN device
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            num_groups: Number of groups for GroupNorm
            grid_size: CoreGrid for GroupNorm (defaults to 8x8)
            input_mask: Pre-created input mask for GroupNorm
        """
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups

        # Store parameters
        self.conv_parameter = {}
        self.conv_parameter["weight"] = parameters["weight"]
        self.conv_parameter["bias"] = ttnn.from_torch(
            torch.zeros(self.conv_parameter["weight"].shape[0]).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
        )
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]

        # Grid size for GroupNorm
        self.grid_size = grid_size if grid_size is not None else ttnn.CoreGrid(y=8, x=8)

        # Input mask for GroupNorm
        self.input_mask = input_mask

        # DRAM slicing config for conv2d
        self.slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
        )

        self.conv = TTConv2D(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            parameters=self.conv_parameter,
            kernel_fidelity=model_config,
            activation=None,
            is_reshape=True,
            # **layer_optimisations.cls_logits,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: Conv2d -> GroupNorm -> ReLU

        Args:
            x: Input tensor in NHWC format
            batch_size: Batch size
            input_height: Input height
            input_width: Input width

        Returns:
            Output tensor after Conv2d + GroupNorm + ReLU
        """

        # Conv2d operation
        x, _ = self.conv(self.device, x, x.shape)
        # x = self.conv(self.device, x, (input_height,input_width))

        # Get output shape after conv
        N, H_out, W_out, C = x.shape

        # Calculate padding needed for tile alignment
        # GroupNorm requires H_out * W_out divisible by (grid_size.y * 32)
        spatial_size = H_out * W_out
        required_size = ((spatial_size + self.grid_size.y * 32 - 1) // (self.grid_size.y * 32)) * (
            self.grid_size.y * 32
        )

        if spatial_size != required_size:
            # Pad spatial dimension to required size
            pad_amount = required_size - spatial_size

            # Reshape to (N, 1, H*W, C) for padding
            x_flat = ttnn.reshape(x, (N, 1, spatial_size, C))

            # Pad along spatial dimension
            x_padded = ttnn.pad(x_flat, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
        else:
            # Reshape to (N, 1, H*W, C) without padding
            x_padded = ttnn.reshape(x, (N, 1, spatial_size, C))

        # Apply GroupNorm
        x_normalized = ttnn.group_norm(
            x_padded,
            num_groups=self.num_groups,
            input_mask=self.input_mask,
            weight=self.norm_weight,
            bias=self.norm_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.grid_size,
            inplace=False,
        )

        # Unpad
        if spatial_size != required_size:
            # Slice back to original spatial size
            x_normalized = x_normalized[:, :, :spatial_size, :]

        # Reshape back using PRESERVED dimensions
        x = ttnn.reshape(x_normalized, (N, x.shape[-3], x.shape[-2], C))
        # Store original spatial dimensions
        H_out = x.shape[-3]
        W_out = x.shape[-2]
        # ReLU activation
        x = ttnn.relu(x)

        return x, x.shape
