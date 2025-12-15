# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores

# Import tt_cnn builder API
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    UpsampleConfiguration,
    TtConv2d,
    TtMaxPool2d,
    TtUpsample,
    AutoShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
    WidthShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
)


# =============================================================================
# Builder API Wrappers - Use these for new code
# =============================================================================


def create_tt_conv2d(
    device,
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    kernel_size: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    weights_dtype=ttnn.bfloat16,
    activation_dtype=ttnn.bfloat16,
    output_dtype=ttnn.bfloat16,
    sharding_strategy=None,
    deallocate_activation=True,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
) -> TtConv2d:
    """
    Create a TtConv2d using the builder API.

    This is the preferred way to create convolutions for BEVDepth.
    """
    # Convert weights to TTNN format
    ttnn_weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    ttnn_bias = None
    if bias is not None:
        bias_reshaped = bias.reshape(1, 1, 1, -1) if len(bias.shape) == 1 else bias
        ttnn_bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.float32)

    # Default sharding strategy
    if sharding_strategy is None:
        sharding_strategy = AutoShardedStrategyConfiguration()

    config = Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight=ttnn_weight,
        bias=ttnn_bias,
        activation=activation,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        output_dtype=output_dtype,
        math_fidelity=math_fidelity,
        sharding_strategy=sharding_strategy,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    return TtConv2d(config, device)


def create_tt_maxpool2d(
    device,
    input_height: int,
    input_width: int,
    channels: int,
    batch_size: int,
    kernel_size: tuple = (2, 2),
    stride: tuple = (2, 2),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    ceil_mode: bool = False,
    dtype=ttnn.bfloat16,
) -> TtMaxPool2d:
    """
    Create a TtMaxPool2d using the builder API.
    """
    config = MaxPool2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        dtype=dtype,
    )

    return TtMaxPool2d(config, device)


def create_tt_upsample(
    device,
    input_height: int,
    input_width: int,
    channels: int,
    batch_size: int,
    scale_factor: int = 2,
    mode: str = "nearest",
) -> TtUpsample:
    """
    Create a TtUpsample using the builder API.
    """
    config = UpsampleConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        batch_size=batch_size,
        scale_factor=scale_factor,
        mode=mode,
    )

    return TtUpsample(config, device)


# =============================================================================
# Legacy Wrapper - Kept for backward compatibility
# =============================================================================


def ttnn_conv2d(
    input_tensor,
    weight_tensor,
    device,
    batch_size,
    input_height,
    input_width,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias_tensor=None,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    weights_dtype=ttnn.bfloat16,
    activations_dtype=ttnn.bfloat16,
    deallocate_activation=True,
    reallocate_halo_output=False,
    shard_layout=None,
    packer_l1_acc=False,
    fp32_dest_acc_en=True,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    slice_config=None,
    act_block_h_override=32,  # Default to 32, but allow per-conv override
):
    """
    Wrapper for ttnn.conv2d with common optimizations.
    Handles conversion of PyTorch tensors to TTNN format.
    """
    import torch

    # Convert PyTorch weights to TTNN if needed
    if isinstance(weight_tensor, torch.Tensor):
        weight_tensor = ttnn.from_torch(
            weight_tensor,
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    if bias_tensor is not None and isinstance(bias_tensor, torch.Tensor):
        # Reshape bias from (out_channels,) to (1, 1, 1, out_channels) for TTNN
        if len(bias_tensor.shape) == 1:
            bias_tensor = bias_tensor.view(1, 1, 1, -1)
        bias_tensor = ttnn.from_torch(
            bias_tensor,
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    # Handle act_block_h_override: None means auto (0), otherwise use the provided value
    act_block_h_val = 0 if act_block_h_override is None else act_block_h_override

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        activation=activation,
        shard_layout=None,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=(
            reallocate_halo_output if deallocate_activation else False
        ),  # Only useful with deallocate_activation
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        config_tensors_in_dram=True,  # Keep tensors in DRAM to reduce L1 usage
        act_block_h_override=act_block_h_val,  # 0 = auto (use maximum), >0 = override (must be multiple of 32)
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    output = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=activations_dtype,
        slice_config=slice_config,
    )

    return output[0]


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
        is_reshape=False,
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

    def _create_sharding_strategy(self, device):
        """Convert old-style sharding parameters to new ShardedStrategyConfiguration."""
        # Determine shard layout
        if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            sharding_strategy = HeightShardedStrategyConfiguration(
                reshard_if_not_optimal=self.reshard_if_not_optimal,
                act_block_h_override=self.act_block_h if self.act_block_h is not None else 0,
            )
        elif self.shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            sharding_strategy = WidthShardedStrategyConfiguration(
                reshard_if_not_optimal=self.reshard_if_not_optimal,
                act_block_w_div=self.act_block_w if self.act_block_w is not None else 1,
            )
        elif self.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            sharding_strategy = BlockShardedStrategyConfiguration(
                reshard_if_not_optimal=self.reshard_if_not_optimal,
                act_block_h_override=self.act_block_h if self.act_block_h is not None else 0,
                act_block_w_div=self.act_block_w if self.act_block_w is not None else 1,
            )
        else:
            # Auto sharding or None
            sharding_strategy = AutoShardedStrategyConfiguration()

        # Handle num_cores_nhw override
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            # Create a new sharding strategy with override_core_grid
            if isinstance(sharding_strategy, HeightShardedStrategyConfiguration):
                sharding_strategy = HeightShardedStrategyConfiguration(
                    reshard_if_not_optimal=sharding_strategy.reshard_if_not_optimal,
                    override_core_grid=shard_grid,
                    act_block_h_override=sharding_strategy.act_block_h_override,
                )
            elif isinstance(sharding_strategy, WidthShardedStrategyConfiguration):
                sharding_strategy = WidthShardedStrategyConfiguration(
                    reshard_if_not_optimal=sharding_strategy.reshard_if_not_optimal,
                    override_core_grid=shard_grid,
                    act_block_w_div=sharding_strategy.act_block_w_div,
                )
            elif isinstance(sharding_strategy, BlockShardedStrategyConfiguration):
                sharding_strategy = BlockShardedStrategyConfiguration(
                    reshard_if_not_optimal=sharding_strategy.reshard_if_not_optimal,
                    override_core_grid=shard_grid,
                    act_block_h_override=sharding_strategy.act_block_h_override,
                    act_block_w_div=sharding_strategy.act_block_w_div,
                )
            else:
                # For auto sharding, convert to height sharded with grid
                sharding_strategy = HeightShardedStrategyConfiguration(
                    reshard_if_not_optimal=self.reshard_if_not_optimal,
                    override_core_grid=shard_grid,
                    act_block_h_override=self.act_block_h if self.act_block_h is not None else 0,
                )

        return sharding_strategy

    def _convert_padding_to_2tuple(self):
        """Convert padding from 4-tuple (top, bottom, left, right) or int to 2-tuple (h, w)."""
        if isinstance(self.padding, tuple):
            if len(self.padding) == 4:
                # 4-tuple: (top, bottom, left, right) -> (h, w) = (top, left) assuming symmetric
                return (self.padding[0], self.padding[2])
            elif len(self.padding) == 2:
                return self.padding
        # int padding - already handled in __init__
        if isinstance(self.padding, int):
            return (self.padding, self.padding)
        return self.padding

    def __call__(self, device, input_tensor, input_shape):
        # Extract dimensions from input_shape (NHWC format)
        batch_size = input_shape[-4] if len(input_shape) >= 4 else input_shape[0]
        input_height = input_shape[-3] if len(input_shape) >= 3 else input_shape[1]
        input_width = input_shape[-2] if len(input_shape) >= 2 else input_shape[2]
        in_channels = input_shape[-1] if len(input_shape) >= 1 else input_shape[3]

        # Convert padding to 2-tuple format
        padding_2tuple = self._convert_padding_to_2tuple()

        # Create sharding strategy
        sharding_strategy = self._create_sharding_strategy(device)

        # Convert weights and bias to TTNN format if needed
        # TtConv2d expects weights to be ttnn.Tensor, but we might have torch.Tensor
        if isinstance(self.weights, torch.Tensor):
            ttnn_weight = ttnn.from_torch(self.weights, dtype=ttnn.float32)
        else:
            ttnn_weight = self.weights

        ttnn_bias = None
        if self.bias is not None:
            if isinstance(self.bias, torch.Tensor):
                bias_reshaped = self.bias.reshape(1, 1, 1, -1) if len(self.bias.shape) == 1 else self.bias
                ttnn_bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.float32)
            else:
                ttnn_bias = self.bias

        # Create Conv2dConfiguration
        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding_2tuple,
            dilation=self.dilation,
            groups=self.groups,
            weight=ttnn_weight,
            bias=ttnn_bias,
            activation=self.activation,
            activation_dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            output_dtype=self.dtype,
            math_fidelity=self.math_fidelity,
            sharding_strategy=sharding_strategy,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            reallocate_halo_output=self.reallocate_halo_output,
            # Note: slice_config is not directly supported in new API
            # Users should migrate to slice_strategy if needed
            slice_strategy=None,
        )

        # Create TtConv2d instance
        tt_conv2d = TtConv2d(config, device)

        # Call TtConv2d
        output_tensor, (h_out, w_out) = tt_conv2d(input_tensor, return_output_dim=True)

        # Handle is_reshape logic if needed
        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
            output_tensor = ttnn.reshape(output_tensor, (batch_size, h_out, w_out, output_tensor.shape[-1]))
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))

        return output_tensor, (batch_size, h_out, w_out, output_tensor.shape[-1])


class TTSplitConvTranspose2D:
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
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
        is_reshape=False,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        dtype=None,
        weights_dtype=None,
        math_fidelity=None,
        split_in=1,
        split_out=1,
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
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            ValueError("Invalid config")
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        elif isinstance(output_padding, tuple):
            self.output_padding = output_padding
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
        self.out_channels = self.weights.shape[1]  # ConvTranspose2d: (in_channels, out_channels, H, W)
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        self.shard_layout = shard_layout
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        self.split_in = split_in
        self.split_out = split_out
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

        # Prepare split weights and bias if needed
        self.split_weights = None
        self.split_bias = None
        self.split_weight_dtype = None  # Store the dtype used for split weights
        if split_in > 1 or split_out > 1:
            self._prepare_split_weights_bias()

    def _prepare_split_weights_bias(self):
        """Prepare split weights and bias for split convolution."""
        import torch

        # Convert weights to torch if needed
        # If weights are on device, ttnn.to_torch will move them to host
        if not isinstance(self.weights, torch.Tensor):
            torch_weights = ttnn.to_torch(self.weights)
        else:
            torch_weights = self.weights

        # ConvTranspose2d weight shape: (in_channels, out_channels, H, W)
        in_channels, out_channels, _, _ = torch_weights.shape
        split_out_channels = out_channels // self.split_out
        split_in_channels = in_channels // self.split_in

        # Split weights: first by output channels, then by input channels
        if self.split_out > 1:
            weight_chunks = list(torch.split(torch_weights, split_out_channels, 1))
        else:
            weight_chunks = [torch_weights]

        for i in range(len(weight_chunks)):
            weight_chunks[i] = list(torch.split(weight_chunks[i], split_in_channels, 0))

        # Convert to supported dtype for conv_transpose2d (bfloat16 or float32)
        # transform_weights_for_conv_transpose2d only supports BFLOAT16, FLOAT32, UINT32
        # Use bfloat16 if weights_dtype is bfloat16, otherwise use float32
        if self.weights_dtype == ttnn.bfloat16:
            self.split_weight_dtype = ttnn.bfloat16
        else:
            # For other dtypes (like bfloat8_b), convert to float32 for weight transformation
            self.split_weight_dtype = ttnn.float32

        # Convert to TTNN tensors - keep on host in ROW_MAJOR_LAYOUT
        # conv_transpose2d will handle device placement and transformation internally
        self.split_weights = [
            [
                ttnn.from_torch(weight, dtype=self.split_weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
                for weight in output_chunk
            ]
            for output_chunk in weight_chunks
        ]

        # Split bias if exists
        if self.bias is not None:
            if not isinstance(self.bias, torch.Tensor):
                torch_bias = ttnn.to_torch(self.bias)
            else:
                torch_bias = self.bias

            # Reshape bias if needed (should be 1D: [out_channels])
            if len(torch_bias.shape) > 1:
                torch_bias = torch_bias.flatten()

            if self.split_out > 1:
                bias_chunks = list(torch.split(torch_bias, split_out_channels, 0))
            else:
                bias_chunks = [torch_bias]

            self.split_bias = [
                ttnn.from_torch(bias, dtype=self.split_weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
                for bias in bias_chunks
            ]
        else:
            self.split_bias = None

    def _split_conv_transpose2d(
        self,
        device,
        input_tensor,
        input_shape,
    ):
        """
        Split conv_transpose2d operation to handle large tensors by splitting along input/output channels.

        Returns:
            output_tensor, (batch, H_out, W_out, C_out) - similar to TTSplitConvTranspose2D.__call__
        """
        # Create conv_config
        # Use split_weight_dtype for weights_dtype in config to match the actual weight dtype
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.split_weight_dtype if self.split_weight_dtype else self.weights_dtype,
            activation=self.activation,
            shard_layout=self.shard_layout if self.shard_layout else ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
        )

        # Create compute_config
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.math_fidelity,
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

        # Get input dimensions from input_shape (NHWC format)
        batch_size = input_shape[-4] if len(input_shape) >= 4 else input_shape[0]
        input_height = input_shape[-3] if len(input_shape) >= 3 else input_shape[1]
        input_width = input_shape[-2] if len(input_shape) >= 2 else input_shape[2]
        in_channels = input_shape[-1] if len(input_shape) >= 1 else input_shape[3]

        # Calculate split channel sizes
        split_in_channels = in_channels // self.split_in
        split_out_channels = self.out_channels // self.split_out
        total_out_channels = self.out_channels

        # Split input tensor along channel dimension (dim=3 for NHWC)
        if self.split_in > 1:
            input_tensor_split = ttnn.split(input_tensor, split_in_channels, 3)
            input_tensor.deallocate(True)
        else:
            input_tensor_split = [input_tensor]

        outputs = []
        Hout = None
        Wout = None

        # Process each output channel split
        for idx_out in range(self.split_out):
            # Process each input channel split
            for idx_in in range(self.split_in):
                # Get bias for this split (if bias exists)
                bias_tensor = None
                if self.split_bias is not None:
                    if isinstance(self.split_bias, list):
                        if len(self.split_bias) > idx_out:
                            bias_tensor = self.split_bias[idx_out]
                    else:
                        bias_tensor = self.split_bias

                [intermediate, [_Hout, _Wout], [d_w, d_b]] = ttnn.conv_transpose2d(
                    input_tensor=input_tensor_split[idx_in],
                    weight_tensor=self.split_weights[idx_out][idx_in],
                    bias_tensor=bias_tensor,
                    device=device,
                    in_channels=split_in_channels,
                    out_channels=split_out_channels,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    conv_config=conv_config,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=True,
                    dtype=self.dtype,
                    memory_config=self.memory_config,
                )

                # Store output dimensions from first call
                if Hout is None:
                    Hout = _Hout
                    Wout = _Wout

                # Accumulate results
                if idx_in == 0:
                    dram_intermediate = ttnn.to_memory_config(intermediate, ttnn.DRAM_MEMORY_CONFIG)
                    intermediate.deallocate(True)
                else:
                    dram_intermediate = ttnn.add(
                        dram_intermediate, intermediate, output_tensor=dram_intermediate, use_legacy=False
                    )
                    intermediate.deallocate(True)

            outputs.append(dram_intermediate)

        # Concatenate output splits along channel dimension (dim=-1 for NHWC)
        if len(outputs) > 1:
            output = ttnn.concat(outputs, dim=-1)
            for output_slice in outputs:
                output_slice.deallocate(True)
        else:
            output = outputs[0]

        # Return format similar to TTSplitConvTranspose2D.__call__: (output_tensor, (batch, H, W, C))
        return output, (batch_size, Hout, Wout, total_out_channels)

    def __call__(self, device, input_tensor, input_shape):
        # Use split method if splits are needed
        if self.split_in > 1 or self.split_out > 1:
            return self._split_conv_transpose2d(device, input_tensor, input_shape)

        # Otherwise use regular conv_transpose2d
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            shard_layout=self.shard_layout if self.shard_layout else ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.math_fidelity,
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

        [output_tensor, [_out_height, _out_width]] = ttnn.conv_transpose2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_shape[-1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            batch_size=input_shape[-4],
            input_height=input_shape[-3],
            input_width=input_shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=self.groups,
            return_weights_and_bias=False,
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
