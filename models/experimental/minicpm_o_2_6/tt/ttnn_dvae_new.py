# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Dict, Any
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
    )

from models.common.lightweightmodule import LightweightModule

# from models.experimental.minicpm_o_2_6.reference_pytorch.minicpm_official.modeling_minicpmo import GFSQ
GFSQ = None


def create_pytorch_conv1d_from_weights(
    weight_tensor, bias_tensor, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1
):
    """Create a PyTorch Conv1d layer from weight and bias tensors"""
    import torch.nn as nn

    conv1d = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias_tensor is not None,
    )

    # Convert weight from ttnn tensor to torch if needed
    if isinstance(weight_tensor, torch.Tensor):
        weight = weight_tensor.clone()
    else:
        weight = ttnn.to_torch(weight_tensor).clone()

    # Handle weight shape: conv1d expects [out_channels, in_channels, kernel_size]
    # If weight is 4D [out, in, 1, kernel], squeeze to 3D
    if len(weight.shape) == 4:
        weight = weight.squeeze(2)  # Remove dimension 2
    elif len(weight.shape) == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)  # Remove dimension 1 if it's 1

    # Ensure weight is in correct shape [out_channels, in_channels, kernel_size]
    if len(weight.shape) != 3:
        raise ValueError(f"Expected weight to be 3D after processing, got shape {weight.shape}")

    conv1d.weight = nn.Parameter(weight)

    if bias_tensor is not None:
        # Convert bias from ttnn tensor to torch if needed
        if isinstance(bias_tensor, torch.Tensor):
            bias = bias_tensor.clone()
        else:
            bias = ttnn.to_torch(bias_tensor).clone()

        # Handle bias shape: remove extra dimensions
        while len(bias.shape) > 1:
            bias = bias.squeeze(0)
        # If bias is still multi-dimensional, flatten it
        if len(bias.shape) > 1:
            bias = bias.flatten()
        conv1d.bias = nn.Parameter(bias)

    return conv1d


def prepare_split_conv1d_weights_bias(
    in_channels,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    torch_weight_tensor,
    torch_bias_tensor,
):
    """Prepare split weights and bias for conv1d operations"""
    split_output_channels = out_channels // conv_out_channel_split_factor
    split_input_channels = in_channels // conv_in_channel_split_factor

    # Split weights - conv1d uses OIHW format [out_channels, in_channels, kernel_height, kernel_width]
    # For conv1d, kernel_height=1, so we split output channels first (dimension 0), then input channels (dimension 1)
    if conv_out_channel_split_factor > 1:
        split_weight_tensors = list(torch.split(torch_weight_tensor, split_output_channels, 0))
    else:
        split_weight_tensors = [torch_weight_tensor]

    for i in range(len(split_weight_tensors)):
        split_weight_tensors[i] = torch.split(split_weight_tensors[i], split_input_channels, 1)

    ttnn_split_weights = [
        [
            ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat16,
            )
            for weight in output_channel_split_weights
        ]
        for output_channel_split_weights in split_weight_tensors
    ]

    if torch_bias_tensor is not None:
        if conv_out_channel_split_factor > 1:
            split_bias_tensors = list(torch.split(torch_bias_tensor, split_output_channels, 0))
        else:
            split_bias_tensors = [torch_bias_tensor]

        ttnn_split_bias = [
            ttnn.from_torch(
                bias,
                dtype=ttnn.bfloat16,
            )
            for bias in split_bias_tensors
        ]
    else:
        ttnn_split_bias = None

    return ttnn_split_weights, ttnn_split_bias


def split_conv1d_and_run(
    hidden_states,
    conv_weight,
    conv_bias,
    device,
    in_channels,
    input_length,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    compute_config,
    conv_config,
    conv_output_dtype,
    kernel_size=3,
    padding=1,
    return_weights_and_bias=False,
    stride=1,
):
    """Run split conv1d operations"""
    split_input_channels = in_channels // conv_in_channel_split_factor
    split_output_channels = out_channels // conv_out_channel_split_factor

    conv_kwargs = {
        "in_channels": split_input_channels,
        "out_channels": split_output_channels,
        "batch_size": 1,
        "input_length": input_length,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": 1,
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    outputs = []
    # Pre-initialize device_weights to match conv_weight structure: [in_channel_slice][out_channel_slice]
    device_weights = [[] for _ in range(conv_in_channel_split_factor)]
    device_bias = []

    for out_channel_slice_id in range(conv_out_channel_split_factor):
        out_channel_slice_output = None
        for in_channel_slice_id in range(conv_in_channel_split_factor):
            # hidden_states is 3D: [batch, time_steps, channels]
            # Use ttnn.slice to slice along the last dimension (channels) to preserve shape information
            channel_start = in_channel_slice_id * split_input_channels
            channel_end = (in_channel_slice_id + 1) * split_input_channels

            # Get the full shape first
            batch_size_full, input_length_full, channels_full = hidden_states.shape

            # Use ttnn.slice for proper tensor slicing
            hidden_states_slice = ttnn.slice(
                hidden_states,
                start=[0, 0, channel_start],
                end=[batch_size_full, input_length_full, channel_end],
            )

            # Ensure tensor is on device and in ROW_MAJOR layout before operations
            hidden_states_slice = ttnn.to_device(hidden_states_slice, device)
            hidden_states_slice = ttnn.to_layout(hidden_states_slice, ttnn.ROW_MAJOR_LAYOUT)

            # Get actual dimensions from the sliced tensor (should be 3D: [batch, time_steps, channels])
            if len(hidden_states_slice.shape) != 3:
                raise ValueError(
                    f"Expected 3D tensor after slicing, got shape with {len(hidden_states_slice.shape)} dimensions: {hidden_states_slice.shape}"
                )

            batch_size_slice, input_length_slice, channels_slice = hidden_states_slice.shape

            # Update input_length and batch_size in conv_kwargs to match the actual tensor shape
            conv_kwargs_slice = conv_kwargs.copy()
            conv_kwargs_slice["input_length"] = input_length_slice
            conv_kwargs_slice["batch_size"] = batch_size_slice

            # Pass 3D tensor and let ttnn.conv1d handle the reshaping internally
            # According to conv1d.cpp, it reshapes [batch, time_steps, channels] to [batch, time_steps, 1, channels]
            bias_tensor = conv_bias[out_channel_slice_id] if conv_bias is not None else None
            results = ttnn.conv1d(
                input_tensor=hidden_states_slice,
                weight_tensor=conv_weight[in_channel_slice_id][out_channel_slice_id],
                bias_tensor=bias_tensor,
                **conv_kwargs_slice,
                compute_config=compute_config,
                return_weights_and_bias=return_weights_and_bias,
                dtype=conv_output_dtype,
            )
            # Deallocate the 3D slice
            hidden_states_slice.deallocate(True)

            if return_weights_and_bias:
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
                    dtype=ttnn.bfloat16,
                )
                in_channel_slice_output.deallocate(True)

        if out_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            out_channel_slice_output = ttnn.to_memory_config(out_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(out_channel_slice_output)

    hidden_states.deallocate(True)

    if len(outputs) > 1:
        output = ttnn.concat(outputs, dim=-1)
        for output_slice in outputs:
            output_slice.deallocate(True)
    else:
        output = outputs[0]

    if return_weights_and_bias:
        return output, device_weights, device_bias
    return output


class TtConv1DSplit(LightweightModule):
    def __init__(
        self,
        conv1d,
        conv1d_pth,
        device=None,
        weights_dtype=ttnn.bfloat8_b,
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
        self.conv1d = conv1d
        self.device = device
        self.in_channels = conv1d.in_channels
        self.out_channels = conv1d.out_channels
        self.kernel_size = conv1d.kernel_size
        self.stride = conv1d.stride
        self.padding = conv1d.padding
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=math_approx_mode,
        )

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=is_dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
        )

        conv_weights, conv_bias = prepare_split_conv1d_weights_bias(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_in_channel_split_factor=self.conv_in_channel_split_factor,
            conv_out_channel_split_factor=self.conv_out_channel_split_factor,
            torch_weight_tensor=conv1d_pth.weight,
            torch_bias_tensor=conv1d_pth.bias,
        )
        if conv1d_pth.bias is not None:
            self.bias = conv_bias
        else:
            self.bias = None

        self.weight = conv_weights
        self.memory_config = memory_config
        self._weights_prepared = False

    def __call__(self, x, shape=None):
        # Ensure tensor is on device
        x = ttnn.to_device(x, self.device)

        if shape is not None:
            batch_size = shape[0]
            input_length = shape[1]
        else:
            # Handle both 3D and 4D input tensors
            if len(x.shape) == 4:
                batch_size, input_length, _, _ = x.shape
            else:
                batch_size = x.shape[0]
                input_length = x.shape[1]

        if not self._weights_prepared:
            output, self.weight, self.bias = split_conv1d_and_run(
                hidden_states=x,
                conv_weight=self.weight,
                conv_bias=self.bias,
                device=self.device,
                in_channels=self.conv1d.in_channels,
                input_length=input_length,
                out_channels=self.conv1d.out_channels,
                conv_in_channel_split_factor=self.conv_in_channel_split_factor,
                conv_out_channel_split_factor=self.conv_out_channel_split_factor,
                compute_config=self.compute_config,
                conv_config=self.conv_config,
                conv_output_dtype=ttnn.bfloat16,
                kernel_size=self.conv1d.kernel_size,
                padding=self.conv1d.padding,
                stride=self.conv1d.stride,
                return_weights_and_bias=True,
            )
            self._weights_prepared = True
        else:
            output = split_conv1d_and_run(
                hidden_states=x,
                conv_weight=self.weight,
                conv_bias=self.bias,
                device=self.device,
                in_channels=self.conv1d.in_channels,
                input_length=input_length,
                out_channels=self.conv1d.out_channels,
                conv_in_channel_split_factor=self.conv_in_channel_split_factor,
                conv_out_channel_split_factor=self.conv_out_channel_split_factor,
                compute_config=self.compute_config,
                conv_config=self.conv_config,
                conv_output_dtype=ttnn.bfloat16,
                kernel_size=self.conv1d.kernel_size,
                padding=self.conv1d.padding,
                stride=self.conv1d.stride,
                return_weights_and_bias=False,
            )

        return output


class TtnnDVAE:
    """
    TTNN implementation of DVAE for audio reconstruction.

    Architecture:
        - Encoder: Downsampling convolutions + ConvNeXt blocks
        - Quantizer: GFSQ (simplified for TTNN compatibility)
        - Decoder: Upsampling convolutions + ConvNeXt blocks
        - Output: Mel spectrogram reconstruction
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize TTNN DVAE with weight loading for MiniCPM-o-2_6.

        Args:
            mesh_device: TTNN mesh device
            weights: Pre-loaded weights from MiniCPM checkpoint
            config: Optional configuration overrides
        """
        self.mesh_device = mesh_device
        self.weights = weights
        self.config = config or self._default_config()
        self.device = mesh_device

        for key, value in self.config.items():
            setattr(self, key, value)

        # Initialize component lists
        self.encoder_conv_in = []
        self.encoder_blocks = []
        self.encoder_conv_out = None
        self.downsample_conv = []
        self.decoder_conv_in = []
        self.decoder_blocks = []
        self.decoder_proj = None  # NEW: decoder projection layer
        self.out_conv = None
        self.coef = None

        if self.config["enable_gfsq"]:
            # self.vq_layer = TtnnGFSQ(
            #     device=self.device,  # Pass device to GFSQ
            #     dim=1024,  # Encoder output dimension
            #     levels=[5, 5, 5, 5],  # 4-level quantization per group
            #     G=2,  # 2 groups
            #     R=2,  # 2 residual levels
            # )

            # self.vq_layer.load_weights({})

            self.vq_layer = GFSQ(
                dim=1024,
                levels=(5, 5, 5, 5),
                G=2,
                R=2,
            )

        else:
            self.vq_layer = None

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat8_b,
            deallocate_activation=True,
            reallocate_halo_output=True,
            act_block_h_override=32,
            # reshard_if_not_optimal=True,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            # core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
        )

        if weights is not None:
            self.load_weights(weights)

        logger.info(
            f"TtnnDVAE initialized (PRODUCTION CONFIG): {self.num_encoder_layers} encoder layers, "
            f"{self.num_decoder_layers} decoder layers, hidden_dim={self.hidden_dim}, bn_dim={self.bn_dim}"
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default DVAE configuration for MiniCPM-o-2_6"""
        return {
            "num_encoder_layers": 12,  # Production: 12 layers
            "num_decoder_layers": 12,  # Production: 12 layers
            "hidden_dim": 256,
            "num_mel_bins": 100,
            "bn_dim": 128,  # Production: 128
            "enable_gfsq": False,  # Enable/disable GFSQ quantization
        }

    def load_weights(self, weights_dict: dict):
        """
        Load weights from PyTorch state dict and prepare them for conv2d operations.

        Args:
            weights_dict: Dictionary containing weight tensors with keys:
                - 'coef': Quantizer coefficient
                - 'downsample_conv.0.weight', 'downsample_conv.0.bias': Downsampling convs
                - 'encoder.conv_in.*': Encoder input convolutions
                - 'encoder.decoder_block.{i}.*': Encoder ConvNeXt blocks
                - 'encoder.conv_out.*': Encoder output convolution
                - 'decoder.conv_in.*': Decoder input convolutions
                - 'decoder.decoder_block.{i}.*': Decoder ConvNeXt blocks
                - 'out_conv.*': Final output convolution
        """
        logger.info("Loading DVAE weights...")

        # Quantizer coefficient
        self.coef = torch_to_ttnn(
            weights_dict["coef"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # Downsampling convolutions - create TtConv1DSplit instances
        self.downsample_conv = []
        # First conv: num_mel_bins -> 512, kernel=3, stride=1, padding=1
        conv1d_pth_0 = create_pytorch_conv1d_from_weights(
            weights_dict["downsample_conv.0.weight"],
            weights_dict["downsample_conv.0.bias"],
            self.num_mel_bins,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_0 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": self.num_mel_bins,
                "out_channels": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.downsample_conv.append(
            TtConv1DSplit(
                conv1d_config_0,
                conv1d_pth_0,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

        # Second conv: 512 -> 512, kernel=4, stride=2, padding=1
        conv1d_pth_2 = create_pytorch_conv1d_from_weights(
            weights_dict["downsample_conv.2.weight"],
            weights_dict["downsample_conv.2.bias"],
            512,
            512,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        conv1d_config_2 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": 4,
                "stride": 2,
                "padding": 1,
            },
        )()
        self.downsample_conv.append(
            TtConv1DSplit(
                conv1d_config_2,
                conv1d_pth_2,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

        # Encoder input convolution - create TtConv1DSplit instances
        self.encoder_conv_in = []
        # First conv: 512 -> bn_dim, kernel=3, stride=1, padding=1
        conv1d_pth_in_0 = create_pytorch_conv1d_from_weights(
            weights_dict["encoder.conv_in.0.weight"],
            weights_dict["encoder.conv_in.0.bias"],
            512,
            self.bn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_in_0 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": 512,
                "out_channels": self.bn_dim,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.encoder_conv_in.append(
            TtConv1DSplit(
                conv1d_config_in_0,
                conv1d_pth_in_0,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=get_activations_memory_config(),
            )
        )

        # Second conv: bn_dim -> hidden_dim, kernel=3, stride=1, padding=1
        conv1d_pth_in_2 = create_pytorch_conv1d_from_weights(
            weights_dict["encoder.conv_in.2.weight"],
            weights_dict["encoder.conv_in.2.bias"],
            self.bn_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_in_2 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": self.bn_dim,
                "out_channels": self.hidden_dim,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.encoder_conv_in.append(
            TtConv1DSplit(
                conv1d_config_in_2,
                conv1d_pth_in_2,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=get_activations_memory_config(),
            )
        )

        # Encoder ConvNeXt blocks - ALL weights need to be on device for consistency
        self.encoder_blocks = []
        for i in range(self.num_encoder_layers):
            block_weights = {
                "dwconv": {
                    "weight": torch_to_ttnn(
                        weights_dict[f"encoder.decoder_block.{i}.dwconv.weight"],
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    "bias": None,  # Disable bias for testing
                },
                "norm": {
                    "weight": torch_to_ttnn(
                        weights_dict[f"encoder.decoder_block.{i}.norm.weight"],
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.TILE_LAYOUT,
                    ),
                    "bias": torch_to_ttnn(
                        weights_dict[f"encoder.decoder_block.{i}.norm.bias"].view(1, 1, 1, -1),
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.TILE_LAYOUT,
                    ),
                },
                "pwconv1": {
                    "weight": torch_to_ttnn(
                        weights_dict[
                            f"encoder.decoder_block.{i}.pwconv1.weight"
                        ].t(),  # Transpose to [in, out] for ttnn.linear
                        self.device,
                        memory_config=get_weights_memory_config(),
                    ),
                    "bias": None,  # Disable bias for testing
                },
                "pwconv2": {
                    "weight": torch_to_ttnn(
                        weights_dict[
                            f"encoder.decoder_block.{i}.pwconv2.weight"
                        ].t(),  # Transpose to [in, out] for ttnn.linear
                        self.device,
                        memory_config=get_weights_memory_config(),
                    ),
                    "bias": None,  # Disable bias for testing
                },
            }
            self.encoder_blocks.append(block_weights)

        # Encoder output convolution - create TtConv1DSplit instance
        conv1d_pth_out = create_pytorch_conv1d_from_weights(
            weights_dict["encoder.conv_out.weight"],
            None,
            self.hidden_dim,
            1024,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv1d_config_out = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": self.hidden_dim,
                "out_channels": 1024,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
            },
        )()
        self.encoder_conv_out = TtConv1DSplit(
            conv1d_config_out,
            conv1d_pth_out,
            device=self.device,
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            is_dealloc_act=True,
            memory_config=get_activations_memory_config(),
        )

        # Decoder input convolution - create TtConv1DSplit instances
        self.decoder_conv_in = []
        # First conv: 1024 -> bn_dim, kernel=3, stride=1, padding=1
        conv1d_pth_dec_in_0 = create_pytorch_conv1d_from_weights(
            weights_dict["decoder.conv_in.0.weight"],
            weights_dict["decoder.conv_in.0.bias"],
            1024,
            self.bn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_dec_in_0 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": 1024,
                "out_channels": self.bn_dim,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.decoder_conv_in.append(
            TtConv1DSplit(
                conv1d_config_dec_in_0,
                conv1d_pth_dec_in_0,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=get_activations_memory_config(),
            )
        )

        # Second conv: bn_dim -> hidden_dim, kernel=3, stride=1, padding=1
        conv1d_pth_dec_in_2 = create_pytorch_conv1d_from_weights(
            weights_dict["decoder.conv_in.2.weight"],
            weights_dict["decoder.conv_in.2.bias"],
            self.bn_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_dec_in_2 = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": self.bn_dim,
                "out_channels": self.hidden_dim,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.decoder_conv_in.append(
            TtConv1DSplit(
                conv1d_config_dec_in_2,
                conv1d_pth_dec_in_2,
                device=self.device,
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                is_dealloc_act=True,
                memory_config=get_activations_memory_config(),
            )
        )

        # Decoder ConvNeXt blocks - ALL weights need to be on device for consistency
        self.decoder_blocks = []
        for i in range(self.num_decoder_layers):
            block_weights = {
                "dwconv": {
                    "weight": torch_to_ttnn(
                        weights_dict[f"decoder.decoder_block.{i}.dwconv.weight"],
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    "bias": None,  # Disable bias for testing
                },
                "norm": {
                    "weight": torch_to_ttnn(
                        weights_dict[f"decoder.decoder_block.{i}.norm.weight"],
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.TILE_LAYOUT,
                    ),
                    "bias": torch_to_ttnn(
                        weights_dict[f"decoder.decoder_block.{i}.norm.bias"].view(1, 1, 1, -1),
                        self.device,
                        memory_config=get_weights_memory_config(),
                        layout=ttnn.TILE_LAYOUT,
                    ),
                },
                "pwconv1": {
                    "weight": torch_to_ttnn(
                        weights_dict[
                            f"decoder.decoder_block.{i}.pwconv1.weight"
                        ].t(),  # Transpose to [in, out] for ttnn.linear
                        self.device,
                        memory_config=get_weights_memory_config(),
                    ),
                    "bias": None,  # Disable bias for testing
                },
                "pwconv2": {
                    "weight": torch_to_ttnn(
                        weights_dict[
                            f"decoder.decoder_block.{i}.pwconv2.weight"
                        ].t(),  # Transpose to [in, out] for ttnn.linear
                        self.device,
                        memory_config=get_weights_memory_config(),
                    ),
                    "bias": None,  # Disable bias for testing
                },
            }
            self.decoder_blocks.append(block_weights)

        # Decoder projection - NEW: hidden_dim -> 512 channels (1x1 conv) - create TtConv1DSplit instance
        conv1d_pth_proj = create_pytorch_conv1d_from_weights(
            weights_dict["decoder.conv_out.weight"],
            None,
            self.hidden_dim,
            512,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv1d_config_proj = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": self.hidden_dim,
                "out_channels": 512,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
            },
        )()
        self.decoder_proj = TtConv1DSplit(
            conv1d_config_proj,
            conv1d_pth_proj,
            device=self.device,
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            is_dealloc_act=True,
            memory_config=get_activations_memory_config(),
        )

        # Output convolution - create TtConv1DSplit instance
        conv1d_pth_out = create_pytorch_conv1d_from_weights(
            weights_dict["out_conv.weight"],
            None,
            512,
            self.num_mel_bins,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv1d_config_out_final = type(
            "Conv1dConfig",
            (),
            {
                "in_channels": 512,
                "out_channels": self.num_mel_bins,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        )()
        self.out_conv = TtConv1DSplit(
            conv1d_config_out_final,
            conv1d_pth_out,
            device=self.device,
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            is_dealloc_act=True,
            memory_config=get_activations_memory_config(),
        )

        logger.info("✅ DVAE weights loaded")

    def __call__(self, mel_spectrogram: ttnn.Tensor, debug_ops: dict = None) -> ttnn.Tensor:
        """
        Forward pass of DVAE.

        Args:
            mel_spectrogram: Input mel spectrogram in NHWC format [batch_size, 1, time_steps, num_mel_bins]
                           (Note: Caller must convert from NCHW to NHWC before calling)

        Returns:
            ttnn.Tensor: Reconstructed mel spectrogram in NHWC format [batch_size, 1, time_steps, num_mel_bins]
                        (Note: Caller must convert back to NCHW for comparison with PyTorch)
        """
        # Input is in NHWC format: [batch, H=1, W=time_steps, C=mel_bins]
        x = mel_spectrogram

        # Encoder
        encoded = self._encode(x, debug_ops)

        # Reshape from [batch, 1, seq, dim] to [batch, seq, dim] for quantization
        batch, _, seq, dim = encoded.shape
        encoded_flat = ttnn.reshape(encoded, [batch, seq, dim])

        # Apply GFSQ quantization (or bypass if disabled)
        if self.enable_gfsq:
            quantized, quant_indices = self.vq_layer.quantize(encoded_flat)
        else:
            # Bypass quantization - pass through unchanged
            quantized = encoded_flat

        # Reshape back to [batch, 1, seq, dim] for decoder
        quantized_4d = ttnn.reshape(quantized, [batch, 1, seq * 2, dim // 2])

        # Decoder
        reconstructed = self._decode(quantized_4d, debug_ops)

        return reconstructed

    def _encode(self, x: ttnn.Tensor, debug_ops: dict = None) -> ttnn.Tensor:
        """
        Encoder forward pass.
        Input x: [batch, H=1, W=time_steps, C=mel_bins] (NHWC format)
        """

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_approx_mode=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        for i, conv in enumerate(self.downsample_conv):
            print("i : ", i)
            # Use TtConv1DSplit instead of ttnn.conv1d
            # Input shape: [batch, 1, time_steps, channels] -> need to reshape to [batch, time_steps, channels] for conv1d
            batch, h, w, c = x.shape
            x_reshaped = ttnn.reshape(x, [batch, w, c])  # [batch, time_steps, channels]
            x = conv(x_reshaped, shape=[batch, w])
            # Convert to TILE_LAYOUT for GELU
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.gelu(x)
            # Convert back to ROW_MAJOR_LAYOUT and reshape back to 4D
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
            batch, w, c = x.shape
            x = ttnn.reshape(x, [batch, 1, w, c])

        # Encoder input convolutions
        # Input: [batch, 1, time_steps//2, 512] (NHWC)
        for i, conv in enumerate(self.encoder_conv_in):
            print("i : ", i)
            # Reshape to 3D for conv1d: [batch, 1, time_steps, channels] -> [batch, time_steps, channels]
            batch, h, w, c = x.shape
            x_reshaped = ttnn.reshape(x, [batch, w, c])
            x = conv(x_reshaped, shape=[batch, w])

            if i == 0:
                # Convert to TILE_LAYOUT for GELU
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = ttnn.gelu(x)
                # Convert back to ROW_MAJOR_LAYOUT
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
            batch, w, c = x.shape
            x = ttnn.reshape(x, [batch, 1, w, c])

        # Encoder ConvNeXt blocks (PRODUCTION: 12 blocks enabled)
        for block_weights in self.encoder_blocks:
            x = self._convnext_block(x, block_weights, debug_ops)

        # Encoder output (1x1 conv)
        # Input: [batch, 1, time_steps//2, hidden_dim], Output: [batch, 1, time_steps//2, 1024]
        batch, h, w, c = x.shape
        x_reshaped = ttnn.reshape(x, [batch, w, c])
        x = self.encoder_conv_out(x_reshaped, shape=[batch, w])
        # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
        batch, w, c = x.shape
        x = ttnn.reshape(x, [batch, 1, w, c])
        return x

    def _decode(self, x: ttnn.Tensor, debug_ops: dict = None) -> ttnn.Tensor:
        """
        Decoder forward pass.
        Input x: [batch, 1, time_steps//2, 1024] (NHWC format)
        Output: [batch, 1, time_steps//2, num_mel_bins] (NHWC format)
        """
        if debug_ops is None:
            debug_ops = {
                "depthwise_conv": True,
                "layer_norm": True,
                "pwconv1": True,
                "gelu": True,
                "pwconv2": True,
                "residual": True,
            }
        # Create conv config for decoder (same as encoder config)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            deallocate_activation=True,  # Free activation memory after use
            shard_layout=None,  # Disable sharding for single device
            act_block_h_override=32,  # Avoid L1_SMALL memory issues
            enable_act_double_buffer=True,  # Enable double buffering for memory efficiency
            enable_weights_double_buffer=True,  # Enable weight double buffering
        )

        # Create compute config
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_approx_mode=True,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Decoder input convolutions
        # Production: decoder processes 1024-channel features from encoder
        # Input: [batch, 1, time_steps//2, 1024] (NHWC from encoder output)
        for i, conv in enumerate(self.decoder_conv_in):
            # Reshape to 3D for conv1d: [batch, 1, time_steps, channels] -> [batch, time_steps, channels]
            batch, h, w, c = x.shape
            x_reshaped = ttnn.reshape(x, [batch, w, c])
            x = conv(x_reshaped, shape=[batch, w])

            if i == 0:
                # Convert to TILE_LAYOUT for GELU
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = ttnn.gelu(x)
                # Convert back to ROW_MAJOR_LAYOUT
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
            batch, w, c = x.shape
            x = ttnn.reshape(x, [batch, 1, w, c])

        # Decoder ConvNeXt blocks (PRODUCTION: 12 blocks enabled)
        for block_weights in self.decoder_blocks:
            x = self._convnext_block(x, block_weights, debug_ops)

        # Decoder projection: hidden_dim -> 512 channels (1x1 conv)
        # Input: [batch, 1, time_steps//2, hidden_dim], Output: [batch, 1, time_steps//2, 512]
        batch, h, w, c = x.shape
        x_reshaped = ttnn.reshape(x, [batch, w, c])
        x = self.decoder_proj(x_reshaped, shape=[batch, w])
        # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
        batch, w, c = x.shape
        x = ttnn.reshape(x, [batch, 1, w, c])

        # Output convolution
        # Production: 512 -> num_mel_bins
        # Input: [batch, 1, time_steps//2, 512], Output: [batch, 1, time_steps//2, num_mel_bins]
        batch, h, w, c = x.shape
        x_reshaped = ttnn.reshape(x, [batch, w, c])
        x = self.out_conv(x_reshaped, shape=[batch, w])
        # Reshape back to 4D: [batch, time_steps, channels] -> [batch, 1, time_steps, channels]
        batch, w, c = x.shape
        x = ttnn.reshape(x, [batch, 1, w, c])

        return x

    def _convnext_block(self, x: ttnn.Tensor, weights: dict, debug_ops: dict = None) -> ttnn.Tensor:
        """
        ConvNeXt block implementation for 2D tensors in NHWC format.

        Args:
            x: Input tensor [batch, 1, time_steps, channels] (NHWC)
            weights: Dictionary containing block weights
            debug_ops: Dictionary controlling which operations to enable for debugging

        Returns:
            ttnn.Tensor: Output tensor [batch, 1, time_steps, channels] (NHWC)
        """
        if debug_ops is None:
            debug_ops = {
                "depthwise_conv": True,
                "layer_norm": True,
                "pwconv1": True,
                "gelu": True,
                "pwconv2": True,
                "residual": True,
            }

        # Ensure input tensor is on device and in correct memory config
        x = ttnn.to_device(x, self.device)
        x = ttnn.to_memory_config(x, get_activations_memory_config())

        # Clone residual and ensure it's on device
        residual = ttnn.clone(x, memory_config=get_activations_memory_config())
        residual = ttnn.to_device(residual, self.device)

        # Step 1: Depthwise Conv
        if debug_ops["depthwise_conv"]:
            # For depthwise conv, we still need to use ttnn.conv1d directly since it's a depthwise operation
            # with groups parameter, which TtConv1DSplit may not handle correctly
            # Reshape to 3D: [B, 1, T, C] -> [B, T, C]
            batch, h, w, c = x.shape
            x_reshaped = ttnn.reshape(x, [batch, w, c])
            x = ttnn.conv1d(
                input_tensor=x_reshaped,
                weight_tensor=weights["dwconv"]["weight"],
                bias_tensor=weights["dwconv"]["bias"],
                in_channels=c,
                out_channels=c,
                device=self.device,
                batch_size=batch,
                input_length=w,
                kernel_size=7,
                stride=1,
                padding=(3, 3),
                groups=c,  # depthwise: groups = channels
                conv_config=self.conv_config,
                memory_config=get_activations_memory_config(),
            )
            # Reshape back to 4D: [B, T, C] -> [B, 1, T, C]
            batch, w, c = x.shape
            x = ttnn.reshape(x, [batch, 1, w, c])
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        # Step 2: LayerNorm
        if debug_ops["layer_norm"]:
            # TTNN layer norm supports 4D tensors directly - normalize over last dimension (C)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # Convert to TILE for LayerNorm
            x = ttnn.layer_norm(
                x,
                weight=weights["norm"]["weight"],
                bias=weights["norm"]["bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Use DRAM for LayerNorm
            )
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Convert back to ROW_MAJOR
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        # Step 3: Pointwise Conv 1 (expand channels)
        if debug_ops["pwconv1"]:
            # Pointwise conv 1: expand channels (Linear layer)
            # TTNN linear requires TILE_LAYOUT
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.linear(
                x,
                weights["pwconv1"]["weight"],
                bias=None,  # Disable bias for testing
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Convert back to ROW_MAJOR
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        # Step 4: GELU activation
        if debug_ops["gelu"]:
            # TTNN GELU requires TILE_LAYOUT for unary operations
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.gelu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Convert back to ROW_MAJOR
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        # Step 5: Pointwise Conv 2 (reduce channels)
        if debug_ops["pwconv2"]:
            # Pointwise conv 2: reduce channels back (Linear layer)
            # TTNN linear requires TILE_LAYOUT
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.linear(
                x,
                weights["pwconv2"]["weight"],
                bias=None,  # Disable bias for testing
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Convert back to ROW_MAJOR
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        # Step 6: Residual connection
        if debug_ops["residual"]:
            # Residual connection - add directly in NHWC format
            # Ensure both tensors are in the same memory config
            x = ttnn.to_memory_config(x, get_activations_memory_config())
            x = ttnn.add(x, residual, memory_config=get_activations_memory_config())
            # Ensure output tensor stays on device
            x = ttnn.to_device(x, self.device)

        return x
