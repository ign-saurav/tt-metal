# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTPatchEmbed(LightweightModule):
    """TTNN Image to Patch Embedding

    Args:
        img_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        device: TTNN device
        dtype: TTNN data type. Default: ttnn.bfloat16
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        device=None,
        dtype=ttnn.bfloat16,
        parameters=None,
        memory_config=None,
    ):
        # Convert to tuples (assuming square images/patches for simplicity)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        # Store projection parameters (weight and bias)
        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"]

        # Initialize compute config for the device
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Initialize conv config with no activation and default output layout
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.dtype,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,  # Free input memory after use
            reallocate_halo_output=True,  # Reduce memory fragmentation
            act_block_h_override=64,  # Use smaller activation blocks
        )

    def forward(self, x):
        batch_size, img_h, img_w, _ = x.shape  # NHWC format

        # Use DRAM slicing for large inputs
        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dSliceHeight, num_slices=6  # Adjust based on memory constraints
        )
        # Validate input dimensions
        assert (
            img_h == self.img_size[0] and img_w == self.img_size[1]
        ), f"Input image size ({img_h}*{img_w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # Corrected unpacking: only expect output tensor and output dimensions
        output, (out_height, out_width) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.proj_weight,
            bias_tensor=self.proj_bias,
            in_channels=self.in_chans,
            out_channels=self.embed_dim,
            device=self.device,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=(0, 0),  # Simplest case: no padding
            batch_size=batch_size,
            input_height=img_h,
            input_width=img_w,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,  # Only return the output tensor for simplest call
            return_weights_and_bias=False,  # Weights and bias are already prepared
            dtype=self.dtype,  # Specify output dtype
            slice_config=slice_config,
        )
        flattened_output = ttnn.reshape(output, (batch_size, out_height * out_width, self.embed_dim))

        return flattened_output
