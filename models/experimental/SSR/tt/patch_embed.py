import ttnn
from models.common.lightweightmodule import LightweightModule
import torch.nn as nn


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
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
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
            weights_dtype=ttnn.bfloat16,
            activation="",
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,  # Free input memory after use
            reallocate_halo_output=True,  # Reduce memory fragmentation
            act_block_h_override=32,  # Use smaller activation blocks
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # Use height sharding
        )

    def forward(self, x, config=None):
        batch_size, img_c, img_h, img_w = x.shape  # NHWC format
        x = ttnn.permute(x, (0, 2, 3, 1), memory_config=self.memory_config)

        # Use DRAM slicing for large inputs
        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dSliceHeight, num_slices=4  # Adjust based on memory constraints
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
            dtype=ttnn.bfloat16,  # Specify output dtype
            slice_config=slice_config,
        )
        flattened_output = ttnn.reshape(output, (batch_size, out_height * out_width, self.embed_dim))
        # transposed_output = ttnn.transpose(flattened_output, 1, 2)

        # return transposed_output
        return flattened_output


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
