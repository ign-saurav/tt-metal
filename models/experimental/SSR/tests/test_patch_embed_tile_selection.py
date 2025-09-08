# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.tt.patch_embed_tile_refinement import TTPatchEmbed
from models.utility_functions import comp_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


def to_2tuple(x):
    """Convert to 2-tuple"""
    if isinstance(x, int):
        return (x, x)
    return x


class PatchEmbed(nn.Module):
    """Reference PyTorch implementation matching your provided code"""

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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


def create_patch_embed_preprocessor_simple(device):
    """Preprocessor for simple PatchEmbed (no conv projection)"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # If the model has normalization
        if hasattr(torch_model, "norm") and torch_model.norm is not None:
            params["norm"] = {
                "weight": ttnn.from_torch(torch_model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                "bias": ttnn.from_torch(torch_model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            }

        return params

    return custom_preprocessor


def create_patch_embed_preprocessor_conv(device):
    """Preprocessor for PatchEmbed with convolution projection"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # If the model has a projection layer (conv2d)
        if hasattr(torch_model, "proj"):
            conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

            params["proj"] = {
                "weight": ttnn.prepare_conv_weights(
                    weight_tensor=ttnn.from_torch(torch_model.proj.weight, dtype=ttnn.bfloat16),
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.TILE_LAYOUT,
                    weights_format="OIHW",
                    in_channels=torch_model.proj.in_channels,
                    out_channels=torch_model.proj.out_channels,
                    batch_size=1,
                    input_height=torch_model.img_size[0],
                    input_width=torch_model.img_size[1],
                    kernel_size=torch_model.patch_size,
                    stride=torch_model.patch_size,
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=True,
                    groups=1,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_config,
                ),
                "bias": ttnn.prepare_conv_bias(
                    bias_tensor=ttnn.from_torch(torch_model.proj.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.TILE_LAYOUT,
                    in_channels=torch_model.proj.in_channels,
                    out_channels=torch_model.proj.out_channels,
                    batch_size=1,
                    input_height=torch_model.img_size[0],
                    input_width=torch_model.img_size[1],
                    kernel_size=torch_model.patch_size,
                    stride=torch_model.patch_size,
                    padding=(0, 0),
                    dilation=(1, 1),
                    groups=1,
                    device=device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_config,
                ),
            }

        # If the model has normalization
        if hasattr(torch_model, "norm") and torch_model.norm is not None:
            params["norm"] = {
                "weight": ttnn.from_torch(torch_model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                "bias": ttnn.from_torch(torch_model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, img_size, patch_size, in_chans, embed_dim, norm_layer",
    [
        (1, 224, 16, 3, 768, None),  # Standard ViT-Base config
        (2, 256, 4, 3, 96, None),  # Smaller embedding
        (1, 32, 4, 180, 180, None),  # Your SSR config
        # (1, 64, 8, 3, 192, nn.LayerNorm),     # With normalization
        (4, 128, 16, 3, 384, None),  # Batch size 4
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patch_embed_simple(device, batch_size, img_size, patch_size, in_chans, embed_dim, norm_layer):
    """Test the simplified PatchEmbed implementation (flatten + transpose only)"""
    torch.manual_seed(0)

    # Create reference model
    ref_model = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )
    ref_model.eval()

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_chans, img_size, img_size)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    # Create TTNN model
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_patch_embed_preprocessor_simple(device),
        device=device,
    )

    tt_model = TTPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        device=device,
        parameters=parameters,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert input to TTNN format
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # TTNN forward pass
    batch_size, channels, height, width = tt_input.shape
    tt_input = ttnn.reshape(tt_input, (batch_size, channels, height * width))
    tt_input = ttnn.transpose(tt_input, 1, 2)  # [batch, height*width, channels]
    tt_output = tt_model(tt_input)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"Batch: {batch_size}, Image: {img_size}x{img_size}, Patch: {patch_size}")
    logger.info(f"Channels: {in_chans}, Embed: {embed_dim}, Norm: {norm_layer is not None}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("Simple PatchEmbed Passed!")
    else:
        logger.warning("Simple PatchEmbed Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"
