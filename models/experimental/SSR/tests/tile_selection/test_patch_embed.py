# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import PatchEmbed
from models.experimental.SSR.tt.tile_selection import TTPatchEmbed
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_patch_embed_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, PatchEmbed):
            # Extract Conv2d weights - keep them in 4D format for conv2d
            conv_weight = torch_model.proj.weight  # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            conv_bias = torch_model.proj.bias  # Shape: [out_channels]

            parameters["proj"] = {}
            # Keep weights in 4D format and use ROW_MAJOR layout
            parameters["proj"]["weight"] = ttnn.from_torch(
                conv_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            # Reshape bias to [1, 1, 1, out_channels] format expected by conv2d
            conv_bias_reshaped = conv_bias.reshape(1, 1, 1, -1)
            parameters["proj"]["bias"] = ttnn.from_torch(
                conv_bias_reshaped, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("img_size, ch, patch_size, embed_dim, norm_layer", ((256, 3, 2, 96, None),))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
def test_patch_embed(device, img_size, ch, patch_size, embed_dim, norm_layer):
    input_shape = (3, ch, img_size, img_size)

    x = torch.randn(input_shape)

    dtype = ttnn.bfloat8_b

    ref_layer = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=ch,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )

    ref_output = ref_layer(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_patch_embed_preprocessor(device),
        device=device,
    )

    tt_layer = TTPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=ch,
        embed_dim=embed_dim,
        device=device,
        parameters=parameters,
        dtype=dtype,
    )

    # NCHW -> NHWC
    x = x.permute(0, 2, 3, 1)

    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(pcc_message)

    if does_pass:
        logger.info("PatchEmbed Passed!")
    else:
        logger.warning("PatchEmbed Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
