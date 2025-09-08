import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import WindowAttention
from timm.models.layers import to_2tuple

from models.experimental.SSR.tt import TTWindowAttention
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
    comp_pcc,
)


def create_window_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "qkv"):  # WindowAttention model
            parameters["qkv"] = {}
            parameters["proj"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16)
            parameters["qkv"]["bias"] = preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16)

            # Preprocess relative position bias
            relative_position_bias = torch_model.relative_position_bias_table[
                torch_model.relative_position_index.view(-1)
            ].view(
                torch_model.window_size[0] * torch_model.window_size[1],
                torch_model.window_size[0] * torch_model.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            parameters["relative_position_bias"] = ttnn.from_torch(
                relative_position_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "input_shape, window_size, num_heads",
    (
        ((361, 49, 96), (7, 7), 3),
        ((100, 49, 192), (7, 7), 3),
        ((512, 64, 192), (8, 8), 3),
        ((49, 49, 96), (7, 7), 3),
        ((49, 49, 96), (7, 7), 8),
        ((49, 49, 192), (7, 7), 24),
        ((49, 49, 120), (7, 7), 5),
        ((24, 24, 120), (4, 6), 5),
        ((4, 49, 1536), (7, 7), 3),
    ),
)
def test_window_attn(input_shape, window_size, num_heads):
    try:
        x = torch.randn(input_shape)

        qkv_bias = True
        qk_scale = None
        attn_drop = 0.0
        proj_drop = 0.0
        dim = input_shape[-1]

        ref_layer = WindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        ref_output = ref_layer(x)

        device = ttnn.open_device(device_id=0)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_layer,
            custom_preprocessor=create_window_attention_preprocessor(device),
            device=device,
        )
        tt_layer = TTWindowAttention(
            parameters=parameters,
            device=device,
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = tt_layer(tt_input)
        tt_torch_output = tt2torch_tensor(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

        logger.info(pcc_message)

        if does_pass:
            logger.info("WindowAttn Passed!")
        else:
            logger.warning("WindowAttn Failed!")

        assert does_pass

    finally:
        ttnn.close_device(device)
