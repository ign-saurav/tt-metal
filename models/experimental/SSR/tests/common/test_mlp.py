# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import Mlp
from models.experimental.SSR.tt.common import TTMlp

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_mlp_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "fc1") and hasattr(torch_model, "fc2"):  # MLP model
            parameters["fc1"] = {}
            parameters["fc2"] = {}

            # Preprocess fc1 layer parameters
            parameters["fc1"]["weight"] = preprocess_linear_weight(torch_model.fc1.weight, dtype=ttnn.bfloat16)
            parameters["fc1"]["bias"] = preprocess_linear_bias(torch_model.fc1.bias, dtype=ttnn.bfloat16)

            # Preprocess fc2 layer parameters
            parameters["fc2"]["weight"] = preprocess_linear_weight(torch_model.fc2.weight, dtype=ttnn.bfloat16)
            parameters["fc2"]["bias"] = preprocess_linear_bias(torch_model.fc2.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "in_features, hidden_features, out_features, input_shape",
    (
        (3072, 3072, 3072, (3, 16, 3072)),  # TTTileSelection -> fea_mlp3
        (3072, 96, 96, (3, 16, 3072)),  # TTTileSelection -> mlp3
        (96, 384, 96, (3, 16384, 96)),  # TTSwinTransformerBlock[0], TTSwinTransformerBlock[1] -> mlp
        (192, 768, 192, (3, 4096, 192)),  # TTSwinTransformerBlock[2], TTSwinTransformerBlock[3] -> mlp
        (384, 1536, 384, (3, 1024, 384)),  # TTSwinTransformerBlock[4], TTSwinTransformerBlock[5] -> mlp
        (768, 3072, 768, (3, 256, 768)),  # TTSwinTransformerBlock[6], TTSwinTransformerBlock[7] -> mlp
        (1536, 6144, 1536, (3, 64, 1536)),  # TTSwinTransformerBlock[6], TTSwinTransformerBlock[7] -> mlp
    ),
)
def test_mlp(device, in_features, hidden_features, out_features, input_shape):
    x = torch.randn(input_shape)

    ref_layer = Mlp(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )

    ref_output = ref_layer(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer, custom_preprocessor=create_mlp_preprocessor(device), device=device
    )

    tt_layer = TTMlp(
        device,
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        parameters=parameters,
    )
    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_input = ttnn.to_memory_config(tt_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(pcc_message)

    if does_pass:
        logger.info("SSR MLP Passed!")
    else:
        logger.warning("SSR MLP Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
