# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.experimental.SSR.reference.SSR.model.tile_selection import mask_token_inference
from models.experimental.SSR.tt.tile_selection import TTMaskTokenInference

from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_mask_token_inference_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if (
            hasattr(torch_model, "norm")
            and hasattr(torch_model, "q")
            and hasattr(torch_model, "k")
            and hasattr(torch_model, "v")
            and hasattr(torch_model, "proj")
        ):
            # Layer norm parameters
            parameters["norm"] = {}
            parameters["norm"]["weight"] = ttnn.from_torch(
                torch_model.norm.weight, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            parameters["norm"]["bias"] = ttnn.from_torch(
                torch_model.norm.bias, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )

            # QKV linear layers
            parameters["proj"] = {}

            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=weight_dtype)

            qkv_weight = torch.cat([torch_model.q.weight, torch_model.k.weight, torch_model.v.weight], dim=0)

            parameters["qkv"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=weight_dtype)

            if torch_model.q.bias is not None:
                qkv_bias = torch.cat([torch_model.q.bias, torch_model.k.bias, torch_model.v.bias], dim=0)
                parameters["qkv"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=weight_dtype)

            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=weight_dtype)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "input_shape, dim, num_heads",
    (((3, 17, 3072), 3072, 1),),
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_mask_token_inference(device, input_shape, dim, num_heads, input_dtype, weight_dtype):
    # Create test input [B, N, C]
    input_tensor = torch.randn(input_shape)

    ref_layer = mask_token_inference(dim=dim, num_heads=num_heads, qkv_bias=False)
    ref_layer.eval()
    ref_output = ref_layer(input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_mask_token_inference_preprocessor(device, weight_dtype),
        device=device,
    )

    tt_layer = TTMaskTokenInference(device=device, parameters=parameters, dim=dim, num_heads=num_heads)

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("MaskTokenInference Passed!")
    else:
        logger.warning("MaskTokenInference Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
