import torch
import pytest
import ttnn
import math
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.experimental.SSR.reference.SSR.model.tile_selection import mask_token_inference
from models.experimental.SSR.tt.mask_token_inference import TTMaskTokenInference

from models.utility_functions import tt2torch_tensor, comp_pcc


def create_mask_token_inference_preprocessor(device):
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
                torch_model.norm.weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
            parameters["norm"]["bias"] = ttnn.from_torch(
                torch_model.norm.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )

            # QKV linear layers
            parameters["q"] = {}
            parameters["k"] = {}
            parameters["v"] = {}
            parameters["proj"] = {}

            parameters["q"]["weight"] = preprocess_linear_weight(torch_model.q.weight, dtype=ttnn.bfloat16)
            parameters["k"]["weight"] = preprocess_linear_weight(torch_model.k.weight, dtype=ttnn.bfloat16)
            parameters["v"]["weight"] = preprocess_linear_weight(torch_model.v.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)

            if torch_model.q.bias is not None:
                parameters["q"]["bias"] = preprocess_linear_bias(torch_model.q.bias, dtype=ttnn.bfloat16)
                parameters["k"]["bias"] = preprocess_linear_bias(torch_model.k.bias, dtype=ttnn.bfloat16)
                parameters["v"]["bias"] = preprocess_linear_bias(torch_model.v.bias, dtype=ttnn.bfloat16)

            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "input_shape, image_size, patch_size, token_size",
    (
        ((1, 17, 3072), 256, 2, 4),  # Original test case
        # ((2, 33, 1536), 128, 2, 2),   # Medium case
        # ((4, 65, 768), 64, 2, 1),     # Smaller case
        # ((8, 129, 384), 32, 1, 1),    # Very small case
    ),
)
def test_mask_token_inference(input_shape, image_size, patch_size, token_size):
    # Create test input [B, N, C] where first token is cls token
    # input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    input_tensor = torch.randn(input_shape)
    num_layers = int(math.log2((image_size // patch_size) // token_size))
    dim = 96 * (2**num_layers)

    ref_layer = mask_token_inference(dim=dim, num_heads=1, qkv_bias=True)
    ref_layer.eval()
    ref_output = ref_layer(input_tensor)

    device = ttnn.open_device(device_id=0)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_mask_token_inference_preprocessor(device),
        device=device,
    )

    tt_layer = TTMaskTokenInference(device=device, parameters=parameters, dim=dim, num_heads=1, qkv_bias=True)

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(pcc_message)

    if does_pass:
        logger.info("MaskTokenInference Passed!")
    else:
        logger.warning("MaskTokenInference Failed!")

    ttnn.close_device(device)

    assert does_pass
