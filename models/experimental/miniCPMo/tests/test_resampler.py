import ttnn
import torch
import pytest

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from transformers import AutoModel

from models.experimental.miniCPMo.tt.tt_resampler import TTResampler

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight
from models.experimental.miniCPMo.tests.test_multi_head_attn import create_self_attn_preprocessor


def create_tensor(shape, dtype):
    return torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)


def create_resampler_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "attn"):
            self_attn_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.attn,
                custom_preprocessor=create_self_attn_preprocessor(device, weight_dtype),
                device=device,
            )
            parameters["attn"] = self_attn_params

        if (
            hasattr(torch_model, "kv_proj")
            and hasattr(torch_model, "ln_q")
            and hasattr(torch_model, "query")
            and hasattr(torch_model, "ln_kv")
            and hasattr(torch_model, "ln_post")
            and hasattr(torch_model, "proj")
        ):
            parameters["kv_proj"] = {}
            parameters["ln_q"] = {}
            parameters["ln_kv"] = {}
            parameters["ln_post"] = {}
            parameters["proj"] = ttnn.from_torch(
                torch_model.proj, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            parameters["query"] = ttnn.from_torch(
                torch_model.query, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            # Linear projection for kv_proj
            parameters["kv_proj"]["weight"] = preprocess_linear_weight(torch_model.kv_proj.weight, dtype=weight_dtype)

            # Layer norm parameters - use ttnn.from_torch directly
            for ln_name in ["ln_q", "ln_kv", "ln_post"]:
                ln_module = getattr(torch_model, ln_name)
                parameters[ln_name]["weight"] = ttnn.from_torch(
                    ln_module.weight.reshape(1, -1),  # Reshape to (1, D)
                    dtype=weight_dtype,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                )
                parameters[ln_name]["bias"] = ttnn.from_torch(
                    ln_module.bias.reshape(1, -1),  # Reshape to (1, D)
                    dtype=weight_dtype,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_mini_cpm_o(device, input_dtype, weight_dtype):
    model_name = "openbmb/MiniCPM-o-2_6"
    logger.info(f"Loading model from HuggingFace: {model_name}")

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )
    model = model.eval()

    vision_embedding = create_tensor((1, 999, 1152), torch.bfloat16)
    tgt_sizes = create_tensor((1, 2), torch.int32)

    resampler = model.resampler
    resampler_out = resampler(vision_embedding, tgt_sizes)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: resampler,
        custom_preprocessor=create_resampler_preprocessor(device, weight_dtype),
        device=device,
    )
    tt_model = TTResampler(
        num_queries=64,
        embed_dim=3584,
        num_heads=28,
        kv_dim=1152,
        parameters=parameters,
        device=device,
        input_dtype=input_dtype,
    )
    tt_vision_embedding = ttnn.from_torch(
        vision_embedding,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=input_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_resampler_out = tt_model(tt_vision_embedding, tgt_sizes)

    tt_torch_output = tt2torch_tensor(tt_resampler_out)

    tt_torch_output = tt_torch_output.reshape(resampler_out.shape)
    does_pass, pcc_message = check_with_pcc(resampler_out, tt_torch_output, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
