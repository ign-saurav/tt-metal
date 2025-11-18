import ttnn
import json
import warnings
import torch
import pytest


from loguru import logger

# Suppress accelerate warnings about unused weights
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint", category=UserWarning)
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from models.experimental.miniCPMo.tt.tt_resampler import TTMultiheadAttention

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight

from models.experimental.miniCPMo.tt.tt_resampler import TTMultiheadAttention
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_self_attn_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if (
            hasattr(torch_model, "in_proj_weight")
            and hasattr(torch_model, "in_proj_bias")
            and hasattr(torch_model, "out_proj")
            and hasattr(torch_model, "bias_k")
            and hasattr(torch_model, "bias_v")
        ):
            parameters["in_proj_weight"] = {}
            parameters["in_proj_bias"] = {}
            parameters["out_proj_weight"] = {}
            parameters["out_proj_bias"] = {}
            parameters["bias_k"] = {}
            parameters["bias_v"] = {}

            # Preprocess in_proj_weight layer parameters
            parameters["in_proj_weight"] = preprocess_linear_weight(torch_model.in_proj_weight, dtype=weight_dtype)
            parameters["in_proj_bias"] = preprocess_linear_bias(torch_model.in_proj_bias, dtype=weight_dtype)

            # Preprocess out_proj_weight layer parameters
            parameters["out_proj_weight"] = preprocess_linear_weight(torch_model.out_proj.weight, dtype=weight_dtype)
            parameters["out_proj_bias"] = preprocess_linear_bias(torch_model.out_proj.bias, dtype=weight_dtype)

            # Preprocess value layer parameters
            if torch_model.bias_k and torch_model.bias_v:
                parameters["bias_k"] = preprocess_linear_bias(torch_model.bias_k, dtype=weight_dtype)
                parameters["bias_v"] = preprocess_linear_bias(torch_model.bias_v, dtype=weight_dtype)
            else:
                parameters["bias_k"] = None
                parameters["bias_v"] = None

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_self_attn(device, input_dtype, weight_dtype):
    # Load config directly from local JSON file
    config_path = "models/experimental/miniCPMo/reference/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = MiniCPMOConfig.from_dict(
        config_dict,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )

    print("Initializing MiniCPM-o model...")
    # Initialize the model directly with the config
    # with torch.device("meta"):
    with init_empty_weights():
        model = MiniCPMO(config)

    # local_checkpoint_path = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/509805e84db1c84f154034d71a21c4f2331e6e11"
    local_checkpoint_path = "models/experimental/miniCPMo/reference/safetensors"
    load_checkpoint_and_dispatch(
        model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    # Set model to eval mode
    model = model.eval()

    # Load tokenizer directly from local reference folder files
    tokenizer_path = "models/experimental/miniCPMo/reference"
    tokenizer = MiniCPMOTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")

    query = torch.load("attn_input_1.pt")
    key = torch.load("attn_input_2.pt")
    value = torch.load("attn_input_3.pt")
    key_padding_mask = torch.load("attn_input_mask.pt")

    print(query.shape, key.shape, value.shape, key_padding_mask.shape)

    # Access the attention module and its parameters
    attn_module = model.resampler.attn  # This is a MultiheadAttention instance

    # mha_output, mha_weights = attn_module(query, key, value, key_padding_mask=key_padding_mask)
    mha_output = attn_module(query, key, value, key_padding_mask=key_padding_mask)

    # Get other attributes
    embed_dim = attn_module.embed_dim
    num_heads = attn_module.num_heads

    add_zero_attn = attn_module.add_zero_attn
    dropout = attn_module.dropout
    training = attn_module.training

    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}")

    tt_query = ttnn.from_torch(
        query, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_key = ttnn.from_torch(
        key, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_value = ttnn.from_torch(
        value, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: attn_module,
        custom_preprocessor=create_self_attn_preprocessor(device, weight_dtype),
        device=device,
    )

    # Create TTMultiheadAttention instance and call multi_head_attention_forward
    tt_mha = TTMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    tt_attn_output = tt_mha.multi_head_attention_forward(
        device,
        input_dtype,
        tt_query,
        tt_key,
        tt_value,
        embed_dim,
        num_heads,
        parameters["in_proj_weight"],
        parameters["in_proj_bias"],
        bias_k=parameters["bias_k"],
        bias_v=parameters["bias_v"],
        add_zero_attn=add_zero_attn,
        dropout_p=dropout,
        out_proj_weight=parameters["out_proj_weight"],
        out_proj_bias=parameters["out_proj_bias"],
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    )

    tt_torch_output = tt2torch_tensor(tt_attn_output[0])
    tt_torch_output = tt_torch_output.reshape(mha_output[0].shape)
    does_pass, pcc_message = check_with_pcc(mha_output[0], tt_torch_output, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
