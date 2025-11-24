import ttnn
import json
import torch
import pytest
import os

from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast

from models.experimental.minicpm_o_2_6.tt.ttnn_siglip_vision import TtSiglipVisionTransformer

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight
from models.experimental.miniCPMo.tests.test_multi_head_attn import create_self_attn_preprocessor


def load_or_create(path, shape, dtype):
    if os.path.exists(path):
        print(f"Loading: {path}")
        return torch.load(path)
    else:
        print(f"File not found, creating random tensor for: {path}")

        return (
            torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)
        )


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

    all_pixel_values = load_or_create("all_pixel_values.pt", (1, 3, 14, 13986), torch.bfloat16)
    patch_attn_mask = load_or_create("patch_attn_mask.pt", (1, 1, 999), torch.bfloat16)
    tgt_sizes = load_or_create("tgt_sizes.pt", (1, 2), torch.int32)

    vpm = model.vpm
    # torch_output = vpm.forward(all_pixel_values, patch_attn_mask, tgt_sizes)
    print(vpm)

    # Get the state dict of vpm
    vpm_state_dict = vpm.state_dict()
    import pdb

    pdb.set_trace()
    print(f"VPM state dict has {len(vpm_state_dict)} parameters")

    tt_model = TtSiglipVisionTransformer(
        mesh_device=device,
        hidden_size=config.vision_config.hidden_size,  # 1152
        num_attention_heads=config.vision_config.num_attention_heads,  # 16
        num_hidden_layers=config.vision_config.num_hidden_layers,  # 28
        patch_size=config.vision_config.patch_size,  # 14
        image_size=config.vision_config.image_size,  # 980
        num_channels=config.vision_config.num_channels,  # 3
    )
    tt_model.load_weights(vpm_state_dict)
    embeddings_model = model.vpm.embeddings
    embeddings = embeddings_model.forward(all_pixel_values, patch_attn_mask, tgt_sizes)

    tt_embeddings = ttnn.from_torch(
        embeddings, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_model_output = tt_model.forward(tt_embeddings)

    # You can also access specific keys, e.g.:
    # print(list(vpm_state_dict.keys())[:5])  # Print first 5 keys
    # resampler_out = resampler(vision_embedding, tgt_sizes)
    # parameters = preprocess_model_parameters(
    #     initialize_model=lambda: resampler,
    #     custom_preprocessor=create_resampler_preprocessor(device, weight_dtype),
    #     device=device,
    # )
    # tt_model = TTResampler(
    #     num_queries=64,
    #     embed_dim=3584,
    #     num_heads=28,
    #     kv_dim=1152,
    #     parameters=parameters,
    #     device=device,
    #     input_dtype=input_dtype,
    # )
    # tt_vision_embedding = ttnn.from_torch(
    #     vision_embedding,
    #     device=device,
    #     layout=ttnn.TILE_LAYOUT,
    #     dtype=input_dtype,
    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # )
    # tt_resampler_out = tt_model(tt_vision_embedding, tgt_sizes)

    # tt_torch_output = tt2torch_tensor(tt_resampler_out)

    # tt_torch_output = tt_torch_output.reshape(resampler_out.shape)
    # does_pass, pcc_message = check_with_pcc(resampler_out, tt_torch_output, 0.99)
    # logger.info(f"PCC: {pcc_message}")
    # assert does_pass, f"PCC check failed"
