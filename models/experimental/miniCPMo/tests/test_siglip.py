import ttnn
import json
import torch
import pytest
import os

from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from models.experimental.miniCPMo.tt.ttnn_siglip_vision import TtSiglipVisionTransformer

from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.miniCPMo.tests.test_siglip_vision_emb import create_siglip_vision_embedding_preprocessor


def load_or_create(path, shape, dtype):
    if os.path.exists(path):
        print(f"Loading: {path}")
        return torch.load(path)
    else:
        print(f"File not found, creating random tensor for: {path}")

        return (
            torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)
        )


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

    all_pixel_values = load_or_create("all_pixel_values.pt", (1, 3, 14, 13986), torch.bfloat16)
    patch_attn_mask = load_or_create("patch_attn_mask.pt", (1, 1, 999), torch.bfloat16)
    tgt_sizes = load_or_create("tgt_sizes.pt", (1, 2), torch.int32)

    vpm = model.vpm
    torch_output = vpm.forward(all_pixel_values, patch_attn_mask, tgt_sizes)

    # Get the state dict of vpm
    vpm_state_dict = vpm.state_dict()

    embeddings_model = model.vpm.embeddings
    parameters = preprocess_model_parameters(
        initialize_model=lambda: embeddings_model,
        custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, weight_dtype),
        device=device,
    )
    tt_model = TtSiglipVisionTransformer(
        mesh_device=device,
        config=config,
        parameters=parameters,
        hidden_size=config.vision_config.hidden_size,  # 1152
        num_attention_heads=config.vision_config.num_attention_heads,  # 16
        num_hidden_layers=config.vision_config.num_hidden_layers,  # 28
        patch_size=config.vision_config.patch_size,  # 14
        image_size=config.vision_config.image_size,  # 980
        num_channels=config.vision_config.num_channels,  # 3
    )
    tt_model.load_weights(vpm_state_dict)

    # After getting position_embedding_weight from parameters
    position_embedding_weight = parameters["position_embedding"]["weight"]

    # Compute position IDs on CPU (bucketing logic)
    batch_size = all_pixel_values.size(0)
    max_im_h, max_im_w = all_pixel_values.size(2), all_pixel_values.size(3)
    patch_size = embeddings_model.patch_size
    max_nb_patches_h = max_im_h // patch_size
    max_nb_patches_w = max_im_w // patch_size

    num_patches_per_side = embeddings_model.num_patches_per_side
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)

    position_ids = torch.full(
        size=(batch_size, max_nb_patches_h * max_nb_patches_w),
        fill_value=0,
    )

    # Compute position IDs for each batch
    for batch_idx, p_attn_mask in enumerate(patch_attn_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    # Convert position_ids to TTNN tensor (must be uint32 for embedding operation)
    position_ids_ttnn = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,  # Important: embedding requires uint32 indices
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Use ttnn.embedding to index into the position embedding weight table
    position_embeddings = ttnn.embedding(
        position_ids_ttnn,
        position_embedding_weight,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_model_output = tt_model.forward(all_pixel_values, position_embeddings)

    tt_model_output = tt2torch_tensor(tt_model_output)

    tt_model_output = tt_model_output.reshape(torch_output.last_hidden_state.shape)
    does_pass, pcc_message = check_with_pcc(tt_model_output, torch_output.last_hidden_state, 0.98)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed"
