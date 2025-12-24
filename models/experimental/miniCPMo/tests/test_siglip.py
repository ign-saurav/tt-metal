import ttnn
import torch
import pytest

from loguru import logger
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

from transformers import AutoModel

from models.experimental.miniCPMo.tt.ttnn_siglip_vision import TtSiglipVisionTransformer

from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.miniCPMo.tests.test_siglip_vision_emb import create_siglip_vision_embedding_preprocessor


def create_tensor(shape, dtype):
    return torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)


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

    all_pixel_values = create_tensor((1, 3, 14, 13986), torch.bfloat16)
    patch_attn_mask = torch.ones((1, 999), dtype=torch.bool)
    tgt_sizes = create_tensor((1, 2), torch.int32)

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
    config = model.config
    tt_model = TtSiglipVisionTransformer(
        mesh_device=device,
        config=config,
        parameters=parameters,
        hidden_size=config.vision_config.hidden_size,
        num_attention_heads=config.vision_config.num_attention_heads,
        num_hidden_layers=config.vision_config.num_hidden_layers,
        patch_size=config.vision_config.patch_size,
        image_size=config.vision_config.image_size,
        num_channels=config.vision_config.num_channels,
    )
    tt_model.load_weights(vpm_state_dict)

    # After getting position_embedding_weight from parameters
    position_embedding_weight = parameters["position_embedding"]["weight"]

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
    does_pass, pcc_message = check_with_pcc(tt_model_output, torch_output.last_hidden_state, 0.90)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed, PCC: {pcc_message}"
