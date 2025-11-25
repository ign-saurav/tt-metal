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
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast

from models.experimental.miniCPMo.tt.tt_siglip_vision_embedding import TTSiglipVisionEmbeddings

from ttnn.model_preprocessing import preprocess_model_parameters


def load_or_create(path, shape, dtype):
    if os.path.exists(path):
        print(f"Loading: {path}")
        return torch.load(path)
    else:
        print(f"File not found, creating random tensor for: {path}")

        return (
            torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)
        )


def create_siglip_vision_embedding_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if hasattr(torch_model, "patch_embedding"):
            parameters["patch_embedding"] = {}

            # Get Conv2d weights: [out_channels, in_channels, kernel_h, kernel_w]
            weight = torch_model.patch_embedding.weight  # [1152, 3, 14, 14]
            bias = torch_model.patch_embedding.bias  # [1152]

            out_channels, in_channels, kh, kw = weight.shape  # 1152, 3, 14, 14

            # CRITICAL FIX: torch.nn.Unfold orders elements as [C, kh, kw] when flattened
            # The correct transformation is permute(1, 2, 3, 0) to match Unfold's order
            # This gives: [in_channels, kh, kw, out_channels] -> reshape to [in_channels * kh * kw, out_channels]
            # This matches the unfolded input format where channels vary fastest, then height, then width
            weight_2d = weight.permute(1, 2, 3, 0)  # [3, 14, 14, 1152]
            weight_2d = weight_2d.reshape(in_channels * kh * kw, out_channels)  # [588, 1152]

            # Pad to nearest 32 for TILE_LAYOUT: 588 -> 608
            from models.common.utility_functions import nearest_32

            padded_in_features = nearest_32(weight_2d.shape[0])
            if padded_in_features > weight_2d.shape[0]:
                pad_rows = padded_in_features - weight_2d.shape[0]
                weight_2d = torch.nn.functional.pad(weight_2d, (0, 0, 0, pad_rows))
                # Now weight_2d is [608, 1152]

            parameters["patch_embedding"]["weight"] = ttnn.from_torch(
                weight_2d, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            parameters["patch_embedding"]["bias"] = ttnn.from_torch(
                bias, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )

        if hasattr(torch_model, "position_embedding"):
            parameters["position_embedding"] = {}
            parameters["position_embedding"]["weight"] = ttnn.from_torch(
                torch_model.position_embedding.weight, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_siglip_vision_embedding(device, input_dtype, weight_dtype):
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

    embeddings_model = model.vpm.embeddings
    torch_output = embeddings_model.forward(all_pixel_values, patch_attn_mask, tgt_sizes)
    print(embeddings_model)

    # Get the state dict of vpm
    embeddings_model_state_dict = embeddings_model.state_dict()
    print(f"Embeddings model state dict has {len(embeddings_model_state_dict)} parameters")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: embeddings_model,
        custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, weight_dtype),
        device=device,
    )
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
    tt_siglip_vision_embedding = TTSiglipVisionEmbeddings(
        device=device,
        config=config,
        parameters=parameters,
    )

    tt_siglip_vision_embedding_out = tt_siglip_vision_embedding(all_pixel_values, position_embeddings)

    tt_torch_output = tt2torch_tensor(tt_siglip_vision_embedding_out)
    tt_torch_output = tt_torch_output.reshape(torch_output.shape)

    does_pass, pcc_message = check_with_pcc(torch_output, tt_torch_output, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed: {pcc_message}"
