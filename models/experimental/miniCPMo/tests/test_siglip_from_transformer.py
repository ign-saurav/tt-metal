import ttnn
import torch
import pytest
from loguru import logger
from models.common.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.miniCPMo.tests.test_siglip_vision_emb import create_siglip_vision_embedding_preprocessor
from models.experimental.miniCPMo.tt.ttnn_siglip_vision import TtSiglipVisionTransformer
from transformers import AutoModel


def create_tensor(shape, dtype):
    return torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.tensor([[27, 37]], dtype=torch.int32)


@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_siglip_vision_transformer_alone(device, input_dtype, weight_dtype):
    model_name = "openbmb/MiniCPM-o-2_6"
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
    vpm_old = model.vpm
    # Create SigLIP vision configuration
    config = SiglipVisionConfig(
        image_size=224,
        patch_size=14,
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
    )

    # Create the reference PyTorch model
    vpm = SiglipVisionTransformer(config)
    vpm.use_head = False
    vpm.head = None  # Remove classification head

    # Load weights (assuming you have siglip_weights.pth)
    state_dict = torch.load("siglip_weights.pth", map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vision_model."):
            # Convert weights to BFloat16 to match input dtype
            new_state_dict[k.replace("vision_model.", "", 1)] = v.to(torch.bfloat16) if v.is_floating_point() else v
    import pdb

    pdb.set_trace()
    missing, unexpected = vpm.load_state_dict(new_state_dict, strict=False)

    # Create test inputs
    all_pixel_values = create_tensor((1, 3, 224, 224), torch.bfloat16)  # Standard image size
    patch_attn_mask = torch.ones((1, 196), dtype=torch.bool)  # 14x14 = 196 patches
    tgt_sizes = create_tensor((1, 2), torch.int32)

    # Get reference output
    vpm = vpm.to(torch.bfloat16)
    # torch_output = vpm.forward(all_pixel_values, patch_attn_mask, tgt_sizes)
    torch_output = vpm.forward(
        all_pixel_values,
        output_attentions=False,  # Explicitly set to False
        output_hidden_states=False,  # Explicitly set to False
        interpolate_pos_encoding=False,
    )

    # Initialize TTNN model parameters using the embeddings component
    parameters = preprocess_model_parameters(
        initialize_model=lambda: vpm.embeddings,  # Use vpm.embeddings instead of undefined embeddings_model
        custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, weight_dtype),
        device=device,
    )

    # Create TTNN model
    tt_model = TtSiglipVisionTransformer(
        mesh_device=device,
        config=config,
        parameters=parameters,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        patch_size=config.patch_size,
        image_size=config.image_size,
        num_channels=config.num_channels,
    )

    # Load weights
    tt_model.load_weights(vpm.state_dict())

    # Handle position embeddings
    position_embedding_weight = parameters["position_embedding"]["weight"]

    # Compute position IDs (simplified for standard image)
    batch_size = all_pixel_values.size(0)
    num_patches = (config.image_size // config.patch_size) ** 2
    position_ids = torch.arange(0, num_patches).unsqueeze(0).expand(batch_size, -1)

    # Convert to TTNN tensors
    position_ids_ttnn = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    position_embeddings = ttnn.embedding(
        position_ids_ttnn,
        position_embedding_weight,
        layout=ttnn.TILE_LAYOUT,
    )

    # Get TTNN output
    tt_model_output = tt_model.forward(all_pixel_values, position_embeddings)
    tt_model_output = tt2torch_tensor(tt_model_output)

    # Compare outputs
    tt_model_output = tt_model_output.reshape(torch_output.last_hidden_state.shape)
    does_pass, pcc_message = check_with_pcc(tt_model_output, torch_output.last_hidden_state, 0.90)
    logger.info(f"PCC: {pcc_message}")
    assert does_pass, f"PCC check failed, PCC: {pcc_message}"
