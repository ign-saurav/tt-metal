import ttnn
import pytest
import json
import torch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from models.experimental.miniCPMo.reference.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from models.experimental.miniCPMo.tests.test_siglip_vision_emb import create_siglip_vision_embedding_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.miniCPMo.tt.tt_modeling_minicpmo import TTMiniCPMO


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
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

    # input Image and question
    image = Image.open("cat_img.jpg").convert("RGB")
    question = "What is in the image?"
    msgs = [{"role": "user", "content": [image, question]}]
    # print("runnning inference...")
    # res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    # print(res)

    vpm = model.vpm
    vpm_state_dict = vpm.state_dict()
    print(f"VPM state dict has {len(vpm_state_dict)} parameters")
    embeddings_model = model.vpm.embeddings
    emb_parameters = preprocess_model_parameters(
        initialize_model=lambda: embeddings_model,
        custom_preprocessor=create_siglip_vision_embedding_preprocessor(device, weight_dtype),
        device=device,
    )
    patch_size = embeddings_model.patch_size
    num_patches_per_side = embeddings_model.num_patches_per_side

    tt_model = TTMiniCPMO(config, device, emb_parameters, vpm_state_dict, patch_size, num_patches_per_side)

    tt_res = tt_model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    print(tt_res)
