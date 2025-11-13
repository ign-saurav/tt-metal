# import ttnn
import torch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO

from transformers import AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer
from PIL import Image


def test_mini_cpm_o():
    config = AutoConfig.from_pretrained(
        "openbmb/MiniCPM-o-2_6",
        trust_remote_code=True,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )

    # Set the model name/path (required for processor loading)
    config._name_or_path = "openbmb/MiniCPM-o-2_6"
    print(config)
    print("Initializing MiniCPM-o model...")
    # Initialize the model directly with the config
    # with torch.device("meta"):
    with init_empty_weights():
        model = MiniCPMO(config)

    local_checkpoint_path = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6/snapshots/509805e84db1c84f154034d71a21c4f2331e6e11"
    load_checkpoint_and_dispatch(
        model,
        local_checkpoint_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    # Set model to eval mode
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
    image = Image.open("cat_img.jpg").convert("RGB")
    question = "What is in the image?"
    msgs = [{"role": "user", "content": [image, question]}]
    print("runnning inference...")
    res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    print(res)
