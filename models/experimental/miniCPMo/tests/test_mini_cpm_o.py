# import ttnn
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO

from transformers import AutoConfig


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
    model = MiniCPMO(config)

    # Set model to eval mode
    model = model.eval()

    print(model)
    print("Model setup complete!")
