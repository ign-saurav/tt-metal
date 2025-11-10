import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    attn_implementation="sdpa",  # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True,
)


model = model.eval()
# model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)

# In addition to vision-only mode, tts processor and vocos also needs to be initialized
model.init_tts()
print(model)
print("DONE")
