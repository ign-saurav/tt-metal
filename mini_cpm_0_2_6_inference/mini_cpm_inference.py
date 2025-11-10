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
print("Inference ....")

# test.py
image = Image.open("cat_img.jpg").convert("RGB")
question = "What is in the image?"
msgs = [{"role": "user", "content": [image, question]}]
res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
# res = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True,
#     stream=True
# )
# generated_text = ""
# for new_text in res:
#     generated_text += new_text
#     print(new_text, flush=True, end='')
