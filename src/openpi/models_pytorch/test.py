import torch
import sys
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
import inspect

# Create a config for PaliGemmaForConditionalGeneration
config = CONFIG_MAPPING["paligemma"]()
config._vocab_size = 257152  # noqa: SLF001
config.image_token_index = 257152
config.text_config.hidden_size = 2048
config.text_config.intermediate_size = 8192
config.text_config.num_attention_heads = 16
config.text_config.num_hidden_layers = 18
config.text_config.num_key_value_heads = 16
config.text_config.head_dim = 128
config.text_config.hidden_activation = "gelu_pytorch_tanh"
config.text_config.torch_dtype = "float32"
config.text_config.vocab_size = 257152
config.text_config.use_adarms = False
config.text_config.adarms_cond_dim = 2048
config.vision_config.intermediate_size = 4304
config.vision_config.projection_dim = 2048
config.vision_config.projector_hidden_act = "gelu_fast"
config.vision_config.torch_dtype = "float32"

# Create the model
model = PaliGemmaForConditionalGeneration(config=config)

print("="*80)
print("CURRENT PaliGemmaForConditionalGeneration Model Structure (from installed transformers)")
print("="*80)
print(model)

