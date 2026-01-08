import pytest
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from models.experimental.granite_speech_33_8b.tt.granite_speech import GraniteSpeech


class TestConfig:
    """Test configuration for Conformer modules."""

    def __init__(self):
        self.input_dim = 160
        self.hidden_dim = 1024
        self.output_dim = 256
        self.feedforward_mult = 4
        self.num_heads = 8
        self.dim_head = 128
        self.max_pos_emb = 512
        self.context_size = 200
        self.conv_expansion_factor = 2
        self.conv_kernel_size = 15
        self.dropout = 0.1
        self.num_layers = 16
        self.num_attention_heads = 16
        self.hidden_size = 1024
        self.encoder_hidden_size = 1024
        self.attention_probs_dropout_prob = 0.1
        self.chunk_size_feed_forward = 0
        self.cross_attention_frequency = 1
        self.use_qformer_text_input = False
        self.num_hidden_layers = 2
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.projector_config_hidden_size = 1024
        self.downsample_rate = 5
        self.window_size = 15
        self.text_config_hidden_size = 4096
        self.audio_token_id = 49159
        self.vocab_size = 49160
        self.text_config_hidden_size = 4096
        self.pad_token_id = 0
        self.optimized = False


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 65535, "trace_region_size": 2, "num_command_queues": 1}],
    indirect=True,
)
def test_model_output(mesh_device):
    config = TestConfig()

    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    torch_model.eval()

    processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    tokenizer = processor.tokenizer

    ttnn_model = GraniteSpeech(device=mesh_device, config=config, tokenizer=tokenizer, torch_ref=torch_model)

    # load audio
    audio_path = hf_hub_download(repo_id="ibm-granite/granite-speech-3.3-8b", filename="10226_10111_000000.wav")
    wav, sr = torchaudio.load(audio_path, normalize=True)
    assert wav.shape[0] == 1 and sr == 16000  # mono, 16khz

    # create text prompt
    system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # run the processor+model
    model_inputs = processor(prompt, wav, device="cpu", return_tensors="pt").to("cpu")
    ttnn_model.forward(
        input_ids=model_inputs["input_ids"],
        input_features=model_inputs["input_features"].to(torch.bfloat16),
        input_features_mask=model_inputs["input_features_mask"],
    )
