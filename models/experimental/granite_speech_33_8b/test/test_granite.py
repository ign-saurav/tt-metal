import pytest
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from models.experimental.granite_speech_33_8b.tt.granite_speech import GraniteSpeechTTNN


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 65535, "trace_region_size": 2, "num_command_queues": 1}],
    indirect=True,
)
def test_model_output(mesh_device):
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model.eval()

    processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    tokenizer = processor.tokenizer

    # TODO: Include LoRA Adapters in tt_transformers.
    ttnn_model = GraniteSpeechTTNN(
        mesh_device=mesh_device,
        config=config,
        tokenizer=tokenizer,
        torch_ref=torch_model,
        use_torch_audio_feat=False,
        include_conformer_layernorm=True,  # Valid only if use_torch_audio_feat is False
        use_optimized_attention_projector=True,  # Valid only if use_torch_audio_feat is False
    )

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
        input_features=model_inputs["input_features"],
        input_features_mask=model_inputs["input_features_mask"],
    )
