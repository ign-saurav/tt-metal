import pytest  
import torch  
import ttnn 
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.granite_speech_33_8b.tt.ttnn_model import GraniteEncoderAndProjector, GraniteSpeech
import os


def calculate_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:  
    """Calculate Pearson Correlation Coefficient between two tensors."""  
    tensor1_flat = tensor1.flatten().float()  
    tensor2_flat = tensor2.flatten().float()  
      
    mean1 = tensor1_flat.mean()  
    mean2 = tensor2_flat.mean()  
      
    numerator = ((tensor1_flat - mean1) * (tensor2_flat - mean2)).sum()  
    denominator = torch.sqrt(((tensor1_flat - mean1)**2).sum() * ((tensor2_flat - mean2)**2).sum())  
      
    return (numerator / denominator).item() 

@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_encoder_and_projector_output(device):  
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16)
    config = torch_model.config
    torch_model.eval()  
      
    ttnn_model = GraniteEncoderAndProjector(device=device, config=config, include_conformer_layernorm=False, use_optimized_attention=True) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model
    )

    processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    tokenizer = processor.tokenizer

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
    model_inputs = processor(prompt, wav, device='cpu', return_tensors="pt").to('cpu')
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(input_ids=model_inputs['input_ids'],input_features=model_inputs['input_features'])  
      
    # TTNN forward pass  
    ttnn_input = ttnn.from_torch(  
        model_inputs['input_features'],
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_input) 
    ttnn_output = ttnn.to_torch(ttnn_output) 
      
    # Compare outputs  
    assert_with_pcc(torch_output, ttnn_output, pcc=0.96)  
    print(f"Encoder+Projector test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}") 


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 65535}],  
    indirect=True,  
)   
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_model_output(mesh_device):
    device = mesh_device  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16)
    config = torch_model.config
    torch_model.eval()  

    processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    tokenizer = processor.tokenizer
      
    ttnn_model = GraniteSpeech(device=device, config=config, tokenizer=tokenizer, torch_ref=torch_model) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model
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
    model_inputs = processor(prompt, wav, device='cpu', return_tensors="pt").to('cpu')
      
    # PyTorch forward pass  
    with torch.no_grad():   
        torch_model(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)
      
    # TTNN forward pass  
    ttnn_model.forward(input_ids=model_inputs['input_ids'], input_features=model_inputs['input_features'], input_features_mask=model_inputs['input_features_mask']) 