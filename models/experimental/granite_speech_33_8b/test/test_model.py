import pytest  
import torch  
import ttnn 
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.granite_speech_33_8b.tt.ttnn_model import GraniteEncoderAndProjector


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
        self.optimized = False


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
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16)
    torch_model.eval()  
      
    ttnn_model = GraniteEncoderAndProjector(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model
    )
      
    # Create test input 
    torch.manual_seed(0) 
    batch_size, seq_len, hidden_dim = 1, 844, 160  
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_input_ids = torch.randint(0, 100, (batch_size, seq_len, hidden_dim), dtype=torch.int32)  

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
        # torch_output = torch_model(input_ids=torch_input_ids,input_features=torch_input)
        torch_output = torch_model(input_ids=model_inputs['input_ids'],input_features=model_inputs['input_features'])  
      
    # TTNN forward pass  
    ttnn_input = ttnn.from_torch(  
        # torch_input,
        model_inputs['input_features'],
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_input) 
    ttnn_output = ttnn.to_torch(ttnn_output) 
      
    # Compare outputs  
    assert_with_pcc(torch_output, ttnn_output, pcc=0.96)  
    print(f"BlipIntermediate test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}") 