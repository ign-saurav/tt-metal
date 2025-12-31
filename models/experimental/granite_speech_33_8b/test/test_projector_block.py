import pytest  
import torch  
import ttnn 

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.granite_speech_33_8B.tt.ttnn_projector_block import Blip2QFormerIntermediateTTNN, Blip2QFormerOutputTTNN, Blip2QFormerSelfOutputTTNN, Blip2QFormerMultiHeadAttentionTTNN, Blip2QFormerAttentionTTNN, Blip2QFormerLayerTTNN


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
def test_blip_intermediate(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    # torch_model = GraniteSpeechConformerFeedForward(config).to(torch.bfloat16)
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].intermediate_query   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerIntermediateTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.dense.weight,  
        torch_model.dense.bias
    )
      
    # Create test input  
    batch_size, seq_len, hidden_dim = 57, 3, 1024  
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(torch_input)  
      
    # TTNN forward pass  
    ttnn_input = ttnn.from_torch(  
        torch_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_input) 
    ttnn_output = ttnn.to_torch(ttnn_output) 
      
    # Compare outputs  
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)  
    print(f"BlipIntermediate test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")  


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    # torch_model = GraniteSpeechConformerFeedForward(config).to(torch.bfloat16)
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].output_query   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerOutputTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.dense.weight,  
        torch_model.dense.bias,
        torch_model.LayerNorm.weight,
        torch_model.LayerNorm.bias
    )
      
    # Create test input  
    batch_size, seq_len, hidden_dim, input_dim = 57, 3, 4096, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_input = torch.randn(batch_size, seq_len, input_dim, dtype=torch.bfloat16)  
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(torch_hidden_input, torch_input)  
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_input = ttnn.from_torch(  
        torch_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_hidden_input, ttnn_input) 
    ttnn_output = ttnn.to_torch(ttnn_output)  
      
    # Compare outputs  
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)  
    print(f"BlipOutput test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}") 


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_self_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    # torch_model = GraniteSpeechConformerFeedForward(config).to(torch.bfloat16)
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].attention.output   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerSelfOutputTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.dense.weight,  
        torch_model.dense.bias,
        torch_model.LayerNorm.weight,
        torch_model.LayerNorm.bias
    )
      
    # Create test input  
    batch_size, seq_len, hidden_dim= 57, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(torch_hidden_input, torch_input)  
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_input = ttnn.from_torch(  
        torch_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_hidden_input, ttnn_input) 
    ttnn_output = ttnn.to_torch(ttnn_output)  
      
    # Compare outputs  
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)  
    print(f"BlipOutput test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)
def test_blip_multi_head_attention_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].attention.attention   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerMultiHeadAttentionTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.query.weight,  
        torch_model.query.bias,
        torch_model.key.weight,
        torch_model.key.bias,
        torch_model.value.weight,
        torch_model.value.bias
    )
      
    # Create test input  
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim= 1, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)  
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(torch_hidden_input, torch_attn_mask_input) 
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0] 
    torch_output3 = torch_output[1][1] 
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_attn_mask_input = ttnn.from_torch(  
        torch_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(ttnn_hidden_input, ttnn_attn_mask_input)
    # ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0] 
    ttnn_output3 = ttnn_output[1][1]  
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)  
      
    # Compare outputs 
    # assert_with_pcc(torch_output, ttnn_output, pcc=0.97)  
    # print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")

    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.97)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")  

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}")  


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_multi_head_cross_attention_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].crossattention.attention   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerMultiHeadAttentionTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.query.weight,  
        torch_model.query.bias,
        torch_model.key.weight,
        torch_model.key.bias,
        torch_model.value.weight,
        torch_model.value.bias
    )
      
    # Create test input  
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim= 1, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input =  torch.zeros(57, 1, 1, 15, dtype=torch.bfloat16)
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(hidden_states=torch_hidden_input, attention_mask=torch_attn_mask_input, encoder_hidden_states=torch_encoder_hidden_input, encoder_attention_mask=torch_encoder_attn_mask_input) 
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0] 
    torch_output3 = torch_output[1][1] 
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_attn_mask_input = ttnn.from_torch(  
        torch_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_hidden_input = ttnn.from_torch(  
        torch_encoder_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_attn_mask_input = ttnn.from_torch(  
        torch_encoder_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input, attention_mask=ttnn_attn_mask_input, encoder_hidden_states=ttnn_encoder_hidden_input, encoder_attention_mask=ttnn_encoder_attn_mask_input)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0] 
    ttnn_output3 = ttnn_output[1][1]  
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3) 
      
    # Compare outputs 
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.97)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")  

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}") 


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_attention_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].attention   
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerAttentionTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.attention,
        torch_model.output
    )
      
    # Create test input  
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim= 1, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)  
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(hidden_states=torch_hidden_input, attention_mask=torch_attn_mask_input) 
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0] 
    torch_output3 = torch_output[1][1] 
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_attn_mask_input = ttnn.from_torch(  
        torch_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input, attention_mask=ttnn_attn_mask_input)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0] 
    ttnn_output3 = ttnn_output[1][1]  
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)  
      
    # Compare outputs 
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.8)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")  

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}")  


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_cross_attention_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0].crossattention 
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerAttentionTTNN(device=device, config=config) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model.attention,  
        torch_model.output,
    )
      
    # Create test input  
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim= 1, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_attn_mask_input = torch.ones(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input =  torch.ones(57, 1, 1, 15, dtype=torch.bfloat16)
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(hidden_states=torch_hidden_input, attention_mask=torch_attn_mask_input, encoder_hidden_states=torch_encoder_hidden_input, encoder_attention_mask=torch_encoder_attn_mask_input) 
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0] 
    torch_output3 = torch_output[1][1] 
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_attn_mask_input = ttnn.from_torch(  
        torch_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_hidden_input = ttnn.from_torch(  
        torch_encoder_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_attn_mask_input = ttnn.from_torch(  
        torch_encoder_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input, attention_mask=ttnn_attn_mask_input, encoder_hidden_states=ttnn_encoder_hidden_input, encoder_attention_mask=ttnn_encoder_attn_mask_input)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0] 
    ttnn_output3 = ttnn_output[1][1]  
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3) 
      
    # Compare outputs 
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.98)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")  

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}") 


@pytest.mark.parametrize(  
    "device_params",  
    [{"l1_small_size": 32767}],  
    indirect=True,  
)   
def test_blip_layer_output(device):  
    """Test FeedForward TTNN implementation against PyTorch."""  
    config = TestConfig()  
      
    # Initialize models  
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).projector.qformer.encoder.layer[0]
    torch_model.eval()  
      
    ttnn_model = Blip2QFormerLayerTTNN(device=device, config=config, layer_idx=0) 
      
    # Prepare weights  
    ttnn_model.prepare_weights(  
        torch_model
    )
      
    # Create test input  
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim= 1, 3, 1024  
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)  
    torch_attn_mask_input = torch.ones(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input =  torch.ones(57, 1, 1, 15, dtype=torch.bfloat16)
      
    # PyTorch forward pass  
    with torch.no_grad():  
        torch_output = torch_model(hidden_states=torch_hidden_input, attention_mask=torch_attn_mask_input, encoder_hidden_states=torch_encoder_hidden_input, encoder_attention_mask=torch_encoder_attn_mask_input, query_length=3) 
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0] 
    torch_output3 = torch_output[1][1] 
      
    # TTNN forward pass  
    ttnn_hidden_input = ttnn.from_torch(  
        torch_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_attn_mask_input = ttnn.from_torch(  
        torch_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_hidden_input = ttnn.from_torch(  
        torch_encoder_hidden_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_encoder_attn_mask_input = ttnn.from_torch(  
        torch_encoder_attn_mask_input,
        dtype=ttnn.bfloat16,  
        layout=ttnn.TILE_LAYOUT,  
        device=device  
    )  
    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input, attention_mask=ttnn_attn_mask_input, encoder_hidden_states=ttnn_encoder_hidden_input, encoder_attention_mask=ttnn_encoder_attn_mask_input, query_length=3)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0] 
    ttnn_output3 = ttnn_output[1][1]  
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3) 
      
    # Compare outputs 
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.95)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")  

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)  
    print(f"BlipMultiHeadAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}") 