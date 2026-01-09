import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.granite_speech_33_8b.tt.ttnn_encoder_block import (
    GraniteSpeechConformerFeedForwardTTNN,
    GraniteSpeechConformerAttentionTTNN,
    GraniteSpeechConformerConvModuleTTNN,
    GraniteSpeechConformerBlockTTNN,
    GraniteSpeechCTCEncoderTTNN,
)
from transformers import AutoModelForSpeechSeq2Seq

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


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_feedforward(device):
    config = TestConfig()

    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).encoder.layers[0].ff1
    torch_model.eval()

    ttnn_model = GraniteSpeechConformerFeedForwardTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.pre_norm.weight,
        torch_model.pre_norm.bias,
        torch_model.up_proj.weight,
        torch_model.up_proj.bias,
        torch_model.down_proj.weight,
        torch_model.down_proj.bias
    )

    # Create test input
    batch_size, seq_len, hidden_dim = 1, 844, 1024
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
    print(f"FeedForward test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_attention(device):
    config = TestConfig()

    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).encoder.layers[0].attn
    torch_model.eval()

    ttnn_model = GraniteSpeechConformerAttentionTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.pre_norm.weight,
        torch_model.pre_norm.bias,
        torch_model.to_q.weight,
        torch_model.to_kv.weight,
        torch_model.to_out.weight,
        torch_model.to_out.bias,
        torch_model.rel_pos_emb.weight
    )

    # Create test input
    batch_size, seq_len, hidden_dim = 1, 844, 1024
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

    # Create attention distances
    attention_dists = torch.randint(
        0,
        config.max_pos_emb + 1,
        (config.context_size, config.context_size)
    )

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_input, attention_dists)

    # TTNN forward pass
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device
    )
    ttnn_output = ttnn_model.forward(ttnn_input, attention_dists)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Compare outputs
    assert_with_pcc(torch_output, ttnn_output, pcc=0.98)
    print(f"Attention test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 65535}],
    indirect=True,
)
def test_conv_module(device):
    config = TestConfig()

    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).encoder.layers[0].conv
    torch_model.eval()

    ttnn_model = GraniteSpeechConformerConvModuleTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.norm.weight,
        torch_model.norm.bias,
        torch_model.up_conv.weight,
        torch_model.down_conv.weight,
        torch_model.batch_norm.weight,
        torch_model.batch_norm.bias,
        torch_model.batch_norm.running_mean,
        torch_model.batch_norm.running_var,
        torch_model.depth_conv.conv.weight
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 844, 1024
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
    print(f"ConvModule test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 65535}],
    indirect=True,
)
def test_conformer_block(device):
    config = TestConfig()
    for i in range(config.num_layers):
        # Initialize models
        torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).encoder.layers[i]
        torch_model.eval()

        ttnn_model = GraniteSpeechConformerBlockTTNN(device=device, config=config, include_layernorm=False)

        # Prepare all weights
        ttnn_model.prepare_weights(
            torch_model.ff1,
            torch_model.ff2,
            torch_model.attn,
            torch_model.conv,
            torch_model.post_norm
        )

        # Create test input
        batch_size, seq_len = 1, 844
        torch_input = torch.randn(batch_size, seq_len, config.hidden_dim, dtype=torch.bfloat16)

        # Create attention distances
        attention_dists = torch.randint(
            0,
            config.max_pos_emb + 1,
            (config.context_size, config.context_size)
        )

        # PyTorch forward pass
        with torch.no_grad():
            torch_output = torch_model(torch_input, attention_dists)

        # TTNN forward pass
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )
        ttnn_output_tensor = ttnn_model.forward(ttnn_input, attention_dists)
        ttnn_output = ttnn.to_torch(ttnn_output_tensor)

        # Compare outputs
        assert_with_pcc(torch_output, ttnn_output, pcc=0.98)
        print(f"ConformerBlock test passed with PCC for block {i}: {calculate_pcc(torch_output, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 65535}],
    indirect=True,
)
def test_encoder_block(device):
    config = TestConfig()

    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16).encoder
    torch_model.eval()

    ttnn_model = GraniteSpeechCTCEncoderTTNN(device=device, config=config, include_conformer_layernorm=False)

    # Prepare all weights
    ttnn_model.prepare_weights(torch_model)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len = 1, 844
    torch_input = torch.randn(batch_size, seq_len, config.input_dim, dtype=torch.bfloat16)

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
    ttnn_output_tensor = ttnn_model.forward(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output_tensor)

    # Compare outputs
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    print(f"EncoderBlock test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")

def calculate_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate Pearson Correlation Coefficient between two tensors."""
    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()

    mean1 = tensor1_flat.mean()
    mean2 = tensor2_flat.mean()

    numerator = ((tensor1_flat - mean1) * (tensor2_flat - mean2)).sum()
    denominator = torch.sqrt(((tensor1_flat - mean1)**2).sum() * ((tensor2_flat - mean2)**2).sum())

    return (numerator / denominator).item()  