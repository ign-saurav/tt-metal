import pytest
import torch
import ttnn

from transformers import AutoModelForSpeechSeq2Seq
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.granite_speech_33_8b.tt.ttnn_projector_block import (
    Blip2QFormerIntermediateTTNN,
    Blip2QFormerOutputTTNN,
    Blip2QFormerSelfOutputTTNN,
    Blip2QFormerMultiHeadAttentionTTNN,
    Blip2QFormerAttentionTTNN,
    Blip2QFormerLayerTTNN,
    Blip2QFormerEncoderTTNN,
    Blip2QFormerModelTTNN,
    GraniteSpeechEncoderProjectorTTNN,
)


def calculate_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate Pearson Correlation Coefficient between two tensors."""
    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()

    mean1 = tensor1_flat.mean()
    mean2 = tensor2_flat.mean()

    numerator = ((tensor1_flat - mean1) * (tensor2_flat - mean2)).sum()
    denominator = torch.sqrt(((tensor1_flat - mean1) ** 2).sum() * ((tensor2_flat - mean2) ** 2).sum())

    return (numerator / denominator).item()


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_blip_intermediate(device):
    """Test Blip2QFormerIntermediate TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].intermediate_query
    torch_model.eval()

    ttnn_model = Blip2QFormerIntermediateTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model.dense.weight, torch_model.dense.bias)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 57, 3, 1024
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    # TTNN forward pass
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
    """Test Blip2QFormerOutput TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].output_query
    torch_model.eval()

    ttnn_model = Blip2QFormerOutputTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.dense.weight, torch_model.dense.bias, torch_model.LayerNorm.weight, torch_model.LayerNorm.bias
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim, input_dim = 57, 3, 4096, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_input = torch.randn(batch_size, seq_len, input_dim, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_hidden_input, torch_input)

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
    """Test Blip2QFormerSelfOutput TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].attention.output
    torch_model.eval()

    ttnn_model = Blip2QFormerSelfOutputTTNN(device=device, config=config)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.dense.weight, torch_model.dense.bias, torch_model.LayerNorm.weight, torch_model.LayerNorm.bias
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 57, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_hidden_input, torch_input)

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
    """Test Blip2QFormerMultiHeadAttention TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].attention.attention
    torch_model.eval()

    ttnn_model = Blip2QFormerMultiHeadAttentionTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.query.weight,
        torch_model.query.bias,
        torch_model.key.weight,
        torch_model.key.bias,
        torch_model.value.weight,
        torch_model.value.bias,
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_hidden_input, torch_attn_mask_input)
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0]
    torch_output3 = torch_output[1][1]

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(ttnn_hidden_input, ttnn_attn_mask_input)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0]
    ttnn_output3 = ttnn_output[1][1]
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)

    # Compare outputs
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.99)
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
    """Test Blip2QFormerMultiHeadAttention cross-attention TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].crossattention.attention
    torch_model.eval()

    ttnn_model = Blip2QFormerMultiHeadAttentionTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.query.weight,
        torch_model.query.bias,
        torch_model.key.weight,
        torch_model.key.bias,
        torch_model.value.weight,
        torch_model.value.bias,
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input = torch.zeros(57, 1, 1, 15, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(
            hidden_states=torch_hidden_input,
            attention_mask=torch_attn_mask_input,
            encoder_hidden_states=torch_encoder_hidden_input,
            encoder_attention_mask=torch_encoder_attn_mask_input,
        )
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0]
    torch_output3 = torch_output[1][1]

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_hidden_input = ttnn.from_torch(
        torch_encoder_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_attn_mask_input = ttnn.from_torch(
        torch_encoder_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(
        hidden_states=ttnn_hidden_input,
        attention_mask=ttnn_attn_mask_input,
        encoder_hidden_states=ttnn_encoder_hidden_input,
        encoder_attention_mask=ttnn_encoder_attn_mask_input,
    )
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0]
    ttnn_output3 = ttnn_output[1][1]
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)

    # Compare outputs
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.99)
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
    """Test Blip2QFormerAttention TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].attention
    torch_model.eval()

    ttnn_model = Blip2QFormerAttentionTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model.attention, torch_model.output)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.zeros(1, 1, 1, 3, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(hidden_states=torch_hidden_input, attention_mask=torch_attn_mask_input)
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0]
    torch_output3 = torch_output[1][1]

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input, attention_mask=ttnn_attn_mask_input)
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0]
    ttnn_output3 = ttnn_output[1][1]
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)

    # Compare outputs
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.99)
    print(f"BlipAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)
    print(f"BlipAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)
    print(f"BlipAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_blip_cross_attention_output(device):
    """Test Blip2QFormerAttention cross-attention TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0].crossattention
    torch_model.eval()

    ttnn_model = Blip2QFormerAttentionTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(
        torch_model.attention,
        torch_model.output,
    )

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.ones(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input = torch.ones(57, 1, 1, 15, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(
            hidden_states=torch_hidden_input,
            attention_mask=torch_attn_mask_input,
            encoder_hidden_states=torch_encoder_hidden_input,
            encoder_attention_mask=torch_encoder_attn_mask_input,
        )
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0]
    torch_output3 = torch_output[1][1]

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_hidden_input = ttnn.from_torch(
        torch_encoder_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_attn_mask_input = ttnn.from_torch(
        torch_encoder_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(
        hidden_states=ttnn_hidden_input,
        attention_mask=ttnn_attn_mask_input,
        encoder_hidden_states=ttnn_encoder_hidden_input,
        encoder_attention_mask=ttnn_encoder_attn_mask_input,
    )
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0]
    ttnn_output3 = ttnn_output[1][1]
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)

    # Compare outputs
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.99)
    print(f"BlipCrossAttn test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)
    print(f"BlipCrossAttn test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)
    print(f"BlipCrossAttn test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_blip_layer_output(device):
    """Test Blip2QFormerLayer TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder.layer[0]
    torch_model.eval()

    ttnn_model = Blip2QFormerLayerTTNN(device=device, config=config, layer_idx=0, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.ones(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input = torch.ones(57, 1, 1, 15, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(
            hidden_states=torch_hidden_input,
            attention_mask=torch_attn_mask_input,
            encoder_hidden_states=torch_encoder_hidden_input,
            encoder_attention_mask=torch_encoder_attn_mask_input,
            query_length=3,
        )
    torch_output1 = torch_output[0]
    torch_output2 = torch_output[1][0]
    torch_output3 = torch_output[1][1]

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_hidden_input = ttnn.from_torch(
        torch_encoder_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_attn_mask_input = ttnn.from_torch(
        torch_encoder_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(
        hidden_states=ttnn_hidden_input,
        attention_mask=ttnn_attn_mask_input,
        encoder_hidden_states=ttnn_encoder_hidden_input,
        encoder_attention_mask=ttnn_encoder_attn_mask_input,
        query_length=3,
    )
    ttnn_output1 = ttnn_output[0]
    ttnn_output2 = ttnn_output[1][0]
    ttnn_output3 = ttnn_output[1][1]
    ttnn_output1 = ttnn.to_torch(ttnn_output1)
    ttnn_output2 = ttnn.to_torch(ttnn_output2)
    ttnn_output3 = ttnn.to_torch(ttnn_output3)

    # Compare outputs
    assert_with_pcc(torch_output1, ttnn_output1, pcc=0.99)
    print(f"BlipLayer test passed with PCC: {calculate_pcc(torch_output1, ttnn_output1):.4f}")

    assert_with_pcc(torch_output2, ttnn_output2, pcc=0.99)
    print(f"BlipLayer test passed with PCC: {calculate_pcc(torch_output2, ttnn_output2):.4f}")

    assert_with_pcc(torch_output3, ttnn_output3, pcc=0.99)
    print(f"BlipLayer test passed with PCC: {calculate_pcc(torch_output3, ttnn_output3):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_blip_encoder_output(device):
    """Test Blip2QFormerEncoder TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer.encoder
    torch_model.eval()

    ttnn_model = Blip2QFormerEncoderTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_attn_mask_input = torch.ones(1, 1, 1, 3, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.randn(57, 15, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_attn_mask_input = torch.ones(57, 1, 1, 15, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(
            hidden_states=torch_hidden_input,
            attention_mask=torch_attn_mask_input,
            encoder_hidden_states=torch_encoder_hidden_input,
            encoder_attention_mask=torch_encoder_attn_mask_input,
            query_length=3,
        )

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attn_mask_input = ttnn.from_torch(
        torch_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_hidden_input = ttnn.from_torch(
        torch_encoder_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_encoder_attn_mask_input = ttnn.from_torch(
        torch_encoder_attn_mask_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(
        hidden_states=ttnn_hidden_input,
        attention_mask=ttnn_attn_mask_input,
        encoder_hidden_states=ttnn_encoder_hidden_input,
        encoder_attention_mask=ttnn_encoder_attn_mask_input,
        query_length=3,
    )
    ttnn_output = ttnn.to_torch(ttnn_output.last_hidden_state)

    # Compare outputs
    assert_with_pcc(torch_output.last_hidden_state, ttnn_output, pcc=0.99)
    print(f"BlipEncoder test passed with PCC: {calculate_pcc(torch_output.last_hidden_state, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_blip_model_output(device):
    """Test Blip2QFormerModel TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector.qformer
    torch_model.eval()

    ttnn_model = Blip2QFormerModelTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 3, 1024
    torch_hidden_input = torch.rand(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_encoder_hidden_input = torch.rand(57, 15, hidden_dim, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(query_embeds=torch_hidden_input, encoder_hidden_states=torch_encoder_hidden_input)

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_encoder_hidden_input = ttnn.from_torch(
        torch_encoder_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_output = ttnn_model.forward(query_embeds=ttnn_hidden_input, encoder_hidden_states=ttnn_encoder_hidden_input)
    ttnn_output = ttnn.to_torch(ttnn_output.last_hidden_state)

    # Compare outputs
    assert_with_pcc(torch_output.last_hidden_state, ttnn_output, pcc=0.99)
    print(f"BlipQFormerModel test passed with PCC: {calculate_pcc(torch_output.last_hidden_state, ttnn_output):.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32767}],
    indirect=True,
)
def test_projector_output(device):
    """Test GraniteSpeechEncoderProjector TTNN implementation against PyTorch."""
    # Initialize models
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    config = torch_model.config
    torch_model = torch_model.projector
    torch_model.eval()

    ttnn_model = GraniteSpeechEncoderProjectorTTNN(device=device, config=config, use_optimized_attention=True)

    # Prepare weights
    ttnn_model.prepare_weights(torch_model)

    # Create test input
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim = 1, 844, 1024
    torch_hidden_input = torch.rand(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_model(hidden_states=torch_hidden_input)

    # TTNN forward pass
    ttnn_hidden_input = ttnn.from_torch(torch_hidden_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn_model.forward(hidden_states=ttnn_hidden_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Compare outputs
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    print(f"Projector test passed with PCC: {calculate_pcc(torch_output, ttnn_output):.4f}")
