"""
Test SAM image encoder inside DeepSeek-OCR: same init as ocr_infer, then sam_model(input) for 2 input sizes.
"""
import pytest
import torch
from transformers import AutoModel


MODEL_NAME = "deepseek-ai/DeepSeek-OCR"


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model exactly as in ocr_infer.py (HuggingFace cache, etc.)."""
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


@pytest.mark.parametrize("image_size", [640, 1024])
def test_sam_model_forward(ocr_model, image_size):
    """Run sam_model(input) with two parameterized input sizes."""
    sam_model = ocr_model.model.sam_model
    device = next(sam_model.parameters()).device
    # (1, 3, H, W) same dtype as model
    x = torch.randn(1, 3, image_size, image_size, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = sam_model(x)
    assert out.dtype == torch.bfloat16
    assert out.dim() == 4
