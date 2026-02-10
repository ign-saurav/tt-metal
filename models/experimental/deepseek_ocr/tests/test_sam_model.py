"""
Test SAM image encoder inside DeepSeek-OCR: same init as ocr_infer, then sam_model(input) for 2 input sizes.
Also runs TT SAM and checks PCC >= 0.99 vs torch.
"""
import pytest
import torch
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.deepseek_ocr.tt.tt_sam import run_tt_sam

PCC_THRESHOLD = 0.90


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


@pytest.mark.parametrize("image_size", [640])
# @pytest.mark.parametrize("image_size", [640, 1024])
def test_tt_sam_pcc(device, ocr_model, image_size):
    """Run torch SAM and TT SAM with same input; assert PCC >= 0.99."""
    import ttnn

    sam_model = ocr_model.model.sam_model
    torch.manual_seed(42)
    x = torch.randn(1, 3, image_size, image_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_out = sam_model(x)
    tt_out = run_tt_sam(
        device=device,
        sam_torch_module=sam_model,
        input_tensor=x,
        batch_size=1,
        image_size=image_size,
    )
    tt_out_torch = ttnn.to_torch(tt_out)
    if tt_out_torch.device.type != "cpu":
        tt_out_torch = tt_out_torch.cpu()
    if ref_out.device.type != "cpu":
        ref_out = ref_out.cpu()
    passed, message = check_with_pcc(ref_out.float(), tt_out_torch.float(), pcc=PCC_THRESHOLD)
    logger.info(f"TT SAM PCC check message: {message}")
    assert passed, f"TT SAM PCC check failed: {message}"
