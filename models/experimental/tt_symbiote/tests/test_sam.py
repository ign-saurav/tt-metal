# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

# from models.experimental.deepseek_ocr.tt.tt_sam import run_tt_sam

from models.experimental.tt_symbiote.modules.activation import TTNNSilu, TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.run_config import DispatchManager

from torch.nn import functional as F

from tqdm import tqdm
import torch.profiler


def get_abs_pos_sam(abs_pos, tgt_size):
    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class LayerNorm2d(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.weight = old_layer.weight
        self.bias = old_layer.bias
        self.eps = old_layer.eps

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(3, keepdim=True)
        s = (x - u).pow(2).mean(3, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class ImageEncoderViT(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.img_size = old_layer.img_size

        self.patch_embed = old_layer.patch_embed

        self.pos_embed = old_layer.pos_embed

        self.blocks = old_layer.blocks

        self.neck = nn.Sequential(
            *[l if isinstance(l, nn.Conv2d) else LayerNorm2d(l) for l in old_layer.neck.children()]
        )

        self.net_2 = old_layer.net_2
        self.net_3 = old_layer.net_3

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).permute(0, 3, 1, 2)  # BCHW -> BHWC
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)
        x2 = self.net_2(x)
        x3 = self.net_3(x2)
        return x3.permute(0, 3, 1, 2)  # BHWC -> BCHW


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model (HuggingFace); SAM is ocr_model.model.sam_model."""
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


class SAMWrapper(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x):
        return self.sam_model(x)


@pytest.mark.parametrize("image_size", [640])
# @pytest.mark.parametrize("image_size", [640, 1024])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_tt_sam_pcc(device, ocr_model, image_size):
    """Run torch SAM and TT SAM with same input; assert PCC >= PCC_THRESHOLD."""

    sam_model = ocr_model.model.sam_model
    model = SAMWrapper(sam_model)
    torch.manual_seed(42)
    # x = torch.ones(6, image_size, image_size, 3, dtype=torch.bfloat16)
    x = torch.load("sam_input.pt")
    ref_out = sam_model(x.permute(0, 3, 1, 2))
    nn_to_nn = {
        model.sam_model.__class__: ImageEncoderViT,
    }

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        nn.Conv2d: TTNNConv2dNHWC,
    }

    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = model(x)

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT SAM PCC (image_size={image_size}): {message}")
    assert passed, f"TT SAM PCC check failed: {message}"
