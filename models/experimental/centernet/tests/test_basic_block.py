# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.network.dlav0 import DLA, BasicBlock
from models.experimental.centernet.tt.basic_block import TtBasicBlock
from models.experimental.centernet.tt.custom_preprocessor import create_basic_block_preprocessor

WEIGHTS_PATH = "ctdet_coco_dlav0_1x.pth"


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_basic_block(device):
    """Test TtBasicBlock using pretrained DLA-34 weights from level2.tree1."""
    torch.manual_seed(42)

    dla_model = DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block=BasicBlock,
    )

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    base_state_dict = {k.replace("base.", ""): v for k, v in state_dict.items() if k.startswith("base.")}
    dla_model.load_state_dict(base_state_dict, strict=False)
    dla_model.eval()

    pytorch_basic_block = dla_model.level2.tree1
    assert isinstance(pytorch_basic_block, BasicBlock), f"Expected BasicBlock, got {type(pytorch_basic_block)}"

    inplanes = 32
    planes = 64
    stride = 2
    dilation = 1
    batch_size = 1
    input_height = 256
    input_width = 256

    torch_input = torch.randn(batch_size, inplanes, input_height, input_width)

    out_h = (input_height - 1) // stride + 1
    out_w = (input_width - 1) // stride + 1

    torch_residual = torch.randn(batch_size, planes, out_h, out_w)

    with torch.no_grad():
        pytorch_output = pytorch_basic_block(torch_input, residual=torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_basic_block,
        custom_preprocessor=create_basic_block_preprocessor(),
        device=None,
    )

    tt_model = TtBasicBlock(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        dilation=dilation,
        parameters=parameters,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    tt_residual = ttnn.from_torch(torch_residual.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_residual = ttnn.to_device(tt_residual, device)

    tt_output = tt_model.forward(tt_input, residual=tt_residual)

    tt_output_torch = tt2torch_tensor(tt_output).reshape(batch_size, -1, planes)
    pytorch_output_flat = pytorch_output.permute(0, 2, 3, 1).reshape(batch_size, -1, planes)

    passing, pcc_value = comp_pcc(pytorch_output_flat, tt_output_torch, pcc=0.99)

    logger.info(f"BasicBlock PCC: {pcc_value}")
    assert passing, f"PCC check failed: {pcc_value} = 0.99"
