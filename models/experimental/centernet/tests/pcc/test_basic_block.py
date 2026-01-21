# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.dlav0 import DLA, BasicBlock
from models.experimental.centernet.tt.basic_block import TtBasicBlock
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor


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

    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_basic_block,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=pytorch_basic_block,
        run_model=lambda model: pytorch_basic_block(torch_input, residual=torch_residual),
        device=device,
    )

    tt_model = TtBasicBlock(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        dilation=dilation,
        parameters=parameters.basic_block,
        device=device,
        layer_args=parameters.layer_args,
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
