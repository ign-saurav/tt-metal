# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.dlav0 import DLASeg
from models.experimental.centernet.tt.dla_seg import TtDLASeg
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_dla_seg(device):
    """Test TtDLASeg module with specific configuration."""
    torch.manual_seed(42)

    # DLASeg configuration from debug output
    heads = {"hm": 80, "wh": 2, "reg": 2}
    down_ratio = 4
    head_conv = 256
    input_shape = (1, 3, 512, 512)

    # Create PyTorch DLASeg module
    pytorch_dla_seg = DLASeg(
        base_name="dla34", heads=heads, pretrained=False, down_ratio=down_ratio, head_conv=head_conv
    )
    pytorch_dla_seg.eval()

    # Create random input
    torch_input = torch.randn(input_shape, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_dla_seg(torch_input)

    # Get mesh mappers
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_dla_seg,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=pytorch_dla_seg, run_model=lambda model: pytorch_dla_seg(torch_input), device=device
    )

    # Create TTNN DLASeg module
    tt_dla_seg = TtDLASeg(
        heads=heads,
        down_ratio=down_ratio,
        head_conv=head_conv,
        parameters=parameters.dla_seg,
        device=device,
        layer_args=parameters.layer_args,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    # TTNN forward pass
    tt_output = tt_dla_seg.forward(tt_input)

    # Convert each head output back to PyTorch format
    tt_output_torch = {}
    for head_name in tt_output[0]:
        head_output = tt2torch_tensor(tt_output[0][head_name]).permute(0, 3, 1, 2)
        tt_output_torch[head_name] = head_output

    # Compare outputs using PCC for each head
    all_passed = True
    for head_name in heads:
        if head_name in pytorch_output[0] and head_name in tt_output_torch:
            passing, pcc_value = comp_pcc(
                pytorch_output[0][head_name], tt_output_torch[head_name], pcc=0.97 if head_name == "reg" else 0.99
            )
            logger.info(f"DLASeg Head '{head_name}' PCC: {pcc_value}")
            if not passing:
                all_passed = False
                logger.warning(
                    f"Head '{head_name}' PCC check failed: {pcc_value} < {0.97 if head_name == 'reg' else 0.99}"
                )
        else:
            all_passed = False
            logger.error(f"Head '{head_name}' missing from outputs")

    assert all_passed, "One or more DLASeg heads failed PCC check"
