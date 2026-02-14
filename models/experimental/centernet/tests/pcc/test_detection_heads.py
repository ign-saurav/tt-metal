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
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.centernet.tt.dla_seg import TtDLASeg


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "head_name,expected_pcc",
    [
        ("hm", 0.99),  # Heatmap head
        ("wh", 0.99),  # Width/Height head
        ("reg", 0.99),  # Regression/Offset head
    ],
)
def test_detection_head(device, head_name, expected_pcc):
    """Test individual detection head in isolation using actual DLASeg heads."""
    torch.manual_seed(42)

    # Create full DLASeg model to get properly initialized heads
    heads = {"hm": 80, "wh": 2, "reg": 2}
    down_ratio = 4
    head_conv = 256

    dla_seg = DLASeg(base_name="dla34", heads=heads, pretrained=False, down_ratio=down_ratio, head_conv=head_conv)
    dla_seg.eval()

    # Extract the specific head we want to test
    pytorch_head = getattr(dla_seg, head_name)

    in_channels = dla_seg.base.channels[dla_seg.first_level]

    # Input shape: after DLA upsampling, we have 128x128 feature maps
    input_shape = (1, in_channels, 128, 128)

    # Create random input
    torch_input = torch.randn(input_shape, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_head(torch_input)

    # Get mesh mappers
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess full DLASeg model
    dummy_input = torch.randn(1, 3, 512, 512)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: dla_seg,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.layer_args = infer_ttnn_module_args(
        model=dla_seg, run_model=lambda model: dla_seg(dummy_input), device=device
    )

    # Create full TtDLASeg model
    tt_dla_seg = TtDLASeg(
        heads=heads,
        down_ratio=down_ratio,
        head_conv=head_conv,
        parameters=parameters.dla_seg,
        device=device,
        layer_args=parameters.layer_args,
    )

    # Extract the specific head from TtDLASeg
    ttnn_head = getattr(tt_dla_seg, head_name)

    # Convert input to TTNN format
    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    # TTNN forward pass
    if isinstance(ttnn_head, list):
        tt_output = ttnn_head[0](tt_input)
        tt_output = ttnn_head[1](tt_output)
    else:
        tt_output = ttnn_head(tt_input)

    # Convert output back to PyTorch format
    tt_output_torch = tt2torch_tensor(tt_output)

    # Convert to NCHW format
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)

    # Compare outputs using PCC
    passing, pcc_value = comp_pcc(pytorch_output, tt_output_torch, pcc=expected_pcc)

    logger.info(f"Detection Head '{head_name}' PCC: {pcc_value}")

    assert passing, f"PCC check failed for {head_name} head: {pcc_value} < {expected_pcc}"
