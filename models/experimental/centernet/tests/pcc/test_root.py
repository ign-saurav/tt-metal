# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.dlav0 import Root
from models.experimental.centernet.tt.root import TtRoot
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,input_shapes",
    [
        # Case 1: 2 inputs, 128->64
        (128, 64, 1, [(1, 64, 128, 128), (1, 64, 128, 128)]),
        # Case 2: 2 inputs, 256->128
        (256, 128, 1, [(1, 128, 64, 64), (1, 128, 64, 64)]),
        # Case 3: 4 inputs, 448->128
        (448, 128, 1, [(1, 128, 64, 64), (1, 128, 64, 64), (1, 64, 64, 64), (1, 128, 64, 64)]),
        # Case 4: 2 inputs, 512->256
        (512, 256, 1, [(1, 256, 32, 32), (1, 256, 32, 32)]),
        # Case 5: 4 inputs, 896->256
        (896, 256, 1, [(1, 256, 32, 32), (1, 256, 32, 32), (1, 128, 32, 32), (1, 256, 32, 32)]),
        # Case 6: 3 inputs, 1280->512
        (1280, 512, 1, [(1, 512, 16, 16), (1, 512, 16, 16), (1, 256, 16, 16)]),
    ],
)
def test_root(device, in_channels, out_channels, kernel_size, input_shapes):
    """Test TtRoot module with various input configurations."""
    torch.manual_seed(42)

    # Create PyTorch Root module
    pytorch_root = Root(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, residual=False)

    assert isinstance(pytorch_root, Root), f"Expected BasicBlock, got {type(pytorch_root)}"

    # Create random inputs
    torch_inputs = [torch.randn(shape, dtype=torch.float32) for shape in input_shapes]

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_root(*torch_inputs)

    # Get mesh mappers
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_root,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=pytorch_root, run_model=lambda model: pytorch_root(*torch_inputs), device=device
    )

    tt_root = TtRoot(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        residual=False,
        parameters=parameters.root,
        device=device,
        layer_args=parameters.layer_args,
    )

    # Convert inputs to TTNN format (NHWC)
    tt_inputs = []
    for torch_input in torch_inputs:
        tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        tt_input = ttnn.to_device(tt_input, device)
        tt_inputs.append(tt_input)

    # TTNN forward pass
    tt_output = tt_root.forward(*tt_inputs)

    # Convert output back to PyTorch format
    tt_output_torch = tt2torch_tensor(tt_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    passing, pcc_value = comp_pcc(pytorch_output, tt_output_torch, pcc=0.9)

    logger.info(f"Root PCC (in={in_channels}, out={out_channels}): {pcc_value}")
    assert passing, f"PCC check failed: {pcc_value} < 0.99"
