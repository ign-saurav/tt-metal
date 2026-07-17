# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.dlav0 import BasicBlock, Tree
from models.experimental.centernet.tt.basic_block import TtBasicBlock
from models.experimental.centernet.tt.tree import TtTree
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "levels,in_channels,out_channels,stride,level_root,root_residual,input_shape",
    [
        # Case 1: Level 1 tree
        (1, 32, 64, 2, False, False, (1, 32, 256, 256)),
        # Case 2: Level 2 tree
        (2, 64, 128, 2, False, False, (1, 64, 128, 128)),
        # Case 3: Level 2 tree with larger channels
        (2, 128, 256, 2, False, False, (1, 128, 64, 64)),
        # Case 4: Level 1 tree with large channels (uses BLOCK_SHARDED strategy like VGG16)
        (1, 256, 512, 2, False, False, (1, 256, 16, 16)),
    ],
)
def test_tree(device, levels, in_channels, out_channels, stride, level_root, root_residual, input_shape):
    """Test TtTree module with various configurations."""
    torch.manual_seed(42)

    # Create PyTorch Tree module
    pytorch_tree = Tree(
        levels=levels,
        block=BasicBlock,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        level_root=level_root,
        root_residual=root_residual,
    )
    pytorch_tree.eval()

    # Create random input
    torch_input = torch.randn(input_shape, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_tree(torch_input)

    # Get mesh mappers
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_tree,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=pytorch_tree, run_model=lambda model: pytorch_tree(torch_input), device=device
    )

    # Create TTNN Tree module

    tt_tree = TtTree(
        levels=levels,
        block=TtBasicBlock,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        level_root=level_root,
        root_residual=root_residual,
        parameters=parameters.tree,
        device=device,
        layer_args=parameters.layer_args,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    # TTNN forward pass
    tt_output = tt_tree.forward(tt_input)

    # Convert output back to PyTorch format
    tt_output_torch = tt2torch_tensor(tt_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    passing, pcc_value = comp_pcc(pytorch_output, tt_output_torch, pcc=0.99)

    logger.info(f"Tree PCC (levels={levels}, in={in_channels}, out={out_channels}): {pcc_value}")
    assert passing, f"PCC check failed: {pcc_value} < 0.99"
