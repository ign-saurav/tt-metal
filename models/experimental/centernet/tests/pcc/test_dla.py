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
from models.experimental.centernet.tt.dla import TtDLA
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_dla(device):
    """Test TtDLA module with specific configuration."""
    torch.manual_seed(42)

    # DLA configuration from debug output
    levels = [1, 1, 1, 2, 2, 1]
    channels = [16, 32, 64, 128, 256, 512]
    num_classes = 1000
    residual_root = False
    return_levels = True
    pool_size = 7
    linear_root = False
    input_shape = (1, 3, 512, 512)

    # Create PyTorch DLA module
    pytorch_dla = DLA(
        levels=levels,
        channels=channels,
        num_classes=num_classes,
        block=BasicBlock,
        residual_root=residual_root,
        return_levels=return_levels,
        pool_size=pool_size,
        linear_root=linear_root,
    )
    pytorch_dla.load_pretrained_model(data="imagenet", name="dla34", hash="ba72cf86")
    pytorch_dla.eval()

    # Create random input
    torch_input = torch.randn(input_shape, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_dla(torch_input)

    # Get mesh mappers
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_dla,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.layer_args = {}
    parameters.layer_args = infer_ttnn_module_args(
        model=pytorch_dla, run_model=lambda model: pytorch_dla(torch_input), device=device
    )

    # Create TTNN DLA module
    tt_dla = TtDLA(
        levels=levels,
        channels=channels,
        num_classes=num_classes,
        block=TtBasicBlock,
        residual_root=residual_root,
        return_levels=return_levels,
        pool_size=pool_size,
        linear_root=linear_root,
        parameters=parameters.dla,
        device=device,
        layer_args=parameters.layer_args,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = tt_dla.forward(tt_input)

    tt_output_torch = []
    for level_output in tt_output:
        level_torch = tt2torch_tensor(level_output).permute(0, 3, 1, 2)
        tt_output_torch.append(level_torch)

    # Compare outputs using PCC for each level
    all_passed = True
    for i, (pytorch_level, tt_level) in enumerate(zip(pytorch_output, tt_output_torch)):
        passing, pcc_value = comp_pcc(pytorch_level, tt_level, pcc=0.98)
        logger.info(f"DLA Level {i} PCC: {pcc_value}")
        if not passing:
            all_passed = False
            logger.warning(f"Level {i} PCC check failed: {pcc_value} < 0.98")

    assert all_passed, "One or more DLA levels failed PCC check"
