# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc, tt2torch_tensor
from models.experimental.centernet.reference.network.dlav0 import get_pose_net
from models.experimental.centernet.tt.tt_head import TtCenterNetHead
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor

WEIGHTS_PATH = "models/experimental/centernet/ctdet_coco_dlav0_1x.pth"


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}],
    indirect=True,
)
def test_centernet_heads(device):
    torch.manual_seed(42)

    heads = {
        "hm": 80,
        "wh": 2,
        "reg": 2,
    }

    pytorch_model = get_pose_net(num_layers=34, heads=heads, head_conv=256)

    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(f"Checkpoint file {WEIGHTS_PATH} not found")

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pytorch_model.load_state_dict(state_dict, strict=False)
    pytorch_model.eval()

    batch_size = 1
    in_channels = 64
    input_height = 128
    input_width = 128

    torch_input = torch.randn(batch_size, in_channels, input_height, input_width)
    tt_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    tt_input = ttnn.to_device(tt_input, device)

    import pdb

    pdb.set_trace()
    for head_name, num_classes in heads.items():
        logger.info(f"Testing {head_name} head with {num_classes} classes")

        pytorch_head = getattr(pytorch_model, head_name)

        with torch.no_grad():
            pytorch_output = pytorch_head(torch_input)

        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_head,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=device,
        )

        tt_model = TtCenterNetHead(
            in_channels=in_channels,
            head_config={head_name: num_classes},
            head_conv=256,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
        )

        tt_output = tt_model.forward(tt_input)

        tt_output_torch = tt2torch_tensor(tt_output).permute(0, 3, 1, 2)
        pytorch_output_flat = pytorch_output.reshape(batch_size, -1)
        tt_output_flat = tt_output_torch.reshape(batch_size, -1)

        passing, pcc_value = comp_pcc(pytorch_output_flat, tt_output_flat, pcc=0.99)

        logger.info(f"{head_name} Head PCC: {pcc_value}")
        assert passing, f"{head_name} head PCC check failed: {pcc_value} < 0.99"

    logger.info("All CenterNet heads passed PCC validation!")
