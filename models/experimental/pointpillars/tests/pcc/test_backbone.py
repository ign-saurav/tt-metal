# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.backbone import TtBackbone
from models.experimental.pointpillars.reference.pointpillars import Backbone
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.pointpillars.common import load_checkpoint, extract_component_state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channel,out_channels,layer_nums,layer_strides",
    [
        (64, [64, 128, 256], [3, 5, 5], [2, 2, 2]),
    ],
)
def test_backbone(device, in_channel, out_channels, layer_nums, layer_strides, reset_seeds):
    """Test TtBackbone against PyTorch reference."""
    torch.manual_seed(0)

    torch_model = Backbone(in_channel, out_channels, layer_nums, layer_strides)

    state_dict = load_checkpoint("epoch_160.pth")
    if state_dict is not None:
        backbone_state_dict = extract_component_state_dict(state_dict, "backbone.")
        torch_model.load_state_dict(backbone_state_dict)

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    batch_size = 1
    height = 496
    width = 432
    torch_input = torch.randn(batch_size, in_channel, height, width, dtype=torch.bfloat16)

    torch_output = torch_model(torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(batch_size, 1, height * width, in_channel),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_model = TtBackbone(
        in_channel=in_channel,
        out_channels=out_channels,
        layer_nums=layer_nums,
        layer_strides=layer_strides,
        parameters=parameters["backbone"],
        device=device,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
    )

    tt_output = tt_model.forward(ttnn_input)

    all_passing = True
    for i, (torch_out, tt_out) in enumerate(zip(torch_output, tt_output)):
        tt_out_torch = tt2torch_tensor(tt_out)
        tt_out_torch = tt_out_torch.reshape(
            torch_out.shape[0], torch_out.shape[2], torch_out.shape[3], torch_out.shape[1]
        )
        tt_out_torch = tt_out_torch.permute(0, 3, 1, 2)

        passing, pcc = comp_pcc(torch_out, tt_out_torch, 0.99)
        logger.info(f"Block {i} PCC: {pcc}")
        if not passing:
            all_passing = False
            logger.warning(f"Block {i} PCC check failed: {pcc}")

    assert all_passing, "Backbone test failed - PCC < 0.99"
    logger.info("Backbone test passed!")
