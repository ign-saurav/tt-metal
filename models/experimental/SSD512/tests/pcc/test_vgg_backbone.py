# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

from models.experimental.SSD512.common import SSD512_L1_SMALL_SIZE
from models.experimental.SSD512.reference.ssd import vgg, base
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import TtVggBackbone
from models.experimental.SSD512.tt.utils import extract_vgg_parameters_from_torch
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("pcc", ((0.99),))
@pytest.mark.parametrize("size", ((512,)))
@pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_vgg_backbone(device, pcc, size, reset_seeds):
    cfg = base[str(size)]
    torch_layers = vgg(cfg, i=3, batch_norm=False)
    torch_model = nn.Sequential(*torch_layers)
    torch_model.eval()

    batch_size = 1
    input_channels = 3
    torch_input = torch.randn(batch_size, input_channels, size, size)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    parameters = extract_vgg_parameters_from_torch(torch_model)

    tt_vgg = TtVggBackbone(
        size=size, input_channels=input_channels, batch_size=batch_size, parameters=parameters, device=device
    )

    tt_output_ttnn = tt_vgg(torch_input)
    tt_output = ttnn.to_torch(tt_output_ttnn)

    expected_shape = torch_output.shape
    if tt_output.shape != (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]):
        B, C, H, W = expected_shape
        tt_output = tt_output.reshape(B, H, W, C)

    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)
    tt_output = tt_output.float()

    if tt_output.shape != torch_output.shape:
        logger.error(f"Shape mismatch! PyTorch: {torch_output.shape}, TTNN: {tt_output.shape}")
        min_shape = [min(s1, s2) for s1, s2 in zip(torch_output.shape, tt_output.shape)]
        torch_output = torch_output[tuple(slice(0, s) for s in min_shape)]
        tt_output = tt_output[tuple(slice(0, s) for s in min_shape)]

    _, pcc_message = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"VGG Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)
