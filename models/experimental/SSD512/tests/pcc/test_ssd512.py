# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from models.experimental.SSD512.common import (
    SSD512_L1_SMALL_SIZE,
    SSD512_NUM_CLASSES,
    load_torch_model,
    create_ssd512_input_tensors,
)
from models.experimental.SSD512.tt.tt_ssd import build_ssd512
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("pcc", ((0.97),))
@pytest.mark.parametrize("size", (512,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_ssd512(device, pcc, size, reset_seeds):
    torch_model = load_torch_model(phase="test", size=size, num_classes=SSD512_NUM_CLASSES)
    torch_input, ttnn_input = create_ssd512_input_tensors(batch=1, input_height=size, input_width=size)

    with torch.no_grad():
        torch_sources = []
        torch_loc_preds = []
        torch_conf_preds = []
        x = torch_input.clone()

        for k in range(23):
            x = torch_model.base[k](x)
        torch_sources.append(torch_model.L2Norm(x.clone()))

        for k in range(23, len(torch_model.base)):
            x = torch_model.base[k](x)
        torch_sources.append(x.clone())

        for k, v in enumerate(torch_model.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                torch_sources.append(x.clone())

        for source, loc_layer, conf_layer in zip(torch_sources, torch_model.loc, torch_model.conf):
            loc_pred = loc_layer(source).permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_layer(source).permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)
            torch_conf_preds.append(conf_pred)

        torch_loc = torch.cat([o.view(o.size(0), -1) for o in torch_loc_preds], 1)
        torch_conf = torch.cat([o.view(o.size(0), -1) for o in torch_conf_preds], 1)

    ttnn_model = build_ssd512(num_classes=SSD512_NUM_CLASSES, device=device)
    ttnn_model.load_weights_from_torch(torch_model)

    ttnn.synchronize_device(device)

    ttnn_loc, ttnn_conf, _ = ttnn_model.forward(torch_input, dtype=ttnn.bfloat16, debug=True)

    ttnn_loc = ttnn_loc.float().flatten()
    ttnn_conf = ttnn_conf.float().flatten()
    torch_loc = torch_loc.flatten()
    torch_conf = torch_conf.flatten()

    min_loc_len = min(len(torch_loc), len(ttnn_loc))
    min_conf_len = min(len(torch_conf), len(ttnn_conf))

    _, pcc_message_loc = comp_pcc(torch_loc[:min_loc_len], ttnn_loc[:min_loc_len], pcc)
    logger.info(f"Location PCC: {pcc_message_loc}")

    _, pcc_message_conf = comp_pcc(torch_conf[:min_conf_len], ttnn_conf[:min_conf_len], pcc)
    logger.info(f"Confidence PCC: {pcc_message_conf}")

    assert_with_pcc(torch_loc[:min_loc_len], ttnn_loc[:min_loc_len], pcc)
    assert_with_pcc(torch_conf[:min_conf_len], ttnn_conf[:min_conf_len], pcc)
