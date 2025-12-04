# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for Head module
"""

import pytest
import torch
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, tt2torch_tensor
from models.experimental.pointpillars.tt.head import TtHead
from models.experimental.pointpillars.reference.model.pointpillars import Head
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channel,n_anchors,n_classes",
    [
        (384, 6, 3),
    ],
)
def test_head(device, in_channel, n_anchors, n_classes, reset_seeds):
    torch.manual_seed(0)

    torch_model = Head(in_channel, n_anchors, n_classes)

    try:
        checkpoint = torch.load(
            "epoch_160.pth",
            map_location="cpu",
        )

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        head_state_dict = {}
        prefix = "head."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                head_state_dict[new_key] = value

        torch_model.load_state_dict(head_state_dict)
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, using random weights")

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # Create input tensor with shape [1, 384, 248, 216] (NCHW format for PyTorch)
    batch_size = 1
    height = 248
    width = 216
    torch_input = torch.randn(batch_size, in_channel, height, width, dtype=torch.bfloat16)

    torch_cls, torch_reg, torch_dir = torch_model(torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_model = TtHead(
        in_channel=in_channel,
        n_anchors=n_anchors,
        n_classes=n_classes,
        parameters=parameters["head"],
        device=device,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
    )

    tt_cls, tt_reg, tt_dir = tt_model.forward(ttnn_input)
    tt_cls_torch = tt2torch_tensor(tt_cls)
    tt_cls_torch = tt_cls_torch.reshape(torch_cls.shape[0], torch_cls.shape[2], torch_cls.shape[3], torch_cls.shape[1])
    tt_cls_torch = tt_cls_torch.permute(0, 3, 1, 2)
    passing_cls, pcc_cls = comp_pcc(torch_cls, tt_cls_torch, 0.99)
    logger.info(f"Classification PCC: {pcc_cls}")
    assert passing_cls, f"Classification PCC check failed: {pcc_cls}"

    tt_reg_torch = tt2torch_tensor(tt_reg)
    tt_reg_torch = tt_reg_torch.reshape(torch_reg.shape[0], torch_reg.shape[2], torch_reg.shape[3], torch_reg.shape[1])
    tt_reg_torch = tt_reg_torch.permute(0, 3, 1, 2)
    passing_reg, pcc_reg = comp_pcc(torch_reg, tt_reg_torch, 0.99)
    logger.info(f"Regression PCC: {pcc_reg}")
    assert passing_reg, f"Regression PCC check failed: {pcc_reg}"

    tt_dir_torch = tt2torch_tensor(tt_dir)
    tt_dir_torch = tt_dir_torch.reshape(torch_dir.shape[0], torch_dir.shape[2], torch_dir.shape[3], torch_dir.shape[1])
    tt_dir_torch = tt_dir_torch.permute(0, 3, 1, 2)
    passing_dir, pcc_dir = comp_pcc(torch_dir, tt_dir_torch, 0.99)
    logger.info(f"Direction PCC: {pcc_dir}")
    assert passing_dir, f"Direction PCC check failed: {pcc_dir}"

    assert (
        torch_cls.shape == tt_cls_torch.shape
    ), f"Classification shape mismatch: {torch_cls.shape} vs {tt_cls_torch.shape}"
    assert (
        torch_reg.shape == tt_reg_torch.shape
    ), f"Regression shape mismatch: {torch_reg.shape} vs {tt_reg_torch.shape}"
    assert torch_dir.shape == tt_dir_torch.shape, f"Direction shape mismatch: {torch_dir.shape} vs {tt_dir_torch.shape}"

    logger.info("Head test passed!")
