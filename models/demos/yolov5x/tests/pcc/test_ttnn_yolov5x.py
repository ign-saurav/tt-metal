# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn
from models.demos.yolov5x.common import YOLOV5X_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov5x.tt.model_preprocessing import create_yolov5x_model_parameters
from models.demos.yolov5x.tt.yolov5x import Yolov5x
from tests.ttnn.utils_for_testing import assert_with_pcc

IM_TENSOR_PATH = Path(__file__).parent / "im_tensor.npy"
INTERMEDIATES_DIR = Path(__file__).parent / "intermediates"


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE}], indirect=True)
def test_yolov5x(device, reset_seeds, model_location_generator, request):
    torch_input = torch.from_numpy(np.load(IM_TENSOR_PATH)).float()
    n, c, h, w = torch_input.shape
    padded_c = 16 if c < 16 else c  # If the channels < 16, pad the channels to 16 to run the Conv layer
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, padded_c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem_config
    )

    torch_model = load_torch_model(model_location_generator)
    torch_model = torch_model.model

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device)

    INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
    torch_model_output = torch_model(torch_input, save_dir=INTERMEDIATES_DIR)[0]
    ttnn_module = Yolov5x(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input, save_dir=INTERMEDIATES_DIR)
    ttnn_output = ttnn.to_torch(ttnn_output)

    print(f"torch_model_output: {torch_model_output.shape}")
    print(f"torch_model_output: {torch_model_output.flatten()[:20]}")
    print(f"ttnn_output: {ttnn_output.shape}")
    print(f"ttnn_output: {ttnn_output.flatten()[:20]}")
    pcc_passed, pcc_value = assert_with_pcc(torch_model_output, ttnn_output, 0.99)
    print(f"PCC value: {pcc_value}")
