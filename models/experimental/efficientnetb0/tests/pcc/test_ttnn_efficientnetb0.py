# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import re
import cv2
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.efficientnetb0.tt.model_preprocessing import (
    create_efficientnetb0_input_tensors,
    create_efficientnetb0_model_parameters,
)
from models.experimental.efficientnetb0.tt import efficientnetb0 as ttnn_efficientnetb0
from models.experimental.efficientnetb0.common import load_torch_model, EFFICIENTNETB0_L1_SMALL_SIZE


def clean_and_filter_state_dict(state_dict, prefix="backbone_net.model."):
    """
    Cleans and filters a checkpoint state_dict to match model naming:
      - Removes 'backbone_net.model.' prefix if present
      - Keeps only:
          _conv_stem.*
          _bn0.*
          _blocks.0.* to _blocks.15.*
          _conv_head.*
          _bn1.*
          _fc.*
      - Converts '_blocks.<n>.' → '_blocks<n>.' (removes dot after 'blocks')
      - Removes nested '.conv.' / '.bn.' / '.linear.' layers to flatten naming
    """

    new_state_dict = {}

    for k, v in state_dict.items():
        # Remove prefix if present
        if k.startswith(prefix):
            k = k[len(prefix) :]

        # Keep conv_stem and bn0
        if k.startswith("_conv_stem.") or k.startswith("_bn0."):
            new_key = re.sub(r"\.(conv|bn|linear)\.", ".", k)  # flatten nested names
            new_state_dict[new_key] = v

        # Keep _blocks0 to _blocks15 (convert '_blocks.<n>.' -> '_blocks<n>.')
        elif re.match(r"_blocks\.(?:[0-9]|1[0-5])\.", k):
            new_key = re.sub(r"^_blocks\.(\d+)\.", r"_blocks\1.", k)
            new_key = re.sub(r"\.(conv|bn|linear)\.", ".", new_key)  # flatten nested
            new_state_dict[new_key] = v

    return new_state_dict


def create_efficientnetb0_input_tensors_from_img(device, torch_input_tensor, mesh_mapper=None):
    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_input_tensor = ttnn.from_torch(
        # torch_input_tensor, dtype=ttnn.bfloat16, device=device, memory_config=input_mem_config, mesh_mapper=mesh_mapper
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    return ttnn_input_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE}], indirect=True)
def test_efficientnetb0_model(device, reset_seeds, model_location_generator):
    torch_model = load_torch_model(model_location_generator)
    pt_file = "efficientdet-d0.pth"
    checkpoint = torch.load(pt_file)
    checkpoint = clean_and_filter_state_dict(checkpoint)
    torch_model.load_state_dict(checkpoint, strict=True)
    input_image = "eff_input.jpg"
    img = cv2.imread(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    # img_resized = cv2.resize(img, (512, 512))
    torch_input = torch.from_numpy(img_resized).float() / 255.0
    torch_input = torch_input.permute(2, 0, 1)
    torch_input = torch_input.unsqueeze(0)
    torch_model.eval()

    torch_input, ttnn_input = create_efficientnetb0_input_tensors(device)
    # ttnn_input = create_efficientnetb0_input_tensors_from_img(device, torch_input)
    torch_output = torch_model(torch_input)
    conv_params, parameters = create_efficientnetb0_model_parameters(torch_model, torch_input, device=device)

    ttnn_model = ttnn_efficientnetb0.Efficientnetb0(device, parameters, conv_params)

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    flag, msg = assert_with_pcc(torch_output, ttnn_output, 0.92)
    print(msg)
