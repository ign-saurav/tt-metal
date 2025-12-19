# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.SSD512.reference.ssd import build_ssd
from models.experimental.SSD512.reference.layers.functions.prior_box import PriorBox

from models.experimental.SSD512.reference.data.config import voc


SSD512_L1_SMALL_SIZE = 98304
SSD512_NUM_CLASSES = 21


def load_torch_model(phase="test", size=512, num_classes=21):
    torch_model = build_ssd(phase, size=size, num_classes=num_classes)
    torch_model.eval()
    return torch_model


def generate_prior_boxes(cfg=None):
    if cfg is None:
        cfg = voc["SSD512"]
    prior_box = PriorBox(cfg)
    priors = prior_box.forward()
    if not isinstance(priors, torch.Tensor):
        priors = torch.tensor(priors)
    return priors


def create_ssd512_input_tensors(batch=1, input_height=512, input_width=512, mesh_mapper=None):
    torch_input = torch.randn(batch, 3, input_height, input_width)
    ttnn_input = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper)
    return torch_input, ttnn_input
