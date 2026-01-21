# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import nn
import pytest

import ttnn
from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.swin2sr.reference.swin2sr import Upsample as TorchUpsample
from models.experimental.swin2sr.tt.tt_upsample import TtUpsample
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def create_upsample_parameters(torch_model, device):
    """Create parameters for TtUpsample from torch model."""
    parameters = []
    for i, module in enumerate(torch_model):
        if isinstance(module, nn.Conv2d):
            weight = module.weight
            bias = module.bias if module.bias is not None else torch.zeros(module.out_channels)

            parameters.append(
                {
                    "weight": ttnn.to_device(
                        ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), device=device
                    ),
                    "bias": ttnn.to_device(
                        ttnn.from_torch(
                            torch.reshape(bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        ),
                        device=device,
                    ),
                }
            )
    return parameters


@pytest.mark.parametrize(
    "scale,num_feat,input_height,input_width",
    [
        (2, 64, 32, 32),
        (4, 64, 32, 32),
        (8, 64, 16, 16),
        (3, 64, 32, 32),
    ],
)
def test_upsample_ttnn_vs_torch(device, scale, num_feat, input_height, input_width, reset_seeds):
    torch_model = TorchUpsample(scale=scale, num_feat=num_feat)
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, num_feat, input_height, input_width)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = create_upsample_parameters(torch_model, device)

    tt_model = TtUpsample(
        device=device,
        parameters=parameters,
        scale=scale,
        num_feat=num_feat,
        input_height=input_height,
        input_width=input_width,
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output_tensor = tt_model(tt_input_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    tt_output_tensor = tt_output_tensor.permute(0, 3, 1, 2)

    pcc_required = 0.99
    passed, pcc = comp_pcc(torch_output_tensor, tt_output_tensor, pcc_required)
    assert passed, f"PCC value {pcc} is lower than required {pcc_required}"


def load_upsample_weights_from_checkpoint(checkpoint_path, scale, num_feat):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint

    if (scale & (scale - 1)) == 0:  # scale = 2^n
        num_ops = int(math.log(scale, 2))
    elif scale == 3:
        num_ops = 1
    else:
        raise ValueError(f"Unsupported scale: {scale}")

    weights_list = []
    for i in range(num_ops):
        prefix = f"upsample.{i * 2}"

        weight_key = f"{prefix}.weight"
        if weight_key not in params:
            available_keys = [k for k in params.keys() if "upsample" in k]
            raise KeyError(f"Upsample layer {weight_key} not found in checkpoint. Available keys: {available_keys}")

        weights_list.append(
            {
                "weight": params[weight_key],
                "bias": params[f"{prefix}.bias"] if f"{prefix}.bias" in params else None,
            }
        )

    return weights_list


@pytest.mark.parametrize(
    "scale,num_feat,input_height,input_width",
    [
        (2, 64, 64, 64),
    ],
)
def test_upsample_checkpoint(device, scale, num_feat, input_height, input_width, reset_seeds):
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights_list = load_upsample_weights_from_checkpoint(checkpoint_path, scale, num_feat)

    torch_model = TorchUpsample(scale=scale, num_feat=num_feat)
    for i, (module, weights) in enumerate(zip([m for m in torch_model if isinstance(m, nn.Conv2d)], weights_list)):
        module.weight.data = weights["weight"]
        if weights["bias"] is not None:
            module.bias.data = weights["bias"]
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, num_feat, input_height, input_width)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = []
    for weights in weights_list:
        weight = weights["weight"]
        bias = weights["bias"] if weights["bias"] is not None else torch.zeros(weight.shape[0])

        parameters.append(
            {
                "weight": ttnn.to_device(
                    ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), device=device
                ),
                "bias": ttnn.to_device(
                    ttnn.from_torch(
                        torch.reshape(bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                    device=device,
                ),
            }
        )

    tt_model = TtUpsample(
        device=device,
        parameters=parameters,
        scale=scale,
        num_feat=num_feat,
        input_height=input_height,
        input_width=input_width,
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output_tensor = tt_model(tt_input_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    tt_output_tensor = tt_output_tensor.permute(0, 3, 1, 2)

    pcc_required = 0.99
    passed, pcc = comp_pcc(torch_output_tensor, tt_output_tensor, pcc_required)
    assert passed, f"PCC value {pcc} is lower than required {pcc_required}"
