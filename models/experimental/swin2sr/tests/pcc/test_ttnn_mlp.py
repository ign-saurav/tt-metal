# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.experimental.swin2sr.reference.mlp import MLP as TorchSwin2SRMLP
from models.experimental.swin2sr.tt.tt_mlp import TtSwin2SRMLP
from models.experimental.swin2sr.tt.utils import get_checkpoint_path


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, TorchSwin2SRMLP):
            parameters.setdefault("fc1", {})
            fc1_weight = preprocess_linear_weight(torch_model.fc1.weight, dtype=ttnn.bfloat16)
            parameters["fc1"]["weight"] = ttnn.to_device(fc1_weight, device)
            if torch_model.fc1.bias is not None:
                fc1_bias = preprocess_linear_bias(torch_model.fc1.bias, dtype=ttnn.bfloat16)
                parameters["fc1"]["bias"] = ttnn.to_device(fc1_bias, device)

            parameters.setdefault("fc2", {})
            fc2_weight = preprocess_linear_weight(torch_model.fc2.weight, dtype=ttnn.bfloat16)
            parameters["fc2"]["weight"] = ttnn.to_device(fc2_weight, device)
            if torch_model.fc2.bias is not None:
                fc2_bias = preprocess_linear_bias(torch_model.fc2.bias, dtype=ttnn.bfloat16)
                parameters["fc2"]["bias"] = ttnn.to_device(fc2_bias, device)
        return parameters

    return custom_preprocessor


def load_mlp_weights_from_checkpoint(checkpoint_path, layer_idx=0, block_idx=0):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    prefix = f"layers.{layer_idx}.residual_group.blocks.{block_idx}.mlp"

    weights = {
        "fc1_weight": params[f"{prefix}.fc1.weight"],
        "fc1_bias": params[f"{prefix}.fc1.bias"],
        "fc2_weight": params[f"{prefix}.fc2.weight"],
        "fc2_bias": params[f"{prefix}.fc2.bias"],
    }

    in_features = weights["fc1_weight"].shape[1]
    hidden_features = weights["fc1_weight"].shape[0]
    out_features = weights["fc2_weight"].shape[0]

    weights["in_features"] = in_features
    weights["hidden_features"] = hidden_features
    weights["out_features"] = out_features

    return weights


@pytest.mark.parametrize(
    "layer_idx,block_idx",
    [
        (0, 0),
        (0, 1),
        (1, 0),
    ],
)
def test_swin2sr_mlp_ttnn_vs_torch_with_checkpoint(device, layer_idx, block_idx, reset_seeds):
    checkpoint_path = get_checkpoint_path("Swin2SR_ClassicalSR_X2_64.pth")

    weights = load_mlp_weights_from_checkpoint(checkpoint_path, layer_idx=layer_idx, block_idx=block_idx)

    in_features = weights["in_features"]
    hidden_features = weights["hidden_features"]
    out_features = weights["out_features"]

    torch_model = TorchSwin2SRMLP(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        act_layer=nn.GELU,
        drop=0.0,
    )

    torch_model.fc1.weight.data.copy_(weights["fc1_weight"])
    torch_model.fc1.bias.data.copy_(weights["fc1_bias"])
    torch_model.fc2.weight.data.copy_(weights["fc2_weight"])
    torch_model.fc2.bias.data.copy_(weights["fc2_bias"])

    torch_model.eval()

    batch_size = 1
    seq_len = 32
    torch_input_tensor = torch.randn(batch_size, seq_len, in_features)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    class ParamNamespace:
        pass

    ns = ParamNamespace()
    ns.fc1 = ParamNamespace()
    ns.fc2 = ParamNamespace()

    ns.fc1.weight = parameters["fc1"]["weight"]
    ns.fc1.bias = parameters["fc1"].get("bias", None)
    ns.fc2.weight = parameters["fc2"]["weight"]
    ns.fc2.bias = parameters["fc2"].get("bias", None)

    ttnn_model = TtSwin2SRMLP(device=device, parameters=ns, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Log PCC value
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"\n[CHECKPOINT - MLP Layer {layer_idx}, Block {block_idx}] PCC: {pcc_message}")
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "in_features,hidden_features,out_features,seq_len",
    [
        (96, 384, 96, 64),
        (180, 360, 180, 32),
        (192, 768, 192, 32),
    ],
)
def test_swin2sr_mlp_ttnn_vs_torch(device, in_features, hidden_features, out_features, seq_len, reset_seeds):
    torch_model = TorchSwin2SRMLP(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        act_layer=nn.GELU,
        drop=0.0,
    )
    torch_model.eval()

    batch_size = 1
    torch_input_tensor = torch.randn(batch_size, seq_len, in_features)

    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=device,
    )

    class ParamNamespace:
        pass

    ns = ParamNamespace()
    ns.fc1 = ParamNamespace()
    ns.fc2 = ParamNamespace()

    ns.fc1.weight = parameters["fc1"]["weight"]
    ns.fc1.bias = parameters["fc1"].get("bias", None)
    ns.fc2.weight = parameters["fc2"]["weight"]
    ns.fc2.bias = parameters["fc2"].get("bias", None)

    ttnn_model = TtSwin2SRMLP(device=device, parameters=ns, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Log PCC value
    pcc_passed, pcc_message = comp_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(
        f"\n[SYNTHETIC - MLP in_features={in_features}, hidden_features={hidden_features}, out_features={out_features}] PCC: {pcc_message}"
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
