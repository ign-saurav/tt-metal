# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import pickle

from tests.ttnn.utils_for_testing import assert_with_pcc
import numpy as np
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("input_features", [576])  # MobileNetV3 classifier input
@pytest.mark.parametrize("output_features", [1024])  # MobileNetV3 classifier output
@pytest.mark.parametrize("use_bias", [True])
@pytest.mark.parametrize("bias_shape", ["2d"])  # Test both (1, -1) and (1, 1, 1, -1)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_mobilenetv3_linear_layer(
    batch_size,
    input_features,
    output_features,
    use_bias,
    bias_shape,
    device,
):
    """Test linear layer with different bias shapes to debug MobileNetV3 PCC issue."""
    mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    torch_average_pool = mobilenet.avgpool
    torch_linear = mobilenet.classifier[0]

    with open("linear_1_input_tensor_torch_pcc.pkl", "rb") as f:
        torch_input = pickle.load(f)

    with open("linear_1_input_tensor_tt_pcc.pkl", "rb") as f:
        ttnn_input = pickle.load(f)
        ttnn_input = ttnn.from_torch(
            ttnn_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    # PyTorch forward pass
    torch_output = torch_average_pool(torch_input)
    print("::::::::::::::::::")
    print(torch_output.shape)
    torch_output = torch.flatten(torch_output, 1)
    torch_output = torch_linear(torch_output)

    # Preprocess weights using standard TTNN preprocessing
    from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias

    ttnn_weight = preprocess_linear_weight(torch_linear.weight, dtype=ttnn.bfloat16)

    if use_bias:
        ttnn_bias = preprocess_linear_bias(torch_linear.bias, dtype=ttnn.bfloat16)
        print("using bias")
    else:
        ttnn_bias = None

    # Move weights to device
    ttnn_weight = ttnn.to_device(ttnn_weight, device)
    if ttnn_bias is not None:
        ttnn_bias = ttnn.to_device(ttnn_bias, device)
        print("using bias")

    np.savetxt("weight_tt_unit.txt", ttnn.to_torch(ttnn_weight).flatten().to(torch.float32).detach().numpy())
    np.savetxt("bias_tt_unit.txt", ttnn.to_torch(ttnn_bias).flatten().to(torch.float32).detach().numpy())

    # TTNN forward pass
    # ttnn_output = ttnn.global_avg_pool2d(ttnn_input)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], -1))
    ttnn_output = ttnn.adaptive_avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=ttnn_input.shape[0],
        input_h=ttnn_input.shape[1],
        input_w=ttnn_input.shape[2],
        channels=ttnn_input.shape[-1],
        output_size=[1, 1],
    )
    print("::::::::::::::::::")
    print(ttnn_output.shape)
    # ttnn_output = ttnn.reshape(ttnn_output, (ttnn_output.shape[0], -1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # ttnn_output = ttnn.to_layout(ttnn_output, layout=ttnn.TILE_LAYOUT)
    padded_width = ((576 + 31) // 32) * 32  # = 608
    ttnn_output = ttnn.reshape(ttnn_output, (ttnn_output.shape[0], -1))
    ttnn_output = torch.nn.functional.pad(ttnn.to_torch(ttnn_output), (0, padded_width - 576))
    ttnn_output = ttnn.from_torch(
        ttnn_output, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print("::::::::::::::::::")
    print(ttnn_output.shape)

    ttnn_output = ttnn.linear(
        ttnn_output,
        ttnn_weight,
        bias=ttnn_bias,
    )

    with open("linear_1_out_tensor_tt_unit.pkl", "wb") as f:
        pickle.dump(ttnn.to_torch(ttnn_output), f)
    with open("linear_1_out_tensor_torch_unit.pkl", "wb") as f:
        pickle.dump(torch_output, f)

    # Convert back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Check shape consistency
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"TTNN output shape: {ttnn_output_torch.shape}")

    # Reshape TTNN output to match PyTorch if needed
    if ttnn_output_torch.shape != torch_output.shape:
        # Squeeze extra dimensions
        ttnn_output_torch = ttnn_output_torch.squeeze()
        if ttnn_output_torch.dim() == 1:
            ttnn_output_torch = ttnn_output_torch.unsqueeze(0)

    assert (
        ttnn_output_torch.shape == torch_output.shape
    ), f"Shape mismatch: expected {torch_output.shape}, got {ttnn_output_torch.shape}"

    # Check PCC
    assert_with_pcc(torch_output, ttnn_output_torch, 0.999)
