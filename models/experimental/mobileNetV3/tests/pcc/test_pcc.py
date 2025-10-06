# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting
from models.experimental.mobileNetV3.tt.ttnn_invertedResidual import (
    ttnn_InvertedResidual,
    Conv2dNormActivation,
)

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

img = Image.open("models/experimental/mobileNetV3/resources/dog.jpeg").convert("RGB")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = preprocess(img)
torch_input_tensor = input_tensor.unsqueeze(0)  # shape [1,3,224,224]


@pytest.mark.parametrize(
    "batch_size,channels,height,width,feature,include_first_conv,include_last_conv,include_classifier",
    [
        (1, 3, 224, 224, [0, 10], True, True, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_invertedResidual(
    device,
    reset_seeds,
    batch_size,
    channels,
    height,
    width,
    feature,
    include_first_conv,
    include_last_conv,
    include_classifier,
):
    with torch.no_grad():
        # torch_input_tensor = torch.randn(batch_size, channels, height, width)
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # mobilenet.classifier[0].bias.data.zero_()
        # mobilenet.classifier[3].bias.data.zero_()

        mobilenet.eval()
        x_torch_og = mobilenet(torch_input_tensor)

        if include_first_conv and include_last_conv and include_classifier:
            torch_model = list(mobilenet.features) + [mobilenet.avgpool] + list(mobilenet.classifier)
        else:
            torch_model = list(mobilenet.features)[feature[0] + 1 : feature[1] + 2]

        parameters = preprocess_model_parameters(
            initialize_model=lambda: mobilenet, custom_preprocessor=create_custom_preprocessor(None), device=None
        )

        ttnn_blocks = []
        ttnn_kernels = []
        index = 1

        if include_first_conv:
            ttnn_blocks.append(
                Conv2dNormActivation(
                    kernel_size=3,
                    stride=2,
                    activation_layer=ttnn.hardswish,
                    parameters=parameters["features"][0],
                )
            )

        for i, cnf in enumerate(inverted_residual_setting):
            ttnn_kernels.append(ttnn_InvertedResidual(cnf, parameters=parameters["features"][index].block))
            index += 1

        ttnn_blocks.extend(ttnn_kernels[feature[0] : feature[1] + 1])

        if include_last_conv:
            ttnn_blocks.append(
                Conv2dNormActivation(
                    kernel_size=1,
                    activation_layer=ttnn.hardswish,
                    parameters=parameters["features"][index],
                )
            )

        # --- Forward pass sequentially through all blocks ---
        x_torch = torch_input_tensor
        x_tt = ttnn_input_tensor

        for idx, (tt_block, torch_layer) in enumerate(zip(ttnn_blocks, torch_model)):
            # PyTorch forward
            x_torch = torch_layer(x_torch)

            # TTNN forward
            x_tt = tt_block(device, x_tt)

            # Convert TTNN to PyTorch format for comparison
            tt_as_torch = ttnn.to_torch(x_tt)
            tt_as_torch = torch.permute(tt_as_torch, (0, 3, 1, 2))

            # PCC check
            try:
                passed, msg = check_with_pcc(tt_as_torch, x_torch, pcc=0.96)
                if passed:
                    print(f"[Block {idx}] PCC[{msg}] passed")
                else:
                    print(f"[Block {idx}] PCC[{msg}] Failed")

            except:
                print(f"[Block {idx}] PCC[{msg}] Failed")

        if include_classifier:
            parameters["classifier"][0].weight = ttnn.to_device(parameters["classifier"][0].weight, device=device)
            parameters["classifier"][3].weight = ttnn.to_device(parameters["classifier"][3].weight, device=device)
            parameters["classifier"][0].bias = ttnn.to_device(parameters["classifier"][0].bias, device=device)
            parameters["classifier"][3].bias = ttnn.to_device(parameters["classifier"][3].bias, device=device)

            # x_torch = torch.nn.functional.adaptive_avg_pool2d(x_torch, (1, 1))
            x_torch = torch_model[-5](x_torch)
            x_torch = torch.flatten(x_torch, 1)

            x_tt = ttnn.global_avg_pool2d(x_tt)
            x_tt = ttnn.reshape(x_tt, (x_tt.shape[0], -1))
            x_tt = ttnn.to_layout(x_tt, layout=ttnn.TILE_LAYOUT)
            idx += 1

            tt_as_torch = ttnn.to_torch(x_tt)
            try:
                passed, msg = check_with_pcc(tt_as_torch, x_torch, pcc=0.96)
                if passed:
                    print(f"[Block {idx} avg pool] PCC[{msg}] passed")
                else:
                    print(f"[Block {idx} avg pool] PCC[{msg}] Failed")

            except:
                print(f"[Block {idx} avg pool] PCC[{msg}] Failed")

            x_torch = torch_model[-4](x_torch)

            x_tt = ttnn.linear(
                x_tt,
                parameters["classifier"][0].weight,
                bias=parameters["classifier"][0].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,  # Use HiFi4 instead of LoFi
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                ),
            )

            idx += 1

            tt_as_torch = ttnn.to_torch(x_tt)
            try:
                passed, msg = check_with_pcc(tt_as_torch, x_torch, pcc=0.96)
                if passed:
                    print(f"[Block {idx} linear 1] PCC[{msg}] passed")
                else:
                    print(f"[Block {idx} linear 1] PCC[{msg}] Failed")

            except:
                print(f"[Block {idx} linear 1] PCC[{msg}] Failed")

            # x_torch = torch_model[-3](x_torch)
            # x_tt = ttnn.hardswish(x_tt)

            x_torch = torch.relu(x_torch)
            x_tt = ttnn.relu(x_tt)
            idx += 1

            tt_as_torch = ttnn.to_torch(x_tt)
            try:
                passed, msg = check_with_pcc(tt_as_torch, x_torch, pcc=0.96)
                if passed:
                    print(f"[Block {idx} hardswish] PCC[{msg}] passed")
                else:
                    print(f"[Block {idx} hardswish] PCC[{msg}] Failed")

            except:
                print(f"[Block {idx} hardswish] PCC[{msg}] Failed")

            x_torch = torch_model[-1](x_torch)

            x_tt = ttnn.linear(
                x_tt,
                parameters["classifier"][3].weight,
                bias=parameters["classifier"][3].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

            idx += 1

            tt_as_torch = ttnn.to_torch(x_tt)
            try:
                passed, msg = check_with_pcc(tt_as_torch, x_torch, pcc=0.96)
                if passed:
                    print(f"[Block {idx} final linear ] PCC[{msg}] passed")
                else:
                    print(f"[Block {idx} final linear ] PCC[{msg}] Failed")

            except:
                print(f"[Block {idx} final linear ] PCC[{msg}] Failed")

        try:
            passed, msg = check_with_pcc(x_torch_og, x_torch, pcc=0.96)

            # Postprocess
            probs = torch.nn.functional.softmax(tt_as_torch, dim=1)[0]
            top1_id = torch.argmax(probs).item()
            label = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"][top1_id]
            confidence = probs[top1_id].item()

            # Draw label on image
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)  # Windows
            except:
                font = ImageFont.load_default()  # fallback

            text = f"{label}: {confidence:.2%}"
            draw.text((10, 10), text, fill="red", font=font)

            # Save / show
            img.save("models/experimental/mobileNetV3/resources/image_with_label.jpg")

            if passed:
                print(f"[Block full torch PCC[{msg}] passed")
            else:
                print(f"[Block full torch ] PCC[{msg}] Failed")

        except:
            print(f"[Block full torch ] PCC[{msg}] Failed")
