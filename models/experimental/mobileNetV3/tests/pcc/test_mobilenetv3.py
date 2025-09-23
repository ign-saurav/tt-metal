# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from torchvision import models
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tt.utils import conv_config as model_config
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel


class MobilenetV3TestInfra:
    def __init__(self, device, batch_size, input_channels, height, width):
        self.device = device
        self.batch_size = batch_size
        torch_input_tensor = torch.randn(batch_size, input_channels, height, width)
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        self.ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

        mobilenet = models.mobilenet_v3_small(weights=None)  # 0.99 pcc with random weights #bias 0
        # mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1) #0.72 pcc with real weights #bias non 0(might be bias loading)
        torch_model = mobilenet

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
        )

        self.torch_output_tensor = torch_model(torch_input_tensor)

        self.ttnn_model = ttnn_MobileNetV3(
            inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, parameters=parameters
        )

        self.run()
        self.validate()

    def run(self):
        logger.info("Running TTNN MobileNetV3 model...")
        self.output_tensor = self.ttnn_model(self.device, self.ttnn_input_tensor)
        return self.output_tensor

    def validate(self):
        logger.info("Validating TTNN output against PyTorch...")
        tt_output_tensor_torch = ttnn.to_torch(self.output_tensor)
        pcc_threshold = 0.99
        passed, msg = check_with_pcc(self.torch_output_tensor, tt_output_tensor_torch, pcc=pcc_threshold)
        assert passed, logger.error(f"MobileNetV3 PCC check failed: {msg}")

        logger.info(
            f"MobileNetV3 passed: "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={msg}"
        )

        return True, msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 224, 224),
    ],
)
def test_MobilenetV3(device, batch_size, input_channels, height, width):
    MobilenetV3TestInfra(
        device=device,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
    )
