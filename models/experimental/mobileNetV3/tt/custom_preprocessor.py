# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidual, SElayer, Conv2dNormActivation


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, MobileNetV3):
            parameters["features"] = {}
            parameters_features = {}
            for index_1, child in enumerate(model.features.children()):
                parameters_features[index_1] = {}
                if isinstance(child, Conv2dNormActivation):
                    parameters_features[index_1][0] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child[0], child[1])
                    parameters_features[index_1][0]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters_features[index_1][0]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )
                elif isinstance(child, InvertedResidual):
                    parameters_features[index_1]["block"] = {}
                    for index, child_1 in enumerate(child.block.children()):
                        parameters_features[index_1]["block"][index] = {}
                        if isinstance(child_1, SElayer):
                            parameters_features[index_1]["block"][index]["fc1"] = {}
                            parameters_features[index_1]["block"][index]["fc1"]["weight"] = ttnn.from_torch(
                                child_1.fc1.weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc1"]["bias"] = ttnn.from_torch(
                                torch.reshape(child_1.fc1.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc2"] = {}
                            parameters_features[index_1]["block"][index]["fc2"]["weight"] = ttnn.from_torch(
                                child_1.fc2.weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc2"]["bias"] = ttnn.from_torch(
                                torch.reshape(child_1.fc2.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
                        elif isinstance(child_1, Conv2dNormActivation):
                            parameters_features[index_1]["block"][index][0] = {}
                            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_1[0], child_1[1])
                            parameters_features[index_1]["block"][index][0]["weight"] = ttnn.from_torch(
                                conv_weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index][0]["bias"] = ttnn.from_torch(
                                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
            parameters["features"] = parameters_features
            parameters["classifier"] = {}
            parameters["classifier"][0] = {}
            parameters["classifier"][0]["weight"] = preprocess_linear_weight(
                model.classifier[0].weight, dtype=ttnn.bfloat16
            )
            parameters["classifier"][0]["bias"] = preprocess_linear_bias(model.classifier[0].bias, dtype=ttnn.bfloat16)

            parameters["classifier"][3] = {}
            parameters["classifier"][3]["weight"] = preprocess_linear_weight(
                model.classifier[3].weight, dtype=ttnn.bfloat16
            )
            parameters["classifier"][3]["bias"] = preprocess_linear_bias(model.classifier[3].bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor
