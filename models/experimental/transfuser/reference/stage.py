# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import timm


class RegNet(nn.Module):
    """
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        assert architecture.startswith("regnet"), f"Only RegNet architecture supported, got: {architecture}"

        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=False)

        self.features.fc = None

        self.features.conv1 = self.features.stem.conv
        self.features.bn1 = self.features.stem.bn
        self.features.act1 = nn.Sequential()  # The Relu is part of the batch norm here.
        self.features.maxpool = nn.Sequential()
        self.features.layer1 = self.features.s1
        self.features.layer2 = self.features.s2
        self.features.layer3 = self.features.s3
        self.features.layer4 = self.features.s4
        self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.features.head = nn.Sequential()


class Stage(nn.Module):
    def __init__(self, config, stage_name="layer1", image_architecture="regnety_032"):
        super().__init__()
        self.config = config
        self.stage_name = stage_name
        self.image_encoder = RegNet(
            architecture=image_architecture, normalize=True, out_features=self.config.perception_output_features
        )

    def forward(self, image):
        # Dynamically access the stage layer based on stage_name
        stage_layer = getattr(self.image_encoder.features, self.stage_name)
        # x = stage_layer(image)
        x = stage_layer.b1(image)  # downsample
        # x = stage_layer.b1.conv1(image)   #downsample
        # x = stage_layer.b1.conv2(x)   #downsample
        # x = stage_layer.b1.se(x)   #downsample
        # x = x.mean((2, 3), keepdim=True)
        # print("req shape", x.shape)
        # x = stage_layer.b1.se.fc1(x)   #downsample
        # x = x.relu()
        # # return x
        # x = stage_layer.b1.se.fc2(x)   #downsample
        # x = x.sigmoid()
        # x = stage_layer.b2(x)       #stride=1, no downsample
        # x = stage_layer.b3(x)
        # x = stage_layer.b4(x)
        # x = stage_layer.b5(x)
        return x

    def fallback(self, image):
        # Dynamically access the stage layer based on stage_name
        stage_layer = getattr(self.image_encoder.features, self.stage_name)
        x = stage_layer.b1.se.fc1(image)  # downsample
        x = x.relu()
        x = stage_layer.b1.se.fc2(x)
        x = x.sigmoid()

        return x
