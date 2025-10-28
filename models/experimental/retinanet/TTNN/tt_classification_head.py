# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.retinanet.TTNN.utils import TTConv2D, Conv2dNormActivation


class TTClassification:
    def __init__(self, parameters, model_config, device, layer_optimisations=None):
        # Grid size for GroupNorm
        grid_size = ttnn.CoreGrid(y=8, x=8)

        # Create input mask for GroupNorm
        input_mask_tensor = ttnn.create_group_norm_input_mask(256, 32, grid_size.y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.conv_blocks = []
        for i in range(4):
            conv_block = Conv2dNormActivation(
                parameters=parameters["conv"][i],
                device=device,
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                num_groups=32,
                grid_size=grid_size,
                input_mask=input_mask_tensor,
                model_config=model_config,
            )
            self.conv_blocks.append(conv_block)

        self.cls_logits = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            parameters=parameters["cls_logits"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_reshape=True,
            # **layer_optimisations.cls_logits,
        )

    def __call__(
        self,
        x,
        device,
        in_shape,
    ):
        all_cls_logits = []
        for feature in x:
            print(f"input shape = {feature.shape}")
            for block in self.conv_blocks:
                feature, shape = block(feature)
                print(f"feature shape = {feature.shape}")

            cls_logits, _ = self.cls_logits(device, feature, shape)
            N, H_out, W_out, _ = cls_logits.shape
            cls_logits = ttnn.reshape(cls_logits, (N, H_out, W_out, 9, 91))
            cls_logits = ttnn.reshape(cls_logits, (N, H_out * W_out * 9, 91))
            all_cls_logits.append(cls_logits)

        output = ttnn.concat(all_cls_logits, dim=1)

        return output
