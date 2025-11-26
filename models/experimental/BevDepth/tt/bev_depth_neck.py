# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.experimental.BevDepth.tt.utils import TTConvTranspose2D


@dataclass
class NeckOptimizer:
    deblock: dict


neck_optimisations = NeckOptimizer(
    deblock={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "reshard_if_not_optimal": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
)


class TtDeblock:
    def __init__(self, in_channels, out_channels, kernel_size, stride, parameters, model_config, layer_optimisations):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize TTConvTranspose2D layer
        self.conv_transpose = TTConvTranspose2D(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            parameters=parameters,
            kernel_fidelity=model_config,
            # **layer_optimisations.deblock,
        )

    def __call__(self, x, device):
        # Input x should be in NHWC format (batch, height, width, channels)
        input_shape = x.shape

        # ConvTranspose2d + ReLU
        x, output_shape = self.conv_transpose(device, x, input_shape)
        x = ttnn.relu(x)

        return x, output_shape


class TtSECONDFPN:
    def __init__(self, parameters, model_config, layer_optimisations=neck_optimisations):
        super().__init__()
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations

        # Initialize 4 deblocks with parameters
        deblocks_params = parameters["neck"].get("deblocks", [])
        print(deblocks_params)
        self.deblocks = [
            TtDeblock(
                in_channels=160,
                out_channels=64,
                kernel_size=1,
                stride=1,
                parameters=deblocks_params[0],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=160,
                out_channels=64,
                kernel_size=2,
                stride=2,
                parameters=deblocks_params[1],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=320,
                out_channels=64,
                kernel_size=4,
                stride=4,
                parameters=deblocks_params[2],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=640,
                out_channels=64,
                kernel_size=8,
                stride=8,
                parameters=deblocks_params[3],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
        ]

    def __call__(self, x0, x1, x2, x3, device=None):
        if device is None:
            raise ValueError("Device must be provided in __call__")

        # Process each input through its corresponding deblock
        y0, _ = self.deblocks[0](x0, device)
        y1, _ = self.deblocks[1](x1, device)
        y2, _ = self.deblocks[2](x2, device)
        y3, _ = self.deblocks[3](x3, device)

        # Concatenate along channel dimension (dim=3 in NHWC format)
        # All outputs should be (B, 128, 128, 64)
        y = ttnn.concat([y0, y1, y2, y3], dim=3)

        return y


class TtBEVDepthHead:
    def __init__(self, parameters, model_config, layer_optimisations=neck_optimisations):
        super().__init__()
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations
        self.secondfpn = TtSECONDFPN(parameters, model_config, layer_optimisations)

    def __call__(self, x0, x1, x2, x3, device=None):
        if device is None:
            raise ValueError("Device must be provided in __call__")

        return self.secondfpn(x0, x1, x2, x3, device)
