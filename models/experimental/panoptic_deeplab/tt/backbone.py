# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List, Optional
from models.experimental.panoptic_deeplab.tt.bottleneck import TTBottleneck, bottleneck_layer_optimisations
from models.experimental.panoptic_deeplab.tt.stem import resnet52Stem, neck_optimisations


class TTBackbone:
    def __init__(self, parameters, model_config, reshard_block_inputs: bool = True, small_tensor: bool = False):
        layers = [3, 4, 6, 3]
        self.inplanes = 128
        self.reshard_block_inputs = reshard_block_inputs
        # stem
        neck_layer_optimistaion = neck_optimisations["optimization_full_tensor"]
        if small_tensor:
            neck_layer_optimistaion = neck_optimisations["optimization_small_tensor"]
        self.stem = resnet52Stem(
            parameters.stem,
            stride=1,
            model_config=model_config,
            layer_optimisations=neck_layer_optimistaion,
        )
        # Four bottleneck stages (layer1, layer2, layer3, layer4)
        self.layer1 = self._make_layer(
            name="layer1",
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=bottleneck_layer_optimisations["layer1"],
        )
        self.layer2 = self._make_layer(
            name="layer2",
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=bottleneck_layer_optimisations["layer2"],
        )
        self.layer3 = self._make_layer(
            name="layer3",
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=bottleneck_layer_optimisations["layer3"],
        )
        self.layer4 = self._make_layer(
            name="layer4",
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=1,
            dilate_config=[2, 4, 8],
            model_config=model_config,
            layer_optimisations=bottleneck_layer_optimisations["layer4"],
        )

    def _make_layer(
        self,
        name: str,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        dilate_config: Optional[List[int]] = None,
        model_config=None,
        layer_optimisations=bottleneck_layer_optimisations["default"],
    ) -> List[TTBottleneck]:
        if dilate_config is None:
            dilate_config = [1] * blocks
        layers = []
        layers.append(
            TTBottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * TTBottleneck.expansion,
                stride=stride,
                model_config=model_config,
                dilation=dilate_config[0],
                name=f"{name}.0",
                layer_optimisations=layer_optimisations,
            )
        )
        self.inplanes = planes * TTBottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                TTBottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                    dilation=dilate_config[block_num],
                    name=f"{name}.{block_num}",
                    layer_optimisations=layer_optimisations,
                )
            )
        return layers

    def __call__(self, x, device):
        x = self.stem(x, device)
        shape = x.shape

        for block in self.layer1:
            # x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.reallocate(x)
            x, shape = block(x, device, shape)

        res_2 = x
        res_2 = ttnn.to_memory_config(res_2, ttnn.DRAM_MEMORY_CONFIG)

        for block in self.layer2:
            # if self.reshard_block_inputs:
            #     x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.reallocate(x)
            x, shape = block(x, device, shape)

        res_3 = x
        res_3 = ttnn.to_memory_config(res_3, ttnn.DRAM_MEMORY_CONFIG)

        for block in self.layer3:
            # x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.reallocate(x)
            x, shape = block(x, device, shape)

        # res_4 = x
        # res_4 = ttnn.to_memory_config(res_4, ttnn.DRAM_MEMORY_CONFIG)

        for block in self.layer4:
            # x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.reallocate(x)
            x, shape = block(x, device, shape)

        res_5 = x
        # res_5 = ttnn.to_memory_config(res_5, ttnn.DRAM_MEMORY_CONFIG)

        return {"res_2": res_2, "res_3": res_3, "res_5": res_5}
