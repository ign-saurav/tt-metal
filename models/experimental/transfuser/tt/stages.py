# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


shard_dict = {
    # stage_name : shard_layout
    "layer1": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "layer2": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "layer3": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "layer4": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}


class Ttstages:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        stage_name,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        self.inplanes = 32
        self.layer = self._make_layer(
            parameters=parameters,
            planes=72,
            blocks=len(parameters.keys()),
            stride=stride,
            groups=3,
            model_config=model_config,
            stage_name=stage_name,
        )

    # def _make_layer(
    #     self,
    #     parameters,
    #     planes: int,
    #     blocks: int,
    #     stride: int,
    #     groups: int = 1,
    #     model_config=None,
    #     stage_name=None,
    # ) -> List[TTRegNetBottleneck]:
    #     layers = []
    #     self.inplanes = 32

    #     shard_layout = shard_dict[stage_name]

    #     # First block (may have downsample)
    #     downsample = stride != 1 or self.inplanes != planes
    #     layers.append(
    #         TTRegNetBottleneck(
    #             parameters=parameters["b1"],
    #             model_config=model_config,
    #             stride=stride,
    #             downsample=downsample,
    #             groups=groups,
    #             shard_layout=shard_layout,
    #         )
    #     )
    #     self.inplanes = planes

    #     # Remaining blocks
    #     for block_num in range(1, blocks):
    #         block_name = f"b{block_num + 1}"
    #         layers.append(
    #             TTRegNetBottleneck(
    #                 parameters=parameters[block_name],
    #                 model_config=model_config,
    #                 stride=1,
    #                 downsample=False,
    #                 groups=groups,
    #                 shard_layout=shard_layout,
    #             )
    #         )

    #     return layers
    @staticmethod
    def _make_layer(
        # self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        """
        parameters:
        - Either a root dict that contains {layer1, layer2, ...} each with {b1,b2,...}
        - Or a stage dict that directly contains {b1,b2,...}
        stage_name:
        - Required if 'parameters' is the root dict (so we can pick the stage).
        - Ignored if 'parameters' already looks like a stage dict.
        """

        # ---- Resolve which stage dict to use ----
        def _resolve_stage_dict(params, stage_key):
            # If it already looks like a stage dict (has b1), just use it
            if isinstance(params, dict) and any(k.startswith("b") for k in params.keys()):
                return params
            # Otherwise expect a root dict with the stage_name present
            if not isinstance(params, dict) or stage_key not in params:
                available = list(params.keys()) if isinstance(params, dict) else []
                raise KeyError(
                    f"Expected a stage dict for '{stage_key}' or a root dict containing it. " f"Got keys: {available}"
                )
            return params[stage_key]

        stage_params = _resolve_stage_dict(parameters, stage_name)

        # ---- Choose shard layout per stage ----
        if stage_name in ("layer1", "layer2"):
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif stage_name in ("layer3", "layer4"):
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            # Default to HEIGHT_SHARDED
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

        # ---- Validate available blocks ----
        # Expected names: b1, b2, ..., b{blocks}
        available_block_names = sorted(
            [k for k in stage_params.keys() if k.startswith("b")],
            key=lambda s: int(s[1:]) if s[1:].isdigit() else 0,
        )

        # If fewer blocks than requested, raise a descriptive error
        if len(available_block_names) < blocks:
            raise KeyError(
                f"Requested {blocks} blocks for {stage_name}, but only found blocks: "
                f"{available_block_names}. "
                f"Did you pass parameters for the wrong stage (e.g., layer1 for layer2)?"
            )

        layers = []

        # ---- First block (may have downsample) ----
        downsample = stride != 1 or inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=stage_params["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                shard_layout=shard_layout,
            )
        )
        inplanes = planes

        # ---- Remaining blocks (stride=1, no downsample) ----
        # Build exactly the number requested, in order b2..b{blocks}
        for idx in range(2, blocks + 1):
            bname = f"b{idx}"
            if bname not in stage_params:
                # Extra guard (should have been caught above)
                raise KeyError(f"Missing block '{bname}' in {stage_name}. " f"Available: {available_block_names}")
            layers.append(
                TTRegNetBottleneck(
                    parameters=stage_params[bname],
                    model_config=model_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    shard_layout=shard_layout,
                )
            )

        return layers

    def __call__(self, x, device, input_shape=None):
        shape = input_shape if input_shape is not None else x.shape
        # Process image input
        for block in self.layer:
            x, shape = block(x, device, shape)

        return x, shape
