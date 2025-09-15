# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.panoptic_deeplab.tt.aspp import TTASPP
from models.experimental.panoptic_deeplab.tt.head import TTHead
from models.experimental.panoptic_deeplab.tt.res_block import TTRes
from models.experimental.panoptic_deeplab.tt.res_block import res_layer_optimisations
from models.experimental.panoptic_deeplab.tt.head import head_layer_optimisations
from dataclasses import dataclass
import ttnn


@dataclass
class DecoderOptimizer:
    res_layer_optimisations: dict
    head_layer_optimisations: dict
    shape: tuple


decoder_layer_optimisations = {
    "default": DecoderOptimizer(
        res_layer_optimisations=res_layer_optimisations["default"],
        head_layer_optimisations=head_layer_optimisations["default"],
        shape=(0, 0, 0, 0),
    ),
    "semantic_decoder": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["semantic_decoder.res3"],
            "res2": res_layer_optimisations["semantic_decoder.res2"],
        },
        head_layer_optimisations={
            "head_1": head_layer_optimisations["semantic_decoder.head_1"],
        },
        shape=(1, 128, 256, 256),
    ),
    "instance_decoder": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["instance_decoder.res3"],
            "res2": res_layer_optimisations["instance_decoder.res2"],
        },
        head_layer_optimisations={
            "head_1": head_layer_optimisations["instance_decoder.head_1"],
            "head_2": head_layer_optimisations["instance_decoder.head_2"],
        },
        shape=(1, 128, 256, 128),
    ),
}


class TTDecoder:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=decoder_layer_optimisations["default"],
        name="semantic_decoder",
    ) -> None:
        super().__init__()
        self.shape = layer_optimisations.shape
        self.name = name

        self.aspp = TTASPP(parameters.aspp, model_config, layer_optimisations=None)
        self.res3 = TTRes(
            parameters.res3,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res3"],
        )
        self.res2 = TTRes(
            parameters.res2,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res2"],
        )
        self.head = TTHead(
            parameters.head_1,
            model_config,
            layer_optimisations=layer_optimisations.head_layer_optimisations["head_1"],
        )
        if self.shape[-1] == 128:
            self.head_2 = TTHead(
                parameters.head_2,
                model_config,
                layer_optimisations=layer_optimisations.head_layer_optimisations["head_2"],
            )
        if self.name == "semantic_decoder":
            self.res3_upsample_channels = 256
            self.res2_upsample_channels = 256
        else:
            self.res3_upsample_channels = 256
            self.res2_upsample_channels = 128

    def __call__(self, x, res3, res2, upsample_channels, device):
        out = self.aspp(x, device)
        out = self.res3(out, res3, self.res3_upsample_channels, device)
        out = self.res2(out, res2, self.res2_upsample_channels, device)

        if self.name == "instance_decoder":
            activation_copy = ttnn.clone(out)
        out = self.head(out, device)

        if self.name == "instance_decoder":
            out_ = self.head_2(activation_copy, device)
        else:
            out_ = None

        return out, out_
