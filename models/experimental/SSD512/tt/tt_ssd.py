# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import TtVggBackbone
from models.experimental.SSD512.tt.layers.tt_extras_backbone import TtExtrasBackbone
from models.experimental.SSD512.tt.layers.tt_multibox_heads import TtMultiboxHeads
from models.experimental.SSD512.tt.layers.tt_l2norm import TtL2Norm
from models.experimental.SSD512.tt.utils import (
    extract_vgg_parameters_from_torch,
    extract_extras_parameters_from_torch,
    extract_multibox_parameters_from_torch,
)
from models.common.utility_functions import tt_to_torch_tensor


class SSD512Network:
    def __init__(self, num_classes: int = 21, device=None):
        self.num_classes = num_classes
        self.device = device
        self.batch_size = 1

        self.vgg_backbone = None
        self.extras_backbone = None
        self.multibox_heads = None
        self.l2norm = TtL2Norm(512, 20, device=device)

        self.vgg_channels = [512, 1024]
        self.extra_channels = [512, 256, 256, 256, 256, 256]

    def load_weights_from_torch(self, torch_model):
        # Extract parameters as fresh torch tensors (already cloned in extract functions)
        vgg_parameters = extract_vgg_parameters_from_torch(torch_model.base)

        self.vgg_backbone = TtVggBackbone(
            size=512,
            input_channels=3,
            batch_size=self.batch_size,
            parameters=vgg_parameters,
            device=self.device,
        )

        extras_parameters = extract_extras_parameters_from_torch(torch_model.extras)

        self.extras_backbone = TtExtrasBackbone(
            size=512,
            input_channels=1024,
            batch_size=self.batch_size,
            parameters=extras_parameters,
            device=self.device,
        )

        loc_parameters = extract_multibox_parameters_from_torch(torch_model.loc)
        conf_parameters = extract_multibox_parameters_from_torch(torch_model.conf)

        self.multibox_heads = TtMultiboxHeads(
            size=512,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            loc_parameters=loc_parameters,
            conf_parameters=conf_parameters,
            vgg_channels=self.vgg_channels,
            extra_channels=self.extra_channels,
            device=self.device,
        )

    def forward(self, x, dtype=ttnn.bfloat16, memory_config=None, debug=False):
        batch_size = x.shape[0]

        # VGG backbone - returns NHWC format
        vgg_result = self.vgg_backbone(x, return_sources=[22])

        if isinstance(vgg_result, tuple):
            conv7, vgg_sources = vgg_result
            conv4_3 = vgg_sources[0]
        else:
            conv7 = vgg_result
            conv4_3 = None

        sources = []

        # Apply L2Norm to conv4_3 if available
        if conv4_3 is not None:
            # L2Norm expects and returns NHWC format
            conv4_3_norm = self.l2norm(conv4_3)
            sources.append(conv4_3_norm)

        # Add conv7
        sources.append(conv7)

        # Extras backbone - expects and returns NHWC format
        _, extra_sources = self.extras_backbone(conv7, return_sources=True)
        sources.extend(extra_sources)

        # Multibox heads - expects sources in NHWC format
        loc_preds, conf_preds = self.multibox_heads(sources)

        loc_outputs = []
        conf_outputs = []

        for idx, (loc_pred, conf_pred) in enumerate(zip(loc_preds, conf_preds)):
            # Get shapes and validate
            loc_shape = loc_pred.shape
            conf_shape = conf_pred.shape

            # Validate shapes
            if len(loc_shape) != 4 or len(conf_shape) != 4:
                raise ValueError(
                    f"Source {idx}: Expected 4D tensors, got loc shape {loc_shape}, conf shape {conf_shape}"
                )

            B_loc, H_loc, W_loc, C_loc = loc_shape
            B_conf, H_conf, W_conf, C_conf = conf_shape

            # Validate dimensions are positive
            if H_loc <= 0 or W_loc <= 0 or C_loc <= 0:
                raise ValueError(f"Source {idx}: Invalid loc dimensions H={H_loc}, W={W_loc}, C={C_loc}")
            if H_conf <= 0 or W_conf <= 0 or C_conf <= 0:
                raise ValueError(f"Source {idx}: Invalid conf dimensions H={H_conf}, W={W_conf}, C={C_conf}")

            # Check divisibility
            total_loc = H_loc * W_loc * C_loc
            total_conf = H_conf * W_conf * C_conf

            if total_loc % 4 != 0:
                raise ValueError(f"Source {idx}: Location total {total_loc} not divisible by 4")
            if total_conf % self.num_classes != 0:
                raise ValueError(f"Source {idx}: Confidence total {total_conf} not divisible by {self.num_classes}")

            # Safe reshape - ensure ROW_MAJOR layout first
            if loc_pred.layout != ttnn.ROW_MAJOR_LAYOUT:
                loc_pred = ttnn.to_layout(loc_pred, ttnn.ROW_MAJOR_LAYOUT)
            if conf_pred.layout != ttnn.ROW_MAJOR_LAYOUT:
                conf_pred = ttnn.to_layout(conf_pred, ttnn.ROW_MAJOR_LAYOUT)

            # Calculate number of boxes
            num_loc_boxes = total_loc // 4
            num_conf_boxes = total_conf // self.num_classes

            # Reshape to [B, num_boxes, 4] and [B, num_boxes, num_classes]
            try:
                loc_out_flat = ttnn.reshape(loc_pred, (B_loc, num_loc_boxes, 4))
                conf_out_flat = ttnn.reshape(conf_pred, (B_conf, num_conf_boxes, self.num_classes))
            except Exception as e:
                raise RuntimeError(
                    f"Source {idx}: Reshape failed - loc {loc_shape} -> ({B_loc}, {num_loc_boxes}, 4), "
                    f"conf {conf_shape} -> ({B_conf}, {num_conf_boxes}, {self.num_classes}). Error: {e}"
                )

            # Convert back to TILE layout
            if loc_out_flat.layout != ttnn.TILE_LAYOUT:
                loc_out_flat = ttnn.to_layout(loc_out_flat, ttnn.TILE_LAYOUT)
            if conf_out_flat.layout != ttnn.TILE_LAYOUT:
                conf_out_flat = ttnn.to_layout(conf_out_flat, ttnn.TILE_LAYOUT)

            loc_outputs.append(loc_out_flat)
            conf_outputs.append(conf_out_flat)

        # Concatenate all predictions
        if len(loc_outputs) > 1:
            loc = ttnn.concat(loc_outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            loc = loc_outputs[0]

        if len(conf_outputs) > 1:
            conf = ttnn.concat(conf_outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            conf = conf_outputs[0]

        if debug:
            debug_sources = [tt_to_torch_tensor(s) for s in sources]
            debug_dict = {
                "sources": debug_sources,
                "loc_preds": [tt_to_torch_tensor(l) for l in loc_outputs],
                "conf_preds": [tt_to_torch_tensor(c) for c in conf_outputs],
            }
            loc_torch = tt_to_torch_tensor(loc)
            conf_torch = tt_to_torch_tensor(conf)
            return loc_torch, conf_torch, debug_dict

        return loc, conf


def build_ssd512(num_classes=21, device=None):
    return SSD512Network(num_classes=num_classes, device=device)
