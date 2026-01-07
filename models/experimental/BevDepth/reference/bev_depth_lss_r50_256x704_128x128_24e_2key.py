# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

########################################################
# Adapted from https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py
# Copyright (c) Megvii Inc. All rights reserved.
########################################################

import os
import torch
from models.experimental.BevDepth.reference.base_exp import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)
from models.experimental.BevDepth.reference.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf["bev_backbone_conf"]["in_channels"] = 80 * (len(self.key_idxes) + 1)
        self.head_conf["bev_neck_conf"]["in_channels"] = [80 * (len(self.key_idxes) + 1), 160, 320, 640]
        self.head_conf["train_cfg"]["code_weights"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.model = BaseBEVDepth(self.backbone_conf, self.head_conf, is_train_depth=True)

    def load_checkpoint(self, checkpoint_path=None, map_location="cpu", verbose=True):
        """
        Load weights from a checkpoint file into the model.
        """
        model = self.model

        if checkpoint_path is None:
            file_dir = os.path.dirname(__file__)
            file_dir = os.path.dirname(file_dir)
            checkpoint_path = os.path.join(
                file_dir, "resources", "checkpoints", "bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
            )

        if not os.path.exists(checkpoint_path):
            downloaded_weights_path = "/tmp/bevdepth_weights.pth"
            if os.path.exists(downloaded_weights_path):
                checkpoint_path = downloaded_weights_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint file not found at: {checkpoint_path} (also checked {downloaded_weights_path})"
                )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                if any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict_to_load = state_dict

        # Load weights into model
        model.load_state_dict(state_dict_to_load, strict=False)
