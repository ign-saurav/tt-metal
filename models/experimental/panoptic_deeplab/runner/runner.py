# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab


class PanopticDeeplabRunner:
    def __init__(self, parameters, model_config):
        self.model = TTPanopticDeepLab(parameters, model_config)

    def run(self, input):
        ttnn_output_tensor = self.model(input)
        return ttnn_output_tensor
