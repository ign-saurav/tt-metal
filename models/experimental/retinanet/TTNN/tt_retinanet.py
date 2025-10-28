# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.retinanet.TTNN.tt_backbone import TTBackbone
from models.experimental.retinanet.TTNN.tt_regression_head import _
from models.experimental.retinanet.TTNN.tt_fpn import _
from loguru import logger


class TTRetinanet:
    def __init__(self, parameters, model_config, device, name="backbone"):
        self.Backbone = TTBackbone(parameters=parameters, model_config=model_config)
        self.classification = _
        self.regression = _

    def __call__(self, x, device):
        out_backbone = self.Backbone(x, device)
        logger.debug("✅✅✅ BACKBONE Complete ✅✅✅")
        out_classification = self.classification(out_backbone, device)
        logger.debug("✅✅✅ CLASSIFICATION HEAD Complete ✅✅✅")
        out_regression = self.regression(out_backbone, device)
        logger.debug("✅✅✅ REGRESSION HEAD Complete ✅✅✅")
        out = {"classification": out_classification, "regression": out_regression}
        return out
