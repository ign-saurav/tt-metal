# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.SSR.reference.SSR.model.tile_refinement import Upsample
from models.experimental.SSR.tt.tile_refinement import TTUpsample

from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_upsample_preprocessor(device):
    def custom_preprocessor(model, name):
        """Custom preprocessor for converting PyTorch weights to TTNN format"""
        parameters = {}
        if isinstance(model, Upsample):
            conv_idx = 0
            for i, layer in enumerate(model):
                if isinstance(layer, nn.Conv2d):
                    parameters[f"conv_{conv_idx}"] = {}
                    parameters[f"conv_{conv_idx}"]["weight"] = ttnn.from_torch(
                        layer.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    if layer.bias is not None:
                        parameters[f"conv_{conv_idx}"]["bias"] = ttnn.from_torch(
                            torch.reshape(layer.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    conv_idx += 1

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "scale,num_feat,batch_size,input_size",
    [(4, 64, 1, 256)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_upsample(device, scale, num_feat, batch_size, input_size):
    """Test Upsample block against PyTorch reference"""
    torch.manual_seed(0)
    if batch_size == 8:
        pytest.xfail(
            "Statically allocated circular buffers in program 136 clash with L1 buffers on core range [(x=0,y=0) - (x=7,y=7)]. L1 buffer allocated at 118272 and static circular buffer region ends at 435840"
        )

    # Create PyTorch reference model
    torch_model = Upsample(scale, num_feat).eval()

    # Create test input
    torch_input = torch.randn(batch_size, num_feat, input_size, input_size)
    torch_output = torch_model(torch_input)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_upsample_preprocessor(device),
        device=device,
    )

    # Convert input to TTNN format (NHWC)
    ttnn_input = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Create TTNN model and run inference
    ttnn_model = TTUpsample(scale, num_feat, device)

    ttnn_output = ttnn_model(ttnn_input, parameters=parameters)
    tt_torch_output = tt2torch_tensor(ttnn_output)
    tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)

    does_pass, pcc_message = check_with_pcc(torch_output, tt_torch_output, 0.99)

    logger.info(pcc_message)

    if does_pass:
        logger.info("Upsample Passed!")
    else:
        logger.warning("Upsample Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
