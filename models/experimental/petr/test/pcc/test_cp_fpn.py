# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger
from models.experimental.petr.tt.tt_cp_fpn import ttnn_CPFPN
from models.experimental.petr.reference.cp_fpn import CPFPN
from models.experimental.petr.tt.common import create_custom_preprocessor_cpfpn, infer_ttnn_module_args_cp_fpn


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cp_fpn(device, reset_seeds):
    torch_model = CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_cpfpn(None), device=None
    )

    batch_size = 6
    input_a = torch.randn(batch_size, 768, 20, 50)
    input_b = torch.randn(batch_size, 1024, 10, 25)
    torch_output = torch_model([input_a, input_b])
    ttnn_module_args = infer_ttnn_module_args_cp_fpn(
        model=torch_model, run_model=lambda model: model([input_a, input_b]), device=device
    )

    ttnn_model = ttnn_CPFPN(
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2,
        batch_size=batch_size,
        parameters=parameters,
        model_config=model_config,
        model_args=ttnn_module_args,
        device=device,
    )

    ttnn_input_1 = ttnn.from_torch(input_a.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_2 = ttnn.from_torch(input_b.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn_model([ttnn_input_1, ttnn_input_2])

    for i in range(len(ttnn_output)):
        ttnn_output_check = ttnn.to_torch(ttnn_output[i])
        ttnn_output_check = ttnn_output_check.permute(0, 3, 1, 2)
        pcc_threshold = 0.99
        passed, msg = check_with_pcc(torch_output[i], ttnn_output_check, pcc=pcc_threshold)
        assert_with_pcc(ttnn_output_check, torch_output[i], pcc=0.99)
        logger.info(f"cp_fpn layer  passed: " f"PCC={msg}")
