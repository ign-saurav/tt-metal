# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.BevDepth.reference.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.ttnn_bevdepth_head import TtBEVDepthHead, head_optimisations
from models.experimental.BevDepth.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.BevDepth.common import download_bevdepth_weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_head(device):
    torch.manual_seed(42)

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    inputs_mesh_mapper = None
    weights_mesh_mapper = None
    output_mesh_composer = None
    if device.get_num_devices() != 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)

    torch_model = BEVDepthLightningModel()
    weights_path = download_bevdepth_weights()
    torch_model.load_checkpoint(weights_path)
    torch_model = torch_model.model.head
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 160, 128, 128, dtype=torch.float32, requires_grad=False)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model,
        run_model=lambda m: m(torch_input_tensor),
        device=None,
    )

    ttnn_model = TtBEVDepthHead(parameters, model_config, layer_optimisations=head_optimisations, device=device)

    tt_host_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    input_tensor = ttnn.to_device(tt_host_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = ttnn_model(input_tensor, device=device)

    for head_idx, (torch_head_list, tt_head_output) in enumerate(zip(torch_output, ttnn_output)):
        torch_head_output = torch_head_list[0]

        for key in torch_head_output.keys():
            torch_tensor = torch_head_output[key]
            tt_output_tuple = tt_head_output[key]
            tt_tensor, _ = tt_output_tuple

            torch_tensor_tt = ttnn.to_torch(tt_tensor, device=device, mesh_composer=output_mesh_composer)
            torch_tensor_tt = torch.reshape(
                torch_tensor_tt,
                (torch_tensor.shape[0], torch_tensor.shape[2], torch_tensor.shape[3], torch_tensor.shape[1]),
            )
            tt_tensor_torch = torch.permute(torch_tensor_tt, (0, 3, 1, 2))

            ttnn.deallocate(tt_tensor)

            passed, msg = check_with_pcc(torch_tensor, tt_tensor_torch, pcc=0.99)

            logger.info(f"Head {head_idx}, key '{key}': {msg}")
            assert passed, f"Head {head_idx}, key '{key}' PCC check failed: {msg}"
