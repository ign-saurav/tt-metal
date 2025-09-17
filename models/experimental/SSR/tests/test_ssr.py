# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

# Import reference and TTNN models
from models.experimental.SSR.reference.SSR.model.ssr import SSR, SSR_wo_conv
from models.experimental.SSR.tt.ssr import TTSSR, TTSSR_wo_conv
from models.experimental.SSR.tests.tile_refinement.test_upsample import create_upsample_preprocessor
from models.experimental.SSR.tests.tile_selection.test_tile_selection import create_tile_selection_preprocessor
from models.experimental.SSR.tests.tile_refinement.test_tile_refinement import create_tile_refinement_preprocessor
from models.experimental.SSR.tests.tile_refinement.test_HAB import create_relative_position_index
from models.experimental.SSR.reference.SSR.model.net_blocks import window_reverse


from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def create_ssr_preprocessor(device, args, num_cls, depth, weight_dtype=ttnn.bfloat16):
    """Custom preprocessor for SSR model"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if isinstance(torch_model, SSR) or isinstance(torch_model, SSR_wo_conv):
            # Preprocess tile selection model
            select_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.select_model,
                custom_preprocessor=create_tile_selection_preprocessor(device, weight_dtype=weight_dtype),
                device=device,
            )
            parameters["select_model"] = select_params

            # Preprocess tile refinement model

            rpi_sa = create_relative_position_index((16, 16))

            attn_mask = None

            # Create RPI for OCAB
            overlap_win_size = int(16 * 0.5) + 16
            rpi_oca = torch.zeros((16 * 16, overlap_win_size * overlap_win_size), dtype=torch.long)

            # Create params dictionary

            tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

            tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

            forward_params = {"rpi_sa": tt_rpi_sa, "attn_mask": attn_mask, "rpi_oca": tt_rpi_oca}
            sr_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.sr_model,
                custom_preprocessor=create_tile_refinement_preprocessor(
                    device, forward_params, window_size=16, rpi_sa=rpi_sa, depth=depth
                ),
                device=device,
            )
            parameters["sr_model"] = sr_params

            # Preprocess conv layers
            conv_layers = ["conv_first", "conv_last"]
            for conv_name in conv_layers:
                if hasattr(torch_model, conv_name):
                    conv_layer = getattr(torch_model, conv_name)
                    parameters[conv_name] = {
                        "weight": ttnn.from_torch(conv_layer.weight, dtype=ttnn.bfloat16),
                        "bias": ttnn.from_torch(conv_layer.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                    }

            # Preprocess conv_before_upsample (Sequential layer)
            if hasattr(torch_model, "conv_before_upsample"):
                conv_layer = torch_model.conv_before_upsample[0]  # Conv2d layer
                parameters["conv_before_upsample"] = {
                    "weight": ttnn.from_torch(conv_layer.weight, dtype=ttnn.bfloat16),
                    "bias": ttnn.from_torch(conv_layer.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                }

            # Preprocess upsample
            if hasattr(torch_model, "upsample"):
                upsample_params = preprocess_model_parameters(
                    initialize_model=lambda: torch_model.upsample,
                    custom_preprocessor=create_upsample_preprocessor(device),
                    device=device,
                )
                parameters["upsample"] = upsample_params

        return parameters

    return custom_preprocessor


class MockArgs:
    """Mock args class for testing"""

    def __init__(self):
        self.token_size = 4
        self.imgsz = 256
        self.patchsz = 2
        self.pretrain = False
        self.ckpt = None
        self.dim = 96


@pytest.mark.parametrize(
    "input_shape, num_cls, with_conv, depth, num_heads",
    [
        # ((1, 3, 256, 256), 1, True),
        ((1, 3, 256, 256), 1, False, [1], [1]),
        ((1, 3, 256, 256), 1, False, [6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 6]),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_ssr_model(input_shape, num_cls, with_conv, depth, num_heads, input_dtype, weight_dtype):
    """Test TTSSR model against PyTorch reference"""
    # Create input tensor
    x = torch.randn(input_shape)
    _, _, H, W = x.shape

    # Create mock args
    args = MockArgs()

    # Create reference PyTorch model
    if with_conv:
        ref_model = SSR(args, num_cls, depth, num_heads)
    else:
        ref_model = SSR_wo_conv(args, num_cls, depth, num_heads)
    ref_model.eval()

    # Get reference output
    with torch.no_grad():
        ref_sr, ref_patch_fea3, _, _ = ref_model(x)
    # Open TTNN device with larger L1 cache to handle memory requirements
    device = ttnn.open_device(device_id=0, l1_small_size=32768)  # 128KB instead of 32KB

    try:
        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_model,
            custom_preprocessor=create_ssr_preprocessor(device, args, num_cls, depth, weight_dtype),
            device=device,
        )
        # Create TTNN model
        if with_conv:
            tt_model = TTSSR(
                device=device,
                parameters=parameters,
                args=args,
                num_cls=num_cls,
                depth=depth,
                num_heads=num_heads,
            )
        else:
            tt_model = TTSSR_wo_conv(
                device=device,
                parameters=parameters,
                args=args,
                num_cls=num_cls,
                depth=depth,
                num_heads=num_heads,
                dtype=input_dtype,
            )

        # Convert input to TTNN tensor
        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)

        # Run TTNN model
        tt_sr, tt_patch_fea3 = tt_model(tt_input)

        # Convert back to torch tensors
        tt_torch_sr = tt2torch_tensor(tt_sr)
        tt_torch_patch_fea3 = tt2torch_tensor(tt_patch_fea3)
        tt_torch_sr = tt_torch_sr.permute(0, 3, 1, 2)

        if not with_conv:
            _, _, H, W = x.shape
            tt_torch_sr = window_reverse(tt_torch_sr.permute(0, 2, 3, 1), window_size=H, H=H * 4, W=W * 4)
            tt_torch_sr = tt_torch_sr.permute(0, 3, 1, 2)

        # Compare outputs
        sr_pass, sr_pcc_message = check_with_pcc(ref_sr, tt_torch_sr, 0.90)
        fea3_pass, fea3_pcc_message = check_with_pcc(ref_patch_fea3, tt_torch_patch_fea3, 0.90)
        logger.info(f"sr_pcc: {sr_pcc_message}")
        logger.info(f"fea3_pcc: {fea3_pcc_message}")

        all_pass = sr_pass and fea3_pass

        if all_pass:
            logger.info("TTSSR Test Passed!")
        else:
            logger.warning("TTSSR Test Failed!")

        assert sr_pass, f"SR output failed PCC check: {sr_pcc_message}"
        assert fea3_pass, f"Patch fea3 failed PCC check: {fea3_pcc_message}"

    finally:
        ttnn.close_device(device)
