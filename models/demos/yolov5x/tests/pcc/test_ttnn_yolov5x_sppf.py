# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ttnn
from models.demos.yolov5x.common import YOLOV5X_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov5x.tt.model_preprocessing import create_yolov5x_input_tensors, create_yolov5x_model_parameters
from models.demos.yolov5x.tt.sppf import TtnnSPPF
from tests.ttnn.utils_for_testing import assert_with_pcc

# from .plot_utils import plot_abs_diff


def _analyze_outputs(torch_out, ttnn_out):
    """Print max absolute error and other stats for torch vs ttnn outputs."""
    t = torch_out.detach().cpu().numpy().astype(np.float64)
    tt = (
        ttnn_out.detach().cpu().numpy().astype(np.float64)
        if hasattr(ttnn_out, "detach")
        else np.asarray(ttnn_out, dtype=np.float64)
    )
    abs_diff = np.abs(t - tt)
    max_ae = float(np.max(abs_diff))
    mean_ae = float(np.mean(abs_diff))
    std_ae = float(np.std(abs_diff))
    median_ae = float(np.median(abs_diff))
    flat_idx = np.argmax(abs_diff)
    max_idx = np.unravel_index(flat_idx, abs_diff.shape)
    torch_at_max = float(t[max_idx])
    tt_at_max = float(tt[max_idx])
    print("SPPF output analysis (Torch vs TTNN):")
    print(f"  Shape: {t.shape}  Total elements: {abs_diff.size}")
    print(f"  Max absolute error:    {max_ae:.6g}")
    print(f"  Mean absolute error:  {mean_ae:.6g}")
    print(f"  Std absolute error:   {std_ae:.6g}")
    print(f"  Median absolute error: {median_ae:.6g}")
    print(f"  Max error at index {max_idx}: torch={torch_at_max:.6g}, ttnn={tt_at_max:.6g}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE}], indirect=True)
def test_yolov5x_SPPF(device, reset_seeds, model_location_generator, request):
    fwd_input_shape = [1, 1280, 20, 20]
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = load_torch_model(model_location_generator)
    torch_model = torch_model.model.model[9]

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnSPPF(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    pcc_passed, pcc_value = assert_with_pcc(torch_model_output, ttnn_output, 0.99)
    print(f"PCC value: {pcc_value}")
    _analyze_outputs(torch_model_output, ttnn_output)
    # plot_abs_diff(torch_model_output, ttnn_output, request.node.name)
