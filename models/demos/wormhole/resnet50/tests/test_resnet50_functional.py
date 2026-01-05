# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.ttnn_resnet.tests.common.resnet50_test_infra import create_test_infra

# Path to the checkpoint file
RESNET50_CHECKPOINT = "resnet50-19c8e357.pth"


def get_checkpoint_model_location_generator(checkpoint_path):
    """
    Create a model_location_generator that returns the specified checkpoint path.
    """

    def model_location_generator(model_version, model_subdir=None):
        # Return the checkpoint path regardless of the requested version
        return checkpoint_path

    return model_location_generator


def run_resnet_50(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    checkpoint_path=None,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")

    if batch_size > 16 and not is_blackhole():
        pytest.skip("Batch size > 16 is not supported on non-blackhole devices")

    # Use checkpoint path if provided, otherwise use default model_location_generator
    if checkpoint_path is not None:
        model_location_gen = get_checkpoint_model_location_generator(checkpoint_path)
    else:
        model_location_gen = None

    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_gen,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # First run configures convs JIT
    test_infra.run()
    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    # More optimized run with caching
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    passed, message = test_infra.validate()
    assert passed, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        (32, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    ),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_resnet_50(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
):
    run_resnet_50(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        checkpoint_path=RESNET50_CHECKPOINT,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        # MapTR backbone config: batch_size=16 for multi-camera processing
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
    ),
)
def test_resnet_50_maptr_backbone(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
):
    """
    Test ResNet50 backbone configuration for MapTR.
    MapTR uses ResNet50 as a feature extraction backbone for HD map construction.
    Uses pretrained ImageNet weights from resnet50-19c8e357.pth checkpoint.
    """
    run_resnet_50(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight=True,
        checkpoint_path=RESNET50_CHECKPOINT,
    )
