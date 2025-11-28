# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


def ttnn_conv2d(
    input_tensor,
    weight_tensor,
    device,
    batch_size,
    input_height,
    input_width,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias_tensor=None,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    weights_dtype=ttnn.bfloat16,
    activations_dtype=ttnn.bfloat16,
    deallocate_activation=True,
    reallocate_halo_output=False,
    shard_layout=None,
    packer_l1_acc=False,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
):
    """
    Wrapper for ttnn.conv2d with common optimizations.
    Handles conversion of PyTorch tensors to TTNN format.
    """
    import torch

    # Convert PyTorch weights to TTNN if needed
    if isinstance(weight_tensor, torch.Tensor):
        weight_tensor = ttnn.from_torch(
            weight_tensor,
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    if bias_tensor is not None and isinstance(bias_tensor, torch.Tensor):
        # Reshape bias from (out_channels,) to (1, 1, 1, out_channels) for TTNN
        if len(bias_tensor.shape) == 1:
            bias_tensor = bias_tensor.view(1, 1, 1, -1)
        bias_tensor = ttnn.from_torch(
            bias_tensor,
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        activation=activation,
        shard_layout=shard_layout if shard_layout else ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=reallocate_halo_output,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        packer_l1_acc=packer_l1_acc,
    )

    output = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=activations_dtype,
    )

    return output[0]
