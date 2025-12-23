# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)

from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
)
from ttnn.dot_access import make_dot_access_dict
from ttnn.torch_tracer import trace, visualize
import torch

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


class MaxPoolConfiguration(MaxPool2dConfiguration):
    @classmethod
    def from_model_args(cls, maxpool2d_args, **kwargs):
        return cls(
            input_height=maxpool2d_args.input_height,
            input_width=maxpool2d_args.input_width,
            channels=maxpool2d_args.input_channels,
            batch_size=maxpool2d_args.batch_size,
            kernel_size=(maxpool2d_args.kernel_size, maxpool2d_args.kernel_size),
            stride=(maxpool2d_args.stride, maxpool2d_args.stride),
            padding=(maxpool2d_args.padding, maxpool2d_args.padding),
            dilation=(maxpool2d_args.dilation, maxpool2d_args.dilation),
            **kwargs,
        )


def post_conv_reshape(x, out_height=1, out_width=1):
    """Convert sharded conv output to [N,1,1,C] tile layout for SE block."""
    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], out_height, out_width, x.shape[3]))
    return ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)


# Helper function to create Conv2dConfiguration from parameters
def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    sharding_strategy=AutoShardedStrategyConfiguration(),
) -> Conv2dConfiguration:
    """
    Conv2dConfiguration from parameters dict for SqueezeExcitation.
    """

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
    )


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "nearest",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(
        self,
        device,
        input_tensor,
        input_shape=None,
        reshape_output=False,
        pad_ch_to_32=False,
        sent_to_dram=False,
        dtype=ttnn.bfloat8_b,
    ):
        # Convert a **sharded** tensor (distributed across cores) into a single **interleaved** tensor, choosing the backing memory
        # - DRAM: use when tensors are large or when later ops expect DRAM residency.
        # - L1  : fastest on-chip memory; use when the tensor fits and you’ll run
        #         compute-heavy kernels immediately after.
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Optionally pad channels to a multiple of 32 to match TT tile/channel alignment.
        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Remove channel padding if added.
        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            B, H, W, C = output_tensor.shape
            output_tensor = ttnn.reshape(output_tensor, [1, 1, B * H * W, C])

        return output_tensor


class ModuleArgs(dict):
    ...


class Conv2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class ConvTranspose2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class MaxPool2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class GroupNormArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    # ------------------------------------------------------------------
    # Run model under TTNN tracing
    # ------------------------------------------------------------------
    with trace():
        output = run_model(model)

    visualize(output, file_name=ttnn.CONFIG.tmp_dir / "model_graph.svg")

    # ------------------------------------------------------------------
    # Helper: insert value into nested dict using module path
    # ------------------------------------------------------------------
    def insert_nested(d, path, value):
        for key in path[:-1]:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            d = d.setdefault(key, {})
        last = path[-1]
        last = int(last) if isinstance(last, str) and last.isdigit() else last
        d[last] = value

    # ------------------------------------------------------------------
    # Recursive graph walk
    # ------------------------------------------------------------------
    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}

        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes.get("operation")

            if not isinstance(operation, ttnn.tracer.TorchModule):
                continue

            # Full hierarchical module path
            module_path = operation.module.__ttnn_tracer_name__.split(".")

            # Infer input shape (assumes single input edge)
            in_edges = list(graph.in_edges(node, data=True))
            if not in_edges:
                continue

            input_node, _, edge_data = in_edges[0]
            input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

            module = operation.module

            # ----------------------------------------------------------
            # Conv2d
            # ----------------------------------------------------------
            if isinstance(module, torch.nn.Conv2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    Conv2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    ),
                )

            # ----------------------------------------------------------
            # ConvTranspose2d
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.ConvTranspose2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    ConvTranspose2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        output_padding=module.output_padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    ),
                )

            # ----------------------------------------------------------
            # MaxPool2d
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.MaxPool2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    MaxPool2dArgs(
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # ----------------------------------------------------------
            # GroupNorm
            # ----------------------------------------------------------
            elif isinstance(module, torch.nn.GroupNorm):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    GroupNormArgs(
                        num_groups=module.num_groups,
                        num_channels=module.num_channels,
                        eps=module.eps,
                        affine=module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # ----------------------------------------------------------
            # Container / composite module → recurse
            # ----------------------------------------------------------
            else:
                nested = _infer_ttnn_module_args(operation.graph)
                if nested:
                    insert_nested(
                        ttnn_module_args,
                        module_path,
                        nested,
                    )

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    # ------------------------------------------------------------------
    # Kick off inference from traced graph
    # ------------------------------------------------------------------
    full_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))

    # Root module is stored under empty name ""
    return full_args.get("", full_args)
