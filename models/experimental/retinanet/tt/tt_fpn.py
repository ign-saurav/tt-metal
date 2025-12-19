# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.tt.utils import TTUpsample
from collections import OrderedDict

from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import Conv2dConfiguration


@dataclass
class FpnOptimizer:
    conv1: dict
    conv2: dict
    conv3: dict
    conv4: dict
    conv5: dict
    conv6: dict
    conv7: dict
    conv8: dict


fpn_optimisations = FpnOptimizer(
    conv1={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv2={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv3={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv4={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv5={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv6={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv7={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv8={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
)


class resnet50Fpn:
    def __init__(
        self,
        device,
        parameters,
        model_config,
        model_args,
        layer_optimisations=fpn_optimisations,
    ) -> None:
        print(model_args)
        self.conv_config_1 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["inner_blocks"].get("0", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("0", {}).get("0", None)["bias"],
            # **layer_optimisations.conv1,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv1 = TtConv2d(self.conv_config_1, device)

        self.conv_config_2 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["inner_blocks"].get("1", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("1", {}).get("0", None)["bias"],
            # **layer_optimisations.conv2,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv2 = TtConv2d(self.conv_config_2, device)

        self.conv_config_3 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["inner_blocks"].get("2", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("2", {}).get("0", None)["bias"],
            # **layer_optimisations.conv3,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv3 = TtConv2d(self.conv_config_3, device)

        self.conv_config_4 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["layer_blocks"].get("0", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("0", {}).get("0", None)["bias"],
            # **layer_optimisations.conv4,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv4 = TtConv2d(self.conv_config_4, device)

        self.conv_config_5 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["layer_blocks"].get("1", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("1", {}).get("0", None)["bias"],
            # **layer_optimisations.conv5,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv5 = TtConv2d(self.conv_config_5, device)

        self.conv_config_6 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters["layer_blocks"].get("2", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("2", {}).get("0", None)["bias"],
            # **layer_optimisations.conv6,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv6 = TtConv2d(self.conv_config_6, device)

        self.conv_config_7 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=getattr(parameters.extra_blocks, "p6", None)["weight"],
            bias=getattr(parameters.extra_blocks, "p6", None)["bias"],
            # **layer_optimisations.conv7,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv7 = TtConv2d(self.conv_config_7, device)

        self.conv_config_8 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=getattr(parameters.extra_blocks, "p7", None)["weight"],
            bias=getattr(parameters.extra_blocks, "p7", None)["bias"],
            # **layer_optimisations.conv8,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv8 = TtConv2d(self.conv_config_8, device)

        self.upsample1 = TTUpsample(
            scale_factor=(2),
            mode="nearest",
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        self.upsample2 = TTUpsample(
            scale_factor=(2),
            mode="nearest",
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        device,
    ):
        C3, C4, C5 = x.values()
        C5_clone = ttnn.clone(C5)

        L3, [_out_height, _out_width] = self.conv1(C3, return_output_dim=True)
        L4, [_out_height, _out_width] = self.conv2(C4, return_output_dim=True)
        L5, [_out_height, _out_width] = self.conv3(C5, return_output_dim=True)

        P5 = L5
        P5_interpolated = self.upsample1(device, P5, P5.shape, reshape_output=False, sent_to_dram=False)

        P4 = ttnn.add(L4, P5_interpolated)
        P4_interpolated = self.upsample1(device, P4, P4.shape, reshape_output=False, sent_to_dram=False)

        P3 = ttnn.add(L3, P4_interpolated)

        P3, [_out_height, _out_width] = self.conv4(P3, return_output_dim=True)
        P4, [_out_height, _out_width] = self.conv5(P4, return_output_dim=True)
        P5, [_out_height, _out_width] = self.conv6(P5, return_output_dim=True)

        P6, [_out_height, _out_width] = self.conv7(C5_clone, return_output_dim=True)
        P6_relu = ttnn.relu(P6)
        P7, [_out_height, _out_width] = self.conv8(P6_relu, return_output_dim=True)

        out = OrderedDict([("0", P3), ("1", P4), ("2", P5), ("p6", P6), ("p7", P7)])
        return out
