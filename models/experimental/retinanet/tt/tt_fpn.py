# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.tt.utils import TTUpsample
from collections import OrderedDict

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.retinanet.tt.utils import _create_conv_config_from_params
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
)


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
        layer_optimisations=fpn_optimisations,
    ) -> None:
        # self.conv1 = TTConv2D(
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     parameters=parameters["inner_blocks"].get("0", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv1,
        # )

        self.conv_config_1 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["inner_blocks"].get("0", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["inner_blocks"].get("0", {}).get("0", None)["weight"].shape[0],
            kernel_size=1,
            batch_size=1,
            parameters=parameters["inner_blocks"].get("0", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv1 = TtConv2d(self.conv_config_1, device)

        # self.conv2 = TTConv2D(
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     parameters=parameters["inner_blocks"].get("1", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv2,
        # )

        self.conv_config_2 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["inner_blocks"].get("1", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["inner_blocks"].get("1", {}).get("0", None)["weight"].shape[0],
            kernel_size=1,
            batch_size=1,
            parameters=parameters["inner_blocks"].get("1", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv2 = TtConv2d(self.conv_config_2, device)

        # self.conv3 = TTConv2D(
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     parameters=parameters["inner_blocks"].get("2", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv3,
        # )

        self.conv_config_3 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["inner_blocks"].get("2", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["inner_blocks"].get("2", {}).get("0", None)["weight"].shape[0],
            kernel_size=1,
            batch_size=1,
            parameters=parameters["inner_blocks"].get("2", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv3 = TtConv2d(self.conv_config_3, device)

        # self.conv4 = TTConv2D(
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     parameters=parameters["layer_blocks"].get("0", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv4,
        # )

        self.conv_config_4 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["layer_blocks"].get("0", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["layer_blocks"].get("0", {}).get("0", None)["weight"].shape[0],
            kernel_size=3,
            batch_size=1,
            padding=(1, 1),
            parameters=parameters["layer_blocks"].get("0", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv4 = TtConv2d(self.conv_config_4, device)

        # self.conv5 = TTConv2D(
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     parameters=parameters["layer_blocks"].get("1", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv5,
        # )

        self.conv_config_5 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["layer_blocks"].get("1", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["layer_blocks"].get("1", {}).get("0", None)["weight"].shape[0],
            kernel_size=3,
            batch_size=1,
            padding=(1, 1),
            parameters=parameters["layer_blocks"].get("1", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv5 = TtConv2d(self.conv_config_5, device)

        # self.conv6 = TTConv2D(
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     parameters=parameters["layer_blocks"].get("2", {}).get("0", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv6,
        # )

        self.conv_config_6 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=parameters["layer_blocks"].get("2", {}).get("0", None)["weight"].shape[1],
            out_channels=parameters["layer_blocks"].get("2", {}).get("0", None)["weight"].shape[0],
            kernel_size=3,
            batch_size=1,
            padding=(1, 1),
            parameters=parameters["layer_blocks"].get("2", {}).get("0", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv6 = TtConv2d(self.conv_config_6, device)

        # self.conv7 = TTConv2D(
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     parameters=getattr(parameters.extra_blocks, "p6", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv7,
        # )

        self.conv_config_7 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=getattr(parameters.extra_blocks, "p6", None)["weight"].shape[1],
            out_channels=getattr(parameters.extra_blocks, "p6", None)["weight"].shape[0],
            kernel_size=3,
            batch_size=1,
            padding=(1, 1),
            parameters=getattr(parameters.extra_blocks, "p6", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv7 = TtConv2d(self.conv_config_7, device)

        # self.conv8 = TTConv2D(
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     parameters=getattr(parameters.extra_blocks, "p7", None),
        #     kernel_fidelity=model_config,
        #     activation=None,
        #     **layer_optimisations.conv8,
        # )

        self.conv_config_8 = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=getattr(parameters.extra_blocks, "p7", None)["weight"].shape[1],
            out_channels=getattr(parameters.extra_blocks, "p7", None)["weight"].shape[0],
            kernel_size=3,
            batch_size=1,
            padding=(1, 1),
            parameters=getattr(parameters.extra_blocks, "p7", None),
            activation=None,
            sharding_strategy=AutoShardedStrategyConfiguration(),
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
