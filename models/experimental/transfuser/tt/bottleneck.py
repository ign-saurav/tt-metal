# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.transfuser.tt.utils import TTConv2D


class TTRegNetBottleneck:
    def __init__(
        self,
        device,
        parameters,
        model_config,
        layer_config,
        stride=1,
        downsample=False,
        groups=1,
    ):
        self.device = device
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.model_config = model_config

        # Extractconfig for each convolution
        conv1_config = layer_config.get("conv1", {})
        conv2_config = layer_config.get("conv2", {})
        se_fc1_config = layer_config.get("se_fc1", {})
        se_fc2_config = layer_config.get("se_fc2", {})
        conv3_config = layer_config.get("conv3", {})
        downsample_config = layer_config.get("downsample", {})

        # conv1: 1x1 convolution
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=conv1_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=conv1_config.get("act_block_h", None),
            enable_act_double_buffer=conv1_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=conv1_config.get("enable_weights_double_buffer", False),
            memory_config=conv1_config.get("memory_config", None),
            deallocate_activation=False,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            is_reshape=False,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )

        # conv2: 3x3 grouped convolution
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            parameters=parameters["conv2"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            groups=groups,
            shard_layout=conv2_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=conv2_config.get("act_block_h", None),
            enable_act_double_buffer=conv2_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=conv2_config.get("enable_weights_double_buffer", False),
            memory_config=conv2_config.get("memory_config", None),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            is_reshape=False,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )

        # SE Module
        self.se_fc1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc1"],
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=se_fc1_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc1_config.get("act_block_h", None),
            enable_act_double_buffer=se_fc1_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=se_fc1_config.get("enable_weights_double_buffer", False),
            memory_config=se_fc1_config.get("memory_config", None),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
            is_reshape=False,
        )

        self.se_fc2 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["se"]["fc2"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=se_fc2_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=se_fc2_config.get("act_block_h", None),
            enable_act_double_buffer=se_fc2_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=se_fc2_config.get("enable_weights_double_buffer", False),
            memory_config=se_fc2_config.get("memory_config", None),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )

        # conv3: 1x1 convolution (no activation)
        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["conv3"],
            kernel_fidelity=model_config,
            activation=None,
            shard_layout=downsample_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            act_block_h=downsample_config.get("act_block_h", None),
            enable_act_double_buffer=downsample_config.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=downsample_config.get("enable_weights_double_buffer", False),
            memory_config=downsample_config.get("memory_config", None),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )

        # Downsample layer if needed
        if downsample:
            self.downsample_layer = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                parameters=parameters["downsample"],
                kernel_fidelity=model_config,
                activation=None,
                shard_layout=conv3_config.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
                act_block_h=conv3_config.get("act_block_h", None),
                enable_act_double_buffer=conv3_config.get("enable_act_double_buffer", False),
                enable_weights_double_buffer=conv3_config.get("enable_weights_double_buffer", False),
                memory_config=conv3_config.get("memory_config", None),
                deallocate_activation=True,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
                packer_l1_acc=model_config.get("packer_l1_acc", True),
                math_approx_mode=model_config.get("math_approx_mode", False),
            )
        else:
            self.downsample_layer = None

    def __call__(self, x, device):
        identity = x
        logger.info(f"conv1- 1x1 convolution")
        print(f"0{x.shape=}")
        # conv1: 1x1 expansion
        out, shape_ = self.conv1(device, x, x.shape)
        print(f"1{out.shape=}")
        print(f"{shape_=}")

        logger.info(f"conv2- 3x3 grouped convolution")
        # conv2: 3x3 grouped convolution
        out, shape_ = self.conv2(device, out, shape_)
        print(f"2{out.shape=}")
        print(f"{shape_=}")

        # SE Module
        logger.info(f"SE module")
        logger.info(f"reduce mean")
        # Global average pooling
        out = ttnn.reshape(out, shape_)
        print(f"3{out.shape=}")
        print(f"{shape_=}")

        se_out = ttnn.mean(out, dim=[1, 2], keepdim=True)
        shape_ = se_out.shape

        logger.info(f"SE fc1")
        se_out, shape_ = self.se_fc1(device, se_out, shape_)

        logger.info(f"SE fc2")
        se_out, shape_ = self.se_fc2(device, se_out, shape_)
        se_out = ttnn.sigmoid(se_out)
        # Apply SE scaling
        out = ttnn.multiply(out, se_out)
        ttnn.deallocate(se_out)
        shape_ = out.shape

        # conv3: 1x1 projection
        out_temp, shape_ = self.conv3(device, out, shape_)
        ttnn.deallocate(out)
        out = ttnn.reshape(out_temp, shape_)

        # Handle downsample
        if self.downsample_layer is not None:
            identity_temp, _ = self.downsample_layer(device, identity, identity.shape)
            ttnn.deallocate(identity)
            identity = ttnn.reshape(identity_temp, shape_)

        logger.info(f"Add")
        out_final = ttnn.add(out, identity)
        ttnn.deallocate(out)
        ttnn.deallocate(identity)
        out = ttnn.relu(out_final)

        return out
