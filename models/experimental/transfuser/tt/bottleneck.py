# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
)


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
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        torch_model=None,
        parameters_torch=None,
        use_fallback=False,
        block_name=None,
        stage_name=None,
    ):
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.model_config = model_config
        self.dtype = ttnn.bfloat16

        # Extract per-layer override dicts
        conv1_cfg = layer_config.get("conv1", {})
        conv2_cfg = layer_config.get("conv2", {})
        se_fc1_cfg = layer_config.get("se_fc1", {})
        se_fc2_cfg = layer_config.get("se_fc2", {})
        conv3_cfg = layer_config.get("conv3", {})
        downsample_cfg = layer_config.get("downsample", {})

        self.torch_model = torch_model
        self.use_fallback = use_fallback
        self.block_name = block_name
        self.stage_name = stage_name

        def make_conv2d(
            params_key,
            *,
            kernel_size,
            stride,
            padding,
            activation,
            cfg_overrides,
            groups=None,
            is_reshape=False,
        ):
            return TTConv2D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                parameters=parameters[params_key],
                kernel_fidelity=model_config,
                activation=activation,
                groups=(groups if groups is not None else 1),
                shard_layout=cfg_overrides.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
                act_block_h=cfg_overrides.get("act_block_h", None),
                memory_config=cfg_overrides.get("memory_config", ttnn.L1_MEMORY_CONFIG),
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                deallocate_activation=True
                if kernel_size != 1 or params_key in ("conv2", "se_fc1", "se_fc2", "conv3", "downsample")
                else False,
                reallocate_halo_output=True,
                reshard_if_not_optimal=True,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
                packer_l1_acc=model_config.get("packer_l1_acc", True),
                math_approx_mode=model_config.get("math_approx_mode", False),
                is_reshape=is_reshape,
            )

        conv1_params = parameters_torch["conv1"]["conv"]
        # ------------------------- conv1: 1x1 + ReLU -------------------------
        conv1_config = self._create_conv_config(
            parameters=parameters["conv1"],
            batch_size=conv1_params["batch_size"],
            input_height=conv1_params["input_height"],
            input_width=conv1_params["input_width"],
            in_channels=conv1_params["in_channels"],
            out_channels=conv1_params["out_channels"],
            stride=conv1_params["stride"],
            kernel_size=conv1_params["kernel_size"],
            padding=conv1_params["padding"],
            groups=conv1_params["groups"],
        )
        self.conv1 = TtConv2d(conv1_config, device=device)

        conv2_params = parameters_torch["conv2"]["conv"]
        conv2_config = self._create_conv_config(
            parameters=parameters["conv2"],
            batch_size=conv2_params["batch_size"],
            input_height=conv2_params["input_height"],
            input_width=conv2_params["input_width"],
            in_channels=conv2_params["in_channels"],
            out_channels=conv2_params["out_channels"],
            stride=conv2_params["stride"],
            kernel_size=conv2_params["kernel_size"],
            padding=conv2_params["padding"],
            groups=conv2_params["groups"],
        )
        # --------------------- conv2: 3x3 grouped + ReLU ---------------------
        self.conv2 = TtConv2d(conv2_config, device=device)
        # --------------------------- SE: fc1 (1x1 + ReLU) --------------------
        # Extract SE fc1 parameters (you'll need to get these from your parameters_torch or similar source)
        se_fc1_params = parameters_torch["se"]["fc1"]  # Adjust based on your actual parameter structure

        # Create configuration for SE fc1
        se_fc1_config = self._create_conv_config(
            parameters=parameters["se"]["fc1"],
            batch_size=se_fc1_params["batch_size"],
            input_height=se_fc1_params["input_height"],
            input_width=se_fc1_params["input_width"],
            in_channels=se_fc1_params["in_channels"],
            out_channels=se_fc1_params["out_channels"],
            stride=se_fc1_params["stride"],
            kernel_size=se_fc1_params["kernel_size"],
            padding=se_fc1_params["padding"],
            groups=se_fc1_params["groups"],
        )
        self.se_fc1 = TtConv2d(se_fc1_config, device=device)
        # --------------------------- SE: fc2 (1x1, no act) -------------------
        # Extract SE fc2 parameters
        se_fc2_params = parameters_torch["se"]["fc2"]

        # Create configuration for SE fc2
        se_fc2_config = self._create_conv_config(
            parameters=parameters["se"]["fc2"],
            batch_size=se_fc2_params["batch_size"],
            input_height=se_fc2_params["input_height"],
            input_width=se_fc2_params["input_width"],
            in_channels=se_fc2_params["in_channels"],
            out_channels=se_fc2_params["out_channels"],
            stride=se_fc2_params["stride"],
            kernel_size=se_fc2_params["kernel_size"],
            padding=se_fc2_params["padding"],
            groups=se_fc2_params["groups"],
            activation=None,
            # enable_act_double_buffer=True,
            # enable_weights_double_buffer=True,
            # deallocate_activation=True,
            # reallocate_halo_output=True,
            # reshard_if_not_optimal=True,
        )

        self.se_fc2 = TtConv2d(se_fc2_config, device=device)
        # ----------------------- conv3: 1x1 projection (no act) --------------
        conv3_params = parameters_torch["conv3"]["conv"]

        conv3_config = self._create_conv_config(
            parameters=parameters["conv3"],
            batch_size=conv3_params["batch_size"],
            input_height=conv3_params["input_height"],
            input_width=conv3_params["input_width"],
            in_channels=conv3_params["in_channels"],
            out_channels=conv3_params["out_channels"],
            stride=conv3_params["stride"],
            kernel_size=conv3_params["kernel_size"],
            padding=conv3_params["padding"],
            groups=conv3_params["groups"],
            activation=None,
        )
        self.conv3 = TtConv2d(conv3_config, device=device)

        # ------------------------------ optional downsample -------------------
        # if downsample:
        #     downsample_conv_params = parameters_torch.downsample[0]
        #     downsample_conv_config = self._create_conv_config(
        #         parameters=parameters["downsample"],
        #         batch_size=downsample_conv_params["batch_size"],
        #         input_height=downsample_conv_params["input_height"],
        #         input_width=downsample_conv_params["input_width"],
        #         in_channels=downsample_conv_params["in_channels"],
        #         out_channels=downsample_conv_params["out_channels"],
        #         stride=downsample_conv_params["stride"],
        #         kernel_size=downsample_conv_params["kernel_size"],
        #         padding=downsample_conv_params["padding"],
        #         groups=downsample_conv_params["groups"],
        #         activation=None,
        #     )
        #     self.downsample_layer = TtConv2d(downsample_conv_config, device=device)
        #     # self.downsample_layer = TTConv2D(
        #     #     kernel_size=1,
        #     #     stride=stride,
        #     #     padding=0,
        #     #     parameters=parameters["downsample"],
        #     #     kernel_fidelity=model_config,
        #     #     activation=None,
        #     #     shard_layout=downsample_cfg.get("shard_layout", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        #     #     act_block_h=downsample_cfg.get("act_block_h", None),
        #     #     memory_config=se_fc1_cfg.get("memory_config", ttnn.L1_MEMORY_CONFIG),
        #     #     enable_act_double_buffer=True,
        #     #     enable_weights_double_buffer=True,
        #     #     deallocate_activation=True,
        #     #     reallocate_halo_output=True,
        #     #     reshard_if_not_optimal=True,
        #     #     dtype=ttnn.bfloat16,
        #     #     fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
        #     #     packer_l1_acc=model_config.get("packer_l1_acc", True),
        #     #     math_approx_mode=model_config.get("math_approx_mode", False),
        #     # )
        # else:
        #     self.downsample_layer = None

    def _create_conv_config(
        self,
        parameters,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        padding,
        groups,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    ):
        # Convert weights to float32 format (required by tt_cnn builder)
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor):
            weight = ttnn.from_torch(ttnn.to_torch(weight), dtype=ttnn.float32)

        # Convert bias to shape (1, 1, 1, out_channels) in float32
        bias = None
        if "bias" in parameters and parameters.bias is not None:
            bias_torch = ttnn.to_torch(parameters.bias).reshape(1, 1, 1, -1)
            bias = ttnn.from_torch(bias_torch, dtype=ttnn.float32)

        # Convert stride to list format (required by ttnn.conv2d)
        if isinstance(stride, int):
            stride_list = [stride, stride]
        elif isinstance(stride, tuple) and len(stride) == 2:
            stride_list = list(stride)
        else:
            stride_list = stride

        # Convert padding to list format (required by ttnn.conv2d)
        if isinstance(padding, int):
            padding_list = [padding, padding]
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding_list = list(padding)
        elif isinstance(padding, tuple) and len(padding) == 4:
            padding_list = list(padding)
        else:
            padding_list = padding

        # Select math fidelity based on block (HiFi4 for block 2 for better accuracy)
        math_fidelity = ttnn.MathFidelity.HiFi4

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            stride=stride_list,  # List format
            padding=padding_list,  # List format
            groups=groups,
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            enable_act_double_buffer=False,
        )

    def __call__(self, x, device, input_shape=None):
        if input_shape is None:
            input_shape = x.shape
        identity = x
        identity_shape = input_shape

        # conv1- 1x1 convolution (using new TtConv2d interface)
        out = self.conv1(x)

        # The rest of the code remains the same but would need similar updates
        # to use the new TtConv2d interface consistently        # conv2- 3x3 grouped convolution
        out, (height, width) = self.conv2(out, return_output_dim=True)
        # return out, out

        # SE module
        # reduce mean
        out1 = ttnn.reallocate(out)
        # Reshape to 4D for mean operation
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, height, width, out.shape[-1]))
        se_out = ttnn.mean(out, dim=[1, 2], keepdim=True)
        if self.use_fallback and self.torch_model is not None:
            # Falling Back SE module
            se_out_torch = ttnn.to_torch(
                se_out,
                device=device,
            )
            se_out_torch = torch.permute(se_out_torch, (0, 3, 1, 2))
            se_out_torch = se_out_torch.to(torch.float32)
            se_out_torch = self.torch_model.fallback(
                se_out_torch, block_name=self.block_name, stage_name=self.stage_name
            )
            se_out = ttnn.from_torch(
                se_out_torch,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                device=device,
            )
            se_out = ttnn.permute(se_out, (0, 2, 3, 1))

        else:
            # SE fc1
            se_out = self.se_fc1(se_out)
            # se_out, se_shape = self.se_fc1(device, se_out, se_out.shape)

            # SE fc2
            se_out = self.se_fc2(se_out)
            se_out = ttnn.sigmoid(se_out)
        out_4d = ttnn.multiply(out1, se_out)
        # Flatten back to match identity format
        batch, channels = out_4d.shape[0], out_4d.shape[-1]
        out = ttnn.reshape(out_4d, (1, 1, batch * height * width, channels))

        # conv3: 1x1 projection - now in flattened format
        out = self.conv3(out)
        return out, out
        # Handle downsample - identity is already in flattened format
        if self.downsample_layer is not None:
            # downsample
            identity = self.downsample_layer(identity)

        return identity, identity
        # Add
        # Both tensors are now in flattened format
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, out
