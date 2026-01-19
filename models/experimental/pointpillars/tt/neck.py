# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.pointpillars.tt.utils import TtPointPillarsConvTranspose2D, TtPointPillarsConvTranspose2DSplit


class TtNeck:
    def __init__(
        self,
        in_channels,
        upsample_strides,
        out_channels,
        parameters,
        device,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.in_channels = in_channels
        self.upsample_strides = upsample_strides
        self.out_channels = out_channels
        self.parameters = parameters

        # Initialize decoder blocks
        self.decoder_blocks = []

        self.bn_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # for i in range(len(in_channels)):
        decoder_block_0 = TtPointPillarsConvTranspose2D(
            conv_transpose=parameters[f"decoder_0"]["conv_args"],
            conv_transpose_pth=parameters[f"decoder_0"],
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=dtype,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=deallocate_activation,
            reshape_output=True,
        )
        self.decoder_blocks.append(decoder_block_0)

        decoder_block_1 = TtPointPillarsConvTranspose2DSplit(
            conv_transpose=parameters[f"decoder_1"]["conv_args"],
            conv_transpose_pth=parameters[f"decoder_1"],
            device=device,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=deallocate_activation,
            conv_in_channel_split_factor=2,
            conv_out_channel_split_factor=2,
        )

        self.decoder_blocks.append(decoder_block_1)

        decoder_block_2 = TtPointPillarsConvTranspose2DSplit(
            conv_transpose=parameters[f"decoder_2"]["conv_args"],
            conv_transpose_pth=parameters[f"decoder_2"],
            device=device,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=deallocate_activation,
            conv_in_channel_split_factor=4,
            conv_out_channel_split_factor=2,
        )

        self.decoder_blocks.append(decoder_block_2)

    def forward(self, x):
        """
        x: list of ttnn tensors [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: ttnn tensor (bs, 384, 248, 216)
        """
        outs = []
        # for i in range(len(self.decoder_blocks)):
        x0 = self.decoder_blocks[0](x[0])
        outs.append(x0)

        # Block 1
        x1 = self.decoder_blocks[1](x[1])

        x1 = ttnn.reshape(x1, outs[0].shape)
        x1 = ttnn.permute(x1, (0, 3, 1, 2))

        x1 = ttnn.batch_norm(
            x1,
            running_mean=self.parameters[f"decoder_1"]["bn_running_mean"],
            running_var=self.parameters[f"decoder_1"]["bn_running_var"],
            training=False,
            eps=1e-05,
            weight=self.parameters[f"decoder_1"]["bn_weight"],
            bias=self.parameters[f"decoder_1"]["bn_bias"],
            compute_kernel_config=self.bn_config,
        )

        x1 = ttnn.relu(x1)
        x1 = ttnn.permute(x1, (0, 2, 3, 1))

        outs.append(x1)

        # Block 2
        x2 = self.decoder_blocks[2](x[2])

        x2 = ttnn.reshape(x2, outs[0].shape)
        x2 = ttnn.permute(x2, (0, 3, 1, 2))

        x2 = ttnn.batch_norm(
            x2,
            running_mean=self.parameters[f"decoder_2"]["bn_running_mean"],
            running_var=self.parameters[f"decoder_2"]["bn_running_var"],
            training=False,
            eps=1e-05,
            weight=self.parameters[f"decoder_2"]["bn_weight"],
            bias=self.parameters[f"decoder_2"]["bn_bias"],
            compute_kernel_config=self.bn_config,
        )

        x2 = ttnn.relu(x2)
        x2 = ttnn.permute(x2, (0, 2, 3, 1))

        outs.append(x2)

        out = ttnn.concat(outs, dim=3)
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        return out
