import ttnn
from models.experimental.pointpillars.tt.utils import TtPointPillarsConvTranspose2D
from models.experimental.pointpillars.tt.utils import (
    prepare_split_conv_transpose2d_weights_bias,
    split_conv_transpose2d_and_run,
)


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

        # for i in range(len(in_channels)):
        decoder_block = TtPointPillarsConvTranspose2D(
            conv_transpose=parameters[f"decoder_0"]["conv_args"],
            conv_transpose_pth=parameters[f"decoder_0"],
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=dtype,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=deallocate_activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            reshape_output=True,
        )
        self.decoder_blocks.append(decoder_block)

    def forward(self, x):
        """
        x: list of ttnn tensors [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: ttnn tensor (bs, 384, 248, 216)
        """
        outs = []
        # for i in range(len(self.decoder_blocks)):
        xi = self.decoder_blocks[0](x[0])
        xi = ttnn.to_memory_config(xi, ttnn.DRAM_MEMORY_CONFIG)
        outs.append(xi)

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=None,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Use HiFi instead of LoFi
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Prepare split weights and bias
        conv_weights, conv_bias = prepare_split_conv_transpose2d_weights_bias(
            in_channels=128,
            out_channels=128,
            conv_in_channel_split_factor=2,  # Split 128 into 2x64
            conv_out_channel_split_factor=2,  # Split 128 into 2x64
            torch_weight_tensor=self.parameters[f"decoder_1"]["weight"],
            torch_bias_tensor=self.parameters[f"decoder_1"]["bias"],
        )
        # Run split convolution
        output1 = split_conv_transpose2d_and_run(
            hidden_states=x[1],
            conv_weight=conv_weights,  # Pre-split weights [2][4]
            conv_bias=conv_bias,  # Pre-split bias [2]
            device=self.device,
            in_channels=128,
            input_height=124,
            input_width=108,
            out_channels=128,
            conv_in_channel_split_factor=2,
            conv_out_channel_split_factor=2,
            compute_config=compute_config,
            conv_config=conv_config,
            conv_output_dtype=ttnn.bfloat16,
            kernel_size=2,
            padding=0,
            output_padding=0,
            stride=2,
        )

        # Prepare split weights and bias
        conv_weights, conv_bias = prepare_split_conv_transpose2d_weights_bias(
            in_channels=256,
            out_channels=128,
            conv_in_channel_split_factor=4,  # Split 256 into 4x64
            conv_out_channel_split_factor=2,  # Split 128 into 2x64
            torch_weight_tensor=self.parameters[f"decoder_2"]["weight"],
            torch_bias_tensor=self.parameters[f"decoder_2"]["bias"],
        )

        output2 = split_conv_transpose2d_and_run(
            hidden_states=x[2],
            conv_weight=conv_weights,  # Pre-split weights [2][4]
            conv_bias=conv_bias,  # Pre-split bias [2]
            device=self.device,
            in_channels=256,
            input_height=62,
            input_width=54,
            out_channels=128,
            conv_in_channel_split_factor=4,
            conv_out_channel_split_factor=2,
            compute_config=compute_config,
            conv_config=conv_config,
            conv_output_dtype=ttnn.bfloat16,
            kernel_size=4,
            padding=0,
            output_padding=0,
            stride=4,
        )
        outs.append(ttnn.reshape(output1, outs[0].shape))
        outs.append(ttnn.reshape(output2, outs[0].shape))

        # # Concatenate along channel dimension (dim=3 in NHWC format)
        out = ttnn.concat(outs, dim=3)
        return out
