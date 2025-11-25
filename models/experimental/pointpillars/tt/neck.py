import ttnn
from models.experimental.pointpillars.tt.utils import TtPointPillarsConvTranspose2D


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

        # Initialize decoder blocks
        self.decoder_blocks = []

        for i in range(len(in_channels)):
            decoder_block = TtPointPillarsConvTranspose2D(
                conv_transpose=parameters["neck"][f"decoder_{i}"]["conv_args"],
                conv_transpose_pth=parameters["neck"][f"decoder_{i}"],
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
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])
            xi = ttnn.to_memory_config(xi, ttnn.DRAM_MEMORY_CONFIG)
            outs.append(xi)

        # Concatenate along channel dimension (dim=3 in NHWC format)
        out = ttnn.concat(outs, dim=3)
        return out
