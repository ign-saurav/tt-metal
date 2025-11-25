import ttnn
from models.experimental.pointpillars.tt.utils import TtPointPillarsConv2D


class TtBackbone:
    def __init__(
        self,
        in_channel,
        out_channels,
        layer_nums,
        layer_strides,
        parameters,
        device,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.out_channels = out_channels
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides

        # Initialize multi_blocks as a list of conv layers
        self.multi_blocks = []
        for i in range(len(layer_strides)):
            block_convs = []

            # First conv in each block (with stride)
            block_convs.append(
                TtPointPillarsConv2D(
                    conv=parameters[f"block_{i}"]["conv_0"]["conv_args"],
                    conv_pth=parameters[f"block_{i}"]["conv_0"],
                    device=device,
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    activation_dtype=dtype,
                    weights_dtype=dtype,
                    shard_layout=shard_layout,
                    # is_dealloc_act=deallocate_activation,
                    is_dealloc_act=True,
                    reshape_output=True,
                    math_fidelity=ttnn.MathFidelity.HiFi4 if i == 2 else ttnn.MathFidelity.HiFi2,
                    # memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )

            # Subsequent convs in the block (no stride)
            for j in range(layer_nums[i]):
                block_convs.append(
                    TtPointPillarsConv2D(
                        conv=parameters[f"block_{i}"][f"conv_{j+1}"]["conv_args"],
                        conv_pth=parameters[f"block_{i}"][f"conv_{j+1}"],
                        device=device,
                        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                        activation_dtype=dtype,
                        weights_dtype=dtype,
                        shard_layout=shard_layout,
                        # is_dealloc_act=deallocate_activation,
                        is_dealloc_act=False if (j == layer_nums[i] - 1) else True,
                        reshape_output=True,
                        math_fidelity=ttnn.MathFidelity.HiFi4 if i == 2 else ttnn.MathFidelity.HiFi2,
                        # memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                )

            self.multi_blocks.append(block_convs)

    def forward(self, x):
        """
        x: ttnn tensor (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for block_convs in self.multi_blocks:
            for conv in block_convs:
                x = conv(x)
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            outs.append(x)
        return outs
