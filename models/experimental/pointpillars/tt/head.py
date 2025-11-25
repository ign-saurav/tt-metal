import ttnn
from models.experimental.pointpillars.tt.utils import TtPointPillarsConv2D


class TtHead:
    def __init__(
        self,
        in_channel,
        n_anchors,
        n_classes,
        parameters,
        device,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.device = device
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        # Initialize three parallel 1x1 convolution branches
        self.conv_cls = TtPointPillarsConv2D(
            conv=parameters["conv_cls"]["conv_args"],
            conv_pth=parameters["conv_cls"],
            device=device,
            activation=None,  # No activation for detection heads
            activation_dtype=dtype,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=False,
            math_fidelity=math_fidelity,
        )

        self.conv_reg = TtPointPillarsConv2D(
            conv=parameters["conv_reg"]["conv_args"],
            conv_pth=parameters["conv_reg"],
            device=device,
            activation=None,
            activation_dtype=dtype,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=False,
            math_fidelity=math_fidelity,
        )

        self.conv_dir_cls = TtPointPillarsConv2D(
            conv=parameters["conv_dir_cls"]["conv_args"],
            conv_pth=parameters["conv_dir_cls"],
            device=device,
            activation=None,
            activation_dtype=dtype,
            weights_dtype=dtype,
            shard_layout=shard_layout,
            is_dealloc_act=deallocate_activation,
            math_fidelity=math_fidelity,
        )

    def forward(self, x):
        """
        x: ttnn tensor (bs, 384, 248, 216) in NHWC format
        return: tuple of ttnn tensors
            bbox_cls_pred: (bs, n_anchors*n_classes, 248, 216)
            bbox_pred: (bs, n_anchors*7, 248, 216)
            bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)

        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
