import ttnn
import torch
from models.experimental.BevDepth.tt.head_preprocessing import load_task_head_weights, fold_batch_norm2d_into_conv2d


class TtTaskHead:
    def __init__(self, device, in_channels, out_channels):
        self.device = device

        # Store preprocessed weights
        self.conv1_weight = None
        self.conv1_bias = None
        self.conv2_weight = None
        self.conv2_bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, x):
        # Input x should be in NHWC format (batch, height, width, channels)

        # First conv (64 -> 64, kernel=3, stride=1, padding=1) + ReLU
        batch_size = x.shape[0]
        x, [out_h, out_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight,
            bias_tensor=self.conv1_bias,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=x.shape[1],
            input_width=x.shape[2],
            return_output_dim=True,
        )
        x = ttnn.relu(x)

        # Second conv (64 -> 2, kernel=3, stride=1, padding=1)
        x, [out_h, out_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            return_output_dim=True,
        )

        return x, (batch_size, out_h, out_w, self.out_channels)


class TtSeparateHead:
    def __init__(self, device, in_channels, heatmap_out):
        super().__init__()

        self.reg = TtTaskHead(device, in_channels, 2)
        self.height = TtTaskHead(device, in_channels, 1)
        self.dim = TtTaskHead(device, in_channels, 3)
        self.rot = TtTaskHead(device, in_channels, 2)
        self.vel = TtTaskHead(device, in_channels, 2)
        self.heatmap = TtTaskHead(device, in_channels, heatmap_out)

    def __call__(self, x):
        return {
            "reg": self.reg(x),
            "height": self.height(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
            "vel": self.vel(x),
            "heatmap": self.heatmap(x),
        }


class TtBEVDepthHead:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.shared_conv_weight = None
        self.shared_conv_bias = None

        heatmap_channels = [1, 2, 2, 1, 2, 2]

        self.task_heads = [TtSeparateHead(self.device, 64, heatmap_out=heatmap_channels[i]) for i in range(6)]

    def __call__(self, x):
        batch_size = x.shape[0]
        x, [out_h, out_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.shared_conv_weight,
            bias_tensor=self.shared_conv_bias,
            in_channels=256,
            out_channels=64,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=x.shape[1],
            input_width=x.shape[2],
            return_output_dim=True,
        )
        x = ttnn.relu(x)
        x = x.reshape(batch_size, out_h, out_w, 64)
        return [head(x) for head in self.task_heads]

    def load_checkpoint(self, weight_path):
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        shared_conv_prefix = "model.head.shared_conv."
        conv_w = state[shared_conv_prefix + "conv.weight"]
        bn_w = state[shared_conv_prefix + "bn.weight"]
        bn_b = state[shared_conv_prefix + "bn.bias"]
        bn_rm = state[shared_conv_prefix + "bn.running_mean"]
        bn_rv = state[shared_conv_prefix + "bn.running_var"]

        shared_conv_weight, shared_conv_bias = fold_batch_norm2d_into_conv2d(conv_w, bn_w, bn_b, bn_rm, bn_rv)
        self.shared_conv_weight = ttnn.from_torch(shared_conv_weight)
        self.shared_conv_bias = ttnn.from_torch(shared_conv_bias.reshape(1, 1, 1, -1))

        for head_id in range(6):
            for task_name in ["reg", "height", "dim", "rot", "vel", "heatmap"]:
                key_prefix = f"model.head.task_heads.{head_id}.{task_name}."

                # extract only tensors belonging to this head/task
                task_tensors = {k: v for k, v in state.items() if k.startswith(key_prefix)}

                # store them in the correct TTNN head block
                ttnn_task_head = getattr(self.task_heads[head_id], task_name)

                load_task_head_weights(
                    ttnn_task_head,
                    task_tensors,
                    key_prefix,
                )

                print(f"Loaded weights for head {head_id} task {task_name}")
