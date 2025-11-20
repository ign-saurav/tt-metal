import ttnn


class TtBEVDepthHead:
    def __init__(self, device):
        self.device = device

        # Store preprocessed weights
        self.conv1_weight = None
        self.conv1_bias = None
        self.conv2_weight = None
        self.conv2_bias = None

    def __call__(self, x):
        # Input x should be in NHWC format (batch, height, width, channels)

        # First conv (64 -> 64, kernel=3, stride=1, padding=1) + ReLU
        batch_size = x.shape[0]
        x, [out_h, out_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight,
            bias_tensor=self.conv1_bias,
            in_channels=64,
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
        print("Out H, W:", out_h, out_w)
        x = ttnn.relu(x)

        # Second conv (64 -> 2, kernel=3, stride=1, padding=1)
        x, [out_h, out_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            in_channels=64,
            out_channels=2,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            return_output_dim=True,
        )

        return x, out_h, out_w
