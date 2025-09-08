import ttnn
from models.common.lightweightmodule import LightweightModule


class TTMlp(LightweightModule):
    def __init__(self, device, memory_config, in_features, hidden_features=None, out_features=None, parameters=None):
        self.device = device

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Initialize weights and biases based on available inputs
        # Use preprocessed parameters
        self.fc1_weight = parameters["fc1"]["weight"]
        self.fc1_bias = parameters["fc1"]["bias"]
        self.fc2_weight = parameters["fc2"]["weight"]
        self.fc2_bias = parameters["fc2"]["bias"]

    def forward(self, x):
        # Debug prints for forward arguments
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # First linear layer
        # program_config = matmul_config(
        #     x.shape[-2], x.shape[-1], self.fc1_bias.shape[-1], (8, 8), fused_activation=(ttnn.UnaryOpType.GELU, True)
        # )
        x = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="gelu",
        )

        # program_config = matmul_config(x.shape[-2], x.shape[-1], self.fc2_bias.shape[-1], (8, 8))
        # Second linear layer
        x = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        return x
