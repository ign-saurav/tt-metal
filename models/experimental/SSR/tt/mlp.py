import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.utils.config_helpers import matmul_config


class TTMlp(LightweightModule):
    def __init__(self, device, in_features, hidden_features=None, out_features=None, parameters=None):
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
        # First linear layer
        program_config = matmul_config(
            x.shape[-2], x.shape[-1], self.fc1_bias.shape[-1], (8, 8), fused_activation=(ttnn.UnaryOpType.GELU, True)
        )
        x = ttnn.linear(
            x, self.fc1_weight, bias=self.fc1_bias, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=program_config
        )

        program_config = matmul_config(x.shape[-2], x.shape[-1], self.fc2_bias.shape[-1], (8, 8))
        # Second linear layer
        x = ttnn.linear(
            x, self.fc2_weight, bias=self.fc2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=program_config
        )

        return x
