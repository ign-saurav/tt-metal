import ttnn
import torch


class TtRNNTJoint:
    """TTNN implementation of RNN-T Joint Network.

    Matches the PyTorch structure:
    - pred: Linear(in_features=640, out_features=640, bias=True)
    - enc: Linear(in_features=1024, out_features=640, bias=True)
    - joint_net: Sequential(ReLU, Linear(640, 1030))
    """

    def __init__(
        self,
        device,
        encoder_hidden: int = 1024,
        pred_hidden: int = 640,
        joint_hidden: int = 640,
        num_classes: int = 1029,
        dtype=ttnn.float32,
        memory_config=None,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.encoder_hidden = encoder_hidden
        self.pred_hidden = pred_hidden
        self.joint_hidden = joint_hidden
        self.num_classes = num_classes

        # Initialize weights from PyTorch-style initialization
        self._init_weights(encoder_hidden, pred_hidden, joint_hidden, num_classes)

        # Create linear layers
        self.pred = self._create_linear(pred_hidden, joint_hidden)
        self.enc = self._create_linear(encoder_hidden, joint_hidden)
        self.joint_net = self._create_joint_net(joint_hidden, num_classes)

    def _init_weights(self, encoder_hidden, pred_hidden, joint_hidden, num_classes):
        """Initialize weights following PyTorch Linear layer initialization."""
        # pred layer weights: (640, 640)
        self.pred_weight = torch.randn(pred_hidden, joint_hidden) * (1.0 / pred_hidden**0.5)
        self.pred_bias = torch.zeros(joint_hidden)

        # enc layer weights: (1024, 640)
        self.enc_weight = torch.randn(encoder_hidden, joint_hidden) * (1.0 / encoder_hidden**0.5)
        self.enc_bias = torch.zeros(joint_hidden)

        # joint_net final layer weights: (640, 1030)
        self.joint_weight = torch.randn(joint_hidden, num_classes) * (1.0 / joint_hidden**0.5)
        self.joint_bias = torch.zeros(num_classes)

    def _create_linear(self, in_features: int, out_features: int):
        """Create a linear layer with weights and bias."""
        # Select the correct weight based on in_features
        if in_features == self.pred_hidden:
            weight_tensor = self.pred_weight
            bias_tensor = self.pred_bias
        elif in_features == self.encoder_hidden:
            weight_tensor = self.enc_weight
            bias_tensor = self.enc_bias
        else:
            raise ValueError(f"Unknown in_features: {in_features}")

        weight = ttnn.from_torch(
            weight_tensor,
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        bias = ttnn.from_torch(
            bias_tensor, dtype=self.dtype, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config
        )
        return {"weight": weight, "bias": bias, "in_features": in_features, "out_features": out_features}

    def _create_joint_net(self, joint_hidden: int, num_classes: int):
        """Create the joint network: ReLU -> Linear."""
        weight = ttnn.from_torch(
            self.joint_weight,
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        bias = ttnn.from_torch(
            self.joint_bias,
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        return {"weight": weight, "bias": bias, "in_features": joint_hidden, "out_features": num_classes}

    def linear(self, x: ttnn.Tensor, layer: dict) -> ttnn.Tensor:
        """Implement linear layer: y = x @ W + b."""
        # x: [B, ..., in_features]
        # weight: [in_features, out_features]
        # bias: [out_features]

        # Matrix multiplication
        output = ttnn.matmul(x, layer["weight"])

        # Add bias
        bias = layer["bias"]
        if len(output.shape) == 2:
            output = output + bias
        else:
            # For higher dimensions, broadcast bias
            bias_shape = [1] * (len(output.shape) - 1) + [layer["out_features"]]
            bias_broadcast = ttnn.reshape(bias, bias_shape)
            output = output + bias_broadcast

        return output

    def project_encoder(self, encoder_output: ttnn.Tensor) -> ttnn.Tensor:
        """Project encoder output to joint hidden dimension.

        Args:
            encoder_output: [B, T, 1024]

        Returns:
            [B, T, 640]
        """
        return ttnn.linear(encoder_output, self.enc["weight"], bias=self.enc["bias"], memory_config=self.memory_config)

    def project_prednet(self, prednet_output: ttnn.Tensor) -> ttnn.Tensor:
        """Project prediction network output to joint hidden dimension.

        Args:
            prednet_output: [B, U, 640]

        Returns:
            [B, U, 640]
        """
        return ttnn.linear(
            prednet_output, self.pred["weight"], bias=self.pred["bias"], memory_config=self.memory_config
        )

    def joint_after_projection(self, f: ttnn.Tensor, g: ttnn.Tensor) -> ttnn.Tensor:
        """Compute joint step after projection.

        Args:
            f: [B, T, 640] - encoder projection
            g: [B, U, 640] - prediction network projection

        Returns:
            [B, T, U, 1030] - joint logits
        """
        # Expand dimensions for broadcasting
        # f: [B, T, 640] -> [B, T, 1, 640]
        f_expanded = ttnn.unsqueeze(f, dim=2)

        # g: [B, U, 640] -> [B, 1, U, 640]
        g_expanded = ttnn.unsqueeze(g, dim=1)

        # Sum: [B, T, U, 640]
        joint_hidden = ttnn.add(f_expanded, g_expanded)

        # Apply ReLU
        activated = ttnn.relu(joint_hidden)

        # Final linear layer: [B, T, U, 1030]
        output = ttnn.linear(
            activated, self.joint_net["weight"], bias=self.joint_net["bias"], memory_config=self.memory_config
        )

        return output

    def forward(self, encoder_outputs: ttnn.Tensor, decoder_outputs: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass of RNN-T Joint network.

        Args:
            encoder_outputs: [B, T, 1024] - encoder features
            decoder_outputs: [B, U, 640] - prediction network features

        Returns:
            [B, T, U, 1030] - joint logits
        """
        # Project encoder and prediction network outputs
        f = self.project_encoder(encoder_outputs)  # [B, T, 640]
        g = self.project_prednet(decoder_outputs)  # [B, U, 640]

        # Compute joint
        joint_output = self.joint_after_projection(f, g)  # [B, T, U, 1030]

        return joint_output
