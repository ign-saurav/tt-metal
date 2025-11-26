import ttnn
from dataclasses import dataclass
from models.experimental.BevDepth.tt.utils import TTConv2D


@dataclass
class HeadOptimizer:
    conv1: dict
    conv2: dict


head_optimisations = HeadOptimizer(
    conv1={
        # "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "reshard_if_not_optimal": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2),
        "dtype": ttnn.bfloat16,
    },
    conv2={
        "act_block_h": 512,
        # "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "reshard_if_not_optimal": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
)


class TtTaskHead:
    def __init__(self, in_channels, out_channels, parameters, model_config, layer_optimisations):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize TTConv2D layers
        # Conv1: in_channels -> in_channels
        conv1_params = parameters.get(0, {})
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=conv1_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv1,
        )

        # Conv2: in_channels -> out_channels
        conv2_params = parameters.get(1, {})
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=conv2_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv2,
        )

    def __call__(self, x, device):
        # Input x should be in NHWC format (batch, height, width, channels)
        input_shape = x.shape

        # First conv (64 -> 64, kernel=3, stride=1, padding=1) + ReLU
        x, output_shape = self.conv1(device, x, input_shape)

        x = ttnn.relu(x)

        # Second conv (64 -> out_channels, kernel=3, stride=1, padding=1)
        x, output_shape = self.conv2(device, x, output_shape)

        return x, output_shape


class TtSeparateHead:
    def __init__(self, in_channels, heatmap_out, parameters, model_config, layer_optimisations):
        super().__init__()

        # Initialize task heads with parameters
        self.reg = TtTaskHead(in_channels, 2, parameters.get("reg"), model_config, layer_optimisations)
        self.height = TtTaskHead(in_channels, 1, parameters.get("height"), model_config, layer_optimisations)
        self.dim = TtTaskHead(in_channels, 3, parameters.get("dim"), model_config, layer_optimisations)
        self.rot = TtTaskHead(in_channels, 2, parameters.get("rot"), model_config, layer_optimisations)
        self.vel = TtTaskHead(in_channels, 2, parameters.get("vel"), model_config, layer_optimisations)
        self.heatmap = TtTaskHead(
            in_channels, heatmap_out, parameters.get("heatmap"), model_config, layer_optimisations
        )

    def __call__(self, x, device):
        return {
            "reg": self.reg(x, device),
            "height": self.height(x, device),
            "dim": self.dim(x, device),
            "rot": self.rot(x, device),
            "vel": self.vel(x, device),
            "heatmap": self.heatmap(x, device),
        }


class TtBEVDepthHead:
    def __init__(self, parameters, model_config, layer_optimisations=head_optimisations):
        super().__init__()
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations

        # Initialize shared_conv as TTConv2D
        shared_conv_params = parameters.get("shared_conv", {})
        # Use conv1 optimisations for shared_conv (similar structure)
        self.shared_conv = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=shared_conv_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv1,
        )

        heatmap_channels = [1, 2, 2, 1, 2, 2]

        # Initialize task heads with parameters
        task_heads_params = parameters.get("task_heads", [])
        self.task_heads = [
            TtSeparateHead(
                64,
                heatmap_out=heatmap_channels[i],
                parameters=task_heads_params[i],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            )
            for i in range(6)
        ]

    def __call__(self, x, device=None):
        if device is None:
            raise ValueError("Device must be provided in __call__")

        input_shape = x.shape

        # Shared conv: 256 -> 64 channels
        x, output_shape = self.shared_conv(device, x, input_shape)
        x = ttnn.relu(x)
        # Reshape if needed
        if len(output_shape) == 4:
            batch_size, out_h, out_w, out_c = output_shape
            x = x.reshape(batch_size, out_h, out_w, out_c)

        return [head(x, device) for head in self.task_heads]
