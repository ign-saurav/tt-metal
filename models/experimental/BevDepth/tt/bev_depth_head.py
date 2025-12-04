import ttnn
import torch
from dataclasses import dataclass

from models.experimental.BevDepth.tt.utils import TTConv2D, TTConvTranspose2D

from models.experimental.BevDepth.tests.bev_depth_neck import SECONDFPN


@dataclass
class HeadOptimizer:
    deblock: dict
    conv1: dict
    conv2: dict


head_optimisations = HeadOptimizer(
    deblock={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "reshard_if_not_optimal": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
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


class TtBasicBlock:
    """TTNN version of BasicBlock - BatchNorm is folded into Conv2d during preprocessing"""

    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride, parameters, model_config, layer_optimisations, downsample=None
    ):
        self.stride = stride
        self.downsample = downsample

        # Conv1: BatchNorm is already folded into Conv2d during preprocessing
        conv1_params = parameters.get("conv1", {})
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            parameters=conv1_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv1,
        )

        # Conv2: BatchNorm is already folded into Conv2d during preprocessing
        conv2_params = parameters.get("conv2", {})
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=conv2_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv2,
        )

    def __call__(self, x, device):
        identity = x
        input_shape = x.shape

        # Conv1 + ReLU (BN already folded)
        out, output_shape = self.conv1(device, x, input_shape)
        out = out.reshape(output_shape)
        out = ttnn.relu(out)

        # Conv2 (BN already folded)
        out, output_shape = self.conv2(device, out, output_shape)
        out = out.reshape(output_shape)

        # Downsample if needed
        if self.downsample is not None:
            identity, identity_shape = self.downsample(identity, device)
            identity = identity.reshape(identity_shape)

        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, out.shape


class TtResLayer:
    """TTNN version of ResLayer - BatchNorm is folded into Conv2d during preprocessing"""

    def __init__(self, in_channels, out_channels, blocks, stride, parameters, model_config, layer_optimisations):
        self.blocks = []

        # Create first block with downsample
        first_block_params = parameters.get(0, {})

        # Create downsample if needed
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample_conv_params = first_block_params.get("downsample", {})  # Conv2d is first in Sequential
            downsample_conv = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                parameters=downsample_conv_params,
                kernel_fidelity=model_config,
            )

            # Wrap in a callable that matches the interface
            def downsample_fn(x, device):
                input_shape = x.shape
                return downsample_conv(device, x, input_shape)

            downsample = downsample_fn

        self.blocks.append(
            TtBasicBlock(
                in_channels,
                out_channels,
                stride,
                first_block_params,
                model_config,
                layer_optimisations,
                downsample=downsample,
            )
        )
        # Create remaining blocks
        for i in range(1, blocks):
            block_params = parameters.get(i, {})
            self.blocks.append(
                TtBasicBlock(
                    out_channels,
                    out_channels,
                    1,
                    block_params,
                    model_config,
                    layer_optimisations,
                    downsample=None,
                )
            )

    def __call__(self, x, device):
        for block in self.blocks:
            x, output_shape = block(x, device)  # copy()
            # x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) #run with bfloat8_b
        return x, output_shape


class TtResNet:
    """TTNN version of ResNet backbone - BatchNorm is folded into Conv2d during preprocessing"""

    def __init__(self, parameters, model_config, layer_optimisations=head_optimisations):
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations
        # print(parameters)

        # Conv1: BatchNorm is already folded into Conv2d during preprocessing
        conv1_params = parameters.get("conv1", {})
        self.conv1 = TTConv2D(
            kernel_size=7,
            stride=2,
            padding=3,
            parameters=conv1_params,
            kernel_fidelity=model_config,
            # **layer_optimisations.conv1,
        )

        # ResLayers
        layer1_params = parameters.get("layer1", {})
        self.layer1 = TtResLayer(
            160,
            160,
            blocks=2,
            stride=1,
            parameters=layer1_params,
            model_config=model_config,
            layer_optimisations=layer_optimisations,
        )

        layer2_params = parameters.get("layer2", {})
        self.layer2 = TtResLayer(
            160,
            320,
            blocks=2,
            stride=2,
            parameters=layer2_params,
            model_config=model_config,
            layer_optimisations=layer_optimisations,
        )

        layer3_params = parameters.get("layer3", {})
        print(f"Layer3 params: {layer3_params}")
        self.layer3 = TtResLayer(
            320,
            640,
            blocks=2,
            stride=2,
            parameters=layer3_params,
            model_config=model_config,
            layer_optimisations=layer_optimisations,
        )

    def __call__(self, x, device=None):
        if device is None:
            raise ValueError("Device must be provided in __call__")

        input_shape = x.shape
        # Conv1 + ReLU (BN already folded)
        # Output 0: After conv1+bn1+relu -> (B, 160, 128, 128) in NHWC format
        x0, output_shape0 = self.conv1(device, x, input_shape)
        x0 = ttnn.relu(x0)
        x0 = x0.reshape(output_shape0)

        # Layer1: (B, 160, 128, 128) -> (B, 160, 128, 128)
        # Output 1: After layer1
        x1, output_shape1 = self.layer1(x0, device)

        # Layer2: (B, 160, 128, 128) -> (B, 320, 64, 64) (downsampling)
        # Output 2: After layer2
        x2, output_shape2 = self.layer2(x1, device)

        # Layer3: (B, 320, 64, 64) -> (B, 640, 32, 32) (downsampling)
        # Output 3: After layer3
        x3, output_shape3 = self.layer3(x2, device)

        # Return 4 outputs: x0 (160ch after conv1), x1 (160ch), x2 (320ch), x3 (640ch)
        # This matches the reference neck config: in_channels=[160, 160, 320, 640]
        return (x, x1, x2, x3)


class TtDeblock:
    def __init__(self, in_channels, out_channels, kernel_size, stride, parameters, model_config, layer_optimisations):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize TTConvTranspose2D layer
        self.conv_transpose = TTConvTranspose2D(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            parameters=parameters,
            kernel_fidelity=model_config,
            # **layer_optimisations.deblock,
        )

    def __call__(self, x, device):
        # Input x should be in NHWC format (batch, height, width, channels)
        input_shape = x.shape

        # ConvTranspose2d + ReLU
        x, output_shape = self.conv_transpose(device, x, input_shape)
        x = ttnn.relu(x)

        return x, output_shape


class TtSECONDFPN:
    def __init__(self, parameters, model_config, layer_optimisations=head_optimisations):
        super().__init__()
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations

        # Initialize 4 deblocks with parameters
        # print(deblocks_params)
        self.deblocks = [
            TtDeblock(
                in_channels=160,
                out_channels=64,
                kernel_size=1,
                stride=1,
                parameters=parameters["deblock_0"],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=160,
                out_channels=64,
                kernel_size=2,
                stride=2,
                parameters=parameters["deblock_1"],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=320,
                out_channels=64,
                kernel_size=4,
                stride=4,
                parameters=parameters["deblock_2"],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
            TtDeblock(
                in_channels=640,
                out_channels=64,
                kernel_size=8,
                stride=8,
                parameters=parameters["deblock_3"],
                model_config=model_config,
                layer_optimisations=layer_optimisations,
            ),
        ]

    def __call__(self, x0, x1, x2, x3, device=None):
        if device is None:
            raise ValueError("Device must be provided in __call__")

        # Process each input through its corresponding deblock
        y0, _ = self.deblocks[0](x0, device)
        y1, _ = self.deblocks[1](x1, device)
        y2, _ = self.deblocks[2](x2, device)
        y3, _ = self.deblocks[3](x3, device)

        # Concatenate along channel dimension (dim=3 in NHWC format)
        # All outputs should be (B, 128, 128, 64)
        y = ttnn.concat([y0, y1, y2, y3], dim=3)

        return y


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
    def __init__(self, parameters, model_config, layer_optimisations=head_optimisations, checkpoint_path=None):
        super().__init__()
        self.parameters = parameters
        self.model_config = model_config
        self.layer_optimisations = layer_optimisations

        self.trunk_params = parameters.get("trunk", {})
        self.trunk = TtResNet(self.trunk_params, model_config, layer_optimisations)

        # self.neck_params = parameters.get("neck", {})
        # self.neck = TtSECONDFPN(self.neck_params, model_config, layer_optimisations)
        # Load BEVDepthLightningModel and use head.neck from it
        self.neck = SECONDFPN()
        self.neck.load_checkpoint("../reference/checkpoints/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
        self.neck = self.neck.eval()
        # Initialize shared_conv as TTConv2D
        shared_conv_params = parameters.get("shared_conv", {})
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

        # Trunk returns (x0, x1, x2, x3) - 4 outputs
        # x0 is after conv1 (160ch), x1 is after layer1 (160ch), x2 is after layer2 (320ch), x3 is after layer3 (640ch)
        # This matches the reference neck config: in_channels=[160, 160, 320, 640]
        trunk_outputs = self.trunk(x, device)
        x0, x1, x2, x3 = trunk_outputs
        # print(f"Trunk outputs shapes: x0: {x0.shape}, x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}")
        # print(f"x0: {ttnn.to_torch(x0).permute(0, 3, 1, 2).reshape(-1)[:10]}")
        # print(f"x1: {ttnn.to_torch(x1).permute(0, 3, 1, 2).reshape(-1)[:10]}")
        # print(f"x2: {ttnn.to_torch(x2).permute(0, 3, 1, 2).reshape(-1)[:10]}")
        # print(f"x3: {ttnn.to_torch(x3).permute(0, 3, 1, 2).reshape(-1)[:10]}")

        # Convert TTNN tensors to PyTorch for reference neck
        # Reference neck expects NCHW format, TTNN uses NHWC
        x0_torch = ttnn.to_torch(x0)
        x1_torch = ttnn.to_torch(x1)
        x2_torch = ttnn.to_torch(x2)
        x3_torch = ttnn.to_torch(x3)

        # Convert from NHWC to NCHW
        x0_torch = x0_torch.permute(0, 3, 1, 2).float()
        x1_torch = x1_torch.permute(0, 3, 1, 2).float()
        x2_torch = x2_torch.permute(0, 3, 1, 2).float()
        x3_torch = x3_torch.permute(0, 3, 1, 2).float()
        print(f"x0_torch: {x0_torch.reshape(-1)[:10]}")
        print(f"x1_torch: {x1_torch.reshape(-1)[:10]}")
        print(f"x2_torch: {x2_torch.reshape(-1)[:10]}")
        print(f"x3_torch: {x3_torch.reshape(-1)[:10]}")

        # Reference neck expects list of tensors matching in_channels=[160, 160, 320, 640]
        neck_inputs = [x0_torch, x1_torch, x2_torch, x3_torch]
        with torch.no_grad():
            neck_output = self.neck(neck_inputs)

        # neck_output is a list with one tensor [out] in NCHW format
        x = neck_output
        print(f"Neck output: {x.shape}")
        print(f"Neck output: {x.reshape(-1)[:10]}")

        # Convert back to TTNN format (NHWC)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        x = ttnn.to_device(x, device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Shared conv: 256 -> 64 channels (64+64+128 = 256 from concatenation)
        x, output_shape = self.shared_conv(device, x, x.shape)
        x = ttnn.relu(x)
        # Reshape if needed
        if len(output_shape) == 4:
            batch_size, out_h, out_w, out_c = output_shape
            x = x.reshape(batch_size, out_h, out_w, out_c)

        return [head(x, device) for head in self.task_heads]
