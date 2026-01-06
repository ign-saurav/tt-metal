import os
import ttnn
import torch
import numpy as np
from dataclasses import dataclass

from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.BevDepth.tt.utils import (
    create_conv2d_config,
    post_process_conv_output,
)
from models.experimental.BevDepth.tt.ttnn_secondfpn import SECONDFPN_Head_TTNN
from models.experimental.BevDepth.tt.custom_preprocessing import prepare_secondfpn_head_parameters


@dataclass
class HeadOptimizations:
    conv_transpose: dict
    conv2d: dict


head_optimizations = HeadOptimizations(
    conv_transpose={
        "deallocate_activation": False,
    },
    conv2d={
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)

head_optimisations = head_optimizations


class TtBasicBlock:
    expansion = 1

    def __init__(self, device, in_channels, out_channels, stride, parameters, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.model_config = model_config

        # Store weights from parameters
        conv1_params = parameters.get("conv1", {})
        self.conv1_weight = conv1_params.get("weight")
        self.conv1_bias = conv1_params.get("bias")

        conv2_params = parameters.get("conv2", {})
        self.conv2_weight = conv2_params.get("weight")
        self.conv2_bias = conv2_params.get("bias")

        # Downsample params (if present)
        self.has_downsample = "downsample" in parameters
        if self.has_downsample:
            ds_params = parameters.get("downsample", {})
            self.downsample_weight = ds_params.get("weight")
            self.downsample_bias = ds_params.get("bias")

        # Conv caches
        self._conv1_cache = {}
        self._conv2_cache = {}
        self._downsample_cache = {}

    def _get_conv1(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                stride=(self.stride, self.stride),
                padding=(1, 1),
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._conv1_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv1_cache[cache_key]

    def _get_conv2(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv2_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.conv2_weight,
                bias=self.conv2_bias,
                stride=(1, 1),
                padding=(1, 1),
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._conv2_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv2_cache[cache_key]

    def _get_downsample(self, batch_size, height, width):
        if not self.has_downsample:
            return None
        cache_key = (batch_size, height, width)
        if cache_key not in self._downsample_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                weight=self.downsample_weight,
                bias=self.downsample_bias,
                stride=(self.stride, self.stride),
                padding=(0, 0),
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._downsample_cache[cache_key] = TtConv2d(config, self.device)
        return self._downsample_cache[cache_key]

    def __call__(self, x, batch_size, height, width):
        identity = x

        # Conv1 + ReLU
        conv1 = self._get_conv1(batch_size, height, width)
        out, (out_h, out_w) = conv1(x, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, out_h, out_w, self.out_channels)

        # Conv2 (no activation)
        conv2 = self._get_conv2(batch_size, out_h, out_w)
        out, (out_h2, out_w2) = conv2(out, return_output_dim=True)
        out = post_process_conv_output(out, batch_size, out_h2, out_w2, self.out_channels)

        # Downsample identity if needed
        if self.has_downsample:
            downsample = self._get_downsample(batch_size, height, width)
            identity, (id_h, id_w) = downsample(identity, return_output_dim=True)
            identity = post_process_conv_output(identity, batch_size, id_h, id_w, self.out_channels)

        # Residual connection
        if identity.is_sharded():
            identity = ttnn.sharded_to_interleaved(identity, ttnn.DRAM_MEMORY_CONFIG)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.add(out, identity)
        out = ttnn.relu(out)

        return out, (out_h2, out_w2)


class TtResLayer:
    def __init__(self, device, in_channels, out_channels, blocks, stride, parameters, model_config):
        self.device = device
        self.blocks = []

        # Create first block (may have downsample)
        first_block_params = parameters.get(0, {})
        needs_downsample = stride != 1 or in_channels != out_channels
        if needs_downsample:
            first_block_params["downsample"] = first_block_params.get("downsample", {})

        self.blocks.append(
            TtBasicBlock(
                device=device,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                parameters=first_block_params,
                model_config=model_config,
            )
        )

        # Create remaining blocks (no downsample, stride=1)
        for i in range(1, blocks):
            block_params = parameters.get(i, {})
            self.blocks.append(
                TtBasicBlock(
                    device=device,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    parameters=block_params,
                    model_config=model_config,
                )
            )

    def __call__(self, x, batch_size, height, width):
        h, w = height, width
        for block in self.blocks:
            x, (h, w) = block(x, batch_size, h, w)
        return x, (h, w)


class TtResNet:
    def __init__(self, device, parameters, model_config):
        self.device = device
        self.model_config = model_config

        # Conv1 parameters
        conv1_params = parameters.get("conv1", {})
        self.conv1_weight = conv1_params.get("weight")
        self.conv1_bias = conv1_params.get("bias")
        self.conv1_out_channels = self.conv1_weight.shape[0] if self.conv1_weight is not None else 160

        self._conv1_cache = {}

        layer1_params = parameters.get("layer1", {})
        self.layer1 = TtResLayer(
            device=device,
            in_channels=160,
            out_channels=160,
            blocks=2,
            stride=1,
            parameters=layer1_params,
            model_config=model_config,
        )

        layer2_params = parameters.get("layer2", {})
        self.layer2 = TtResLayer(
            device=device,
            in_channels=160,
            out_channels=320,
            blocks=2,
            stride=2,
            parameters=layer2_params,
            model_config=model_config,
        )

        layer3_params = parameters.get("layer3", {})
        self.layer3 = TtResLayer(
            device=device,
            in_channels=320,
            out_channels=640,
            blocks=2,
            stride=2,
            parameters=layer3_params,
            model_config=model_config,
        )

    def _get_conv1(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            in_channels = self.conv1_weight.shape[1] if self.conv1_weight is not None else 160
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=in_channels,
                out_channels=self.conv1_out_channels,
                batch_size=batch_size,
                kernel_size=(7, 7),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                stride=(2, 2),
                padding=(3, 3),
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._conv1_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv1_cache[cache_key]

    def __call__(self, x, batch_size=1):
        height, width = x.shape[1], x.shape[2]

        # Conv1 + ReLU - produces feature map at half resolution
        conv1 = self._get_conv1(batch_size, height, width)
        x_conv1, (h0, w0) = conv1(x, return_output_dim=True)
        x_conv1 = post_process_conv_output(x_conv1, batch_size, h0, w0, self.conv1_out_channels)

        # Layer1 (same resolution as conv1 output)
        x1, (h1, w1) = self.layer1(x_conv1, batch_size, h0, w0)

        # Layer2 (half resolution)
        x2, (h2, w2) = self.layer2(x1, batch_size, h1, w1)

        # Layer3 (quarter resolution)
        x3, (h3, w3) = self.layer3(x2, batch_size, h2, w2)

        return (x, x1, x2, x3)


class TtDeblock:
    def __init__(self, device, in_channels, out_channels, kernel_size, stride, parameters, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Prepare weights
        weight = parameters.get("weight")
        bias = parameters.get("bias")

        if isinstance(weight, torch.Tensor):
            weight = weight.float()
        elif isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float()
        elif isinstance(weight, ttnn.Tensor):
            weight = ttnn.to_torch(weight).float()

        if isinstance(bias, torch.Tensor):
            bias = bias.float()
        elif isinstance(bias, np.ndarray):
            bias = torch.from_numpy(bias).float()
        elif isinstance(bias, ttnn.Tensor):
            bias = ttnn.to_torch(bias).float()

        self._weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        if bias is not None:
            if len(bias.shape) == 1:
                bias = bias.view(1, 1, 1, -1)
            self._bias_ttnn = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            self._bias_ttnn = None

    def _create_conv_transpose_config(self):
        return ttnn.Conv2dConfig(
            weights_dtype=self.model_config.get("WEIGHTS_DTYPE", ttnn.bfloat16),
            shard_layout=None,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
        )

    def _create_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, batch_size, height, width):
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        conv_config = self._create_conv_transpose_config()
        compute_config = self._create_compute_config()

        out, [out_h, out_w] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self._weight_ttnn,
            bias_tensor=self._bias_ttnn,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
        )

        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.relu(out)

        out = post_process_conv_output(out, batch_size, out_h, out_w, self.out_channels)

        return out, (out_h, out_w)


class TtSECONDFPN:
    def __init__(self, device, parameters, model_config):
        self.device = device
        self.model_config = model_config

        # Initialize 4 deblocks
        self.deblocks = [
            TtDeblock(
                device=device,
                in_channels=160,
                out_channels=64,
                kernel_size=1,
                stride=1,
                parameters=parameters["deblock_0"],
                model_config=model_config,
            ),
            TtDeblock(
                device=device,
                in_channels=160,
                out_channels=64,
                kernel_size=2,
                stride=2,
                parameters=parameters["deblock_1"],
                model_config=model_config,
            ),
            TtDeblock(
                device=device,
                in_channels=320,
                out_channels=64,
                kernel_size=4,
                stride=4,
                parameters=parameters["deblock_2"],
                model_config=model_config,
            ),
            TtDeblock(
                device=device,
                in_channels=640,
                out_channels=64,
                kernel_size=8,
                stride=8,
                parameters=parameters["deblock_3"],
                model_config=model_config,
            ),
        ]

    def __call__(self, x0, x1, x2, x3, batch_size=1):
        # Get dimensions from inputs
        h0, w0 = x0.shape[1], x0.shape[2]
        h1, w1 = x1.shape[1], x1.shape[2]
        h2, w2 = x2.shape[1], x2.shape[2]
        h3, w3 = x3.shape[1], x3.shape[2]

        # Process each input through its deblock
        y0, _ = self.deblocks[0](x0, batch_size, h0, w0)
        y1, _ = self.deblocks[1](x1, batch_size, h1, w1)
        y2, _ = self.deblocks[2](x2, batch_size, h2, w2)
        y3, _ = self.deblocks[3](x3, batch_size, h3, w3)

        # Ensure all tensors are in interleaved DRAM for concat
        tensors = []
        for y in [y0, y1, y2, y3]:
            if y.is_sharded():
                y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
            tensors.append(y)

        # Concatenate along channel dimension (dim=3 in NHWC format)
        out = ttnn.concat(tensors, dim=3)

        return out


class TtTaskHead:
    def __init__(self, device, in_channels, out_channels, parameters, model_config):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config

        # Conv1 parameters
        conv1_params = parameters.get(0, {})
        self.conv1_weight = conv1_params.get("weight")
        self.conv1_bias = conv1_params.get("bias")

        # Conv2 parameters
        conv2_params = parameters.get(1, {})
        self.conv2_weight = conv2_params.get("weight")
        self.conv2_bias = conv2_params.get("bias")

        self._conv1_cache = {}
        self._conv2_cache = {}

    def _get_conv1(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv1_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                stride=(1, 1),
                padding=(1, 1),
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._conv1_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv1_cache[cache_key]

    def _get_conv2(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._conv2_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.conv2_weight,
                bias=self.conv2_bias,
                stride=(1, 1),
                padding=(1, 1),
                model_config=self.model_config,
                activation=None,
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._conv2_cache[cache_key] = TtConv2d(config, self.device)
        return self._conv2_cache[cache_key]

    def __call__(self, x, batch_size, height, width):
        # Conv1 + ReLU
        conv1 = self._get_conv1(batch_size, height, width)
        x, (out_h, out_w) = conv1(x, return_output_dim=True)
        x = post_process_conv_output(x, batch_size, out_h, out_w, self.in_channels)

        # Conv2 (no activation)
        conv2 = self._get_conv2(batch_size, out_h, out_w)
        x, (out_h2, out_w2) = conv2(x, return_output_dim=True)
        x = post_process_conv_output(x, batch_size, out_h2, out_w2, self.out_channels)

        return x, (out_h2, out_w2)


class TtSeparateHead:
    def __init__(self, device, in_channels, heatmap_out, parameters, model_config):
        self.device = device

        # Initialize task heads
        self.reg = TtTaskHead(device, in_channels, 2, parameters.get("reg"), model_config)
        self.height = TtTaskHead(device, in_channels, 1, parameters.get("height"), model_config)
        self.dim = TtTaskHead(device, in_channels, 3, parameters.get("dim"), model_config)
        self.rot = TtTaskHead(device, in_channels, 2, parameters.get("rot"), model_config)
        self.vel = TtTaskHead(device, in_channels, 2, parameters.get("vel"), model_config)
        self.heatmap = TtTaskHead(device, in_channels, heatmap_out, parameters.get("heatmap"), model_config)

    def __call__(self, x, batch_size, height, width):
        return {
            "reg": self.reg(x, batch_size, height, width),
            "height": self.height(x, batch_size, height, width),
            "dim": self.dim(x, batch_size, height, width),
            "rot": self.rot(x, batch_size, height, width),
            "vel": self.vel(x, batch_size, height, width),
            "heatmap": self.heatmap(x, batch_size, height, width),
        }


class TtBEVDepthHead:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=None,
        device=None,
        checkpoint_path=None,
    ):
        self.device = device
        self.model_config = model_config

        if device is None:
            raise ValueError("Device must be provided for TtBEVDepthHead")

        # Trunk
        trunk_params = parameters.get("trunk", {})
        self.trunk = TtResNet(device, trunk_params, model_config)

        # Neck
        if checkpoint_path is None:
            file_dir = os.path.dirname(__file__)
            for _ in range(4):
                file_dir = os.path.dirname(file_dir)
            default_path = os.path.join(
                file_dir,
                "models",
                "experimental",
                "BevDepth",
                "reference",
                "checkpoints",
                "bev_depth_lss_r50_256x704_128x128_24e_2key.pth",
            )

            # Check if default path exists, otherwise try downloaded weights
            if os.path.exists(default_path):
                checkpoint_path = default_path
            else:
                # Fallback to downloaded weights location
                downloaded_path = "/tmp/bevdepth_weights.pth"
                if os.path.exists(downloaded_path):
                    checkpoint_path = downloaded_path
                else:
                    raise FileNotFoundError(f"Checkpoint file not found. Tried: {default_path} and {downloaded_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        neck_params = prepare_secondfpn_head_parameters(state_dict)
        self.neck = SECONDFPN_Head_TTNN(
            device=device,
            parameters=neck_params,
            in_channels=[160, 160, 320, 640],
            out_channels=[64, 64, 64, 64],
            upsample_strides=[1, 2, 4, 8],
            model_config=model_config,
        )

        # Shared conv
        shared_conv_params = parameters.get("shared_conv", {})
        self.shared_conv_weight = shared_conv_params.get("weight")
        self.shared_conv_bias = shared_conv_params.get("bias")
        self.shared_conv_in_channels = self.shared_conv_weight.shape[1] if self.shared_conv_weight is not None else 256
        self.shared_conv_out_channels = self.shared_conv_weight.shape[0] if self.shared_conv_weight is not None else 64
        self._shared_conv_cache = {}

        # Task heads
        heatmap_channels = [1, 2, 2, 1, 2, 2]
        task_heads_params = parameters.get("task_heads", [])
        self.task_heads = [
            TtSeparateHead(
                device=device,
                in_channels=64,
                heatmap_out=heatmap_channels[i],
                parameters=task_heads_params[i] if i < len(task_heads_params) else {},
                model_config=model_config,
            )
            for i in range(6)
        ]

    def _get_shared_conv(self, batch_size, height, width):
        cache_key = (batch_size, height, width)
        if cache_key not in self._shared_conv_cache:
            config = create_conv2d_config(
                input_height=height,
                input_width=width,
                in_channels=self.shared_conv_in_channels,
                out_channels=self.shared_conv_out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                weight=self.shared_conv_weight,
                bias=self.shared_conv_bias,
                stride=(1, 1),
                padding=(1, 1),
                model_config=self.model_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=None,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self._shared_conv_cache[cache_key] = TtConv2d(config, self.device)
        return self._shared_conv_cache[cache_key]

    def __call__(self, x, device=None, batch_size=1):
        # Trunk
        trunk_outputs = self.trunk(x, batch_size=batch_size)
        x0, x1, x2, x3 = trunk_outputs

        # Neck
        neck_inputs = [x0, x1, x2, x3]
        x = self.neck(neck_inputs, batch_size=batch_size)
        if isinstance(x, list):
            x = x[0]

        if not ttnn.is_tensor_storage_on_device(x):
            x = ttnn.to_device(x, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        height, width = x.shape[1], x.shape[2]

        shared_conv = self._get_shared_conv(batch_size, height, width)
        x, (out_h, out_w) = shared_conv(x, return_output_dim=True)
        x = post_process_conv_output(x, batch_size, out_h, out_w, self.shared_conv_out_channels)

        # Task heads
        return [head(x, batch_size, out_h, out_w) for head in self.task_heads]
