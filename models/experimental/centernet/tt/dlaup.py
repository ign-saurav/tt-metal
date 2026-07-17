# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    TtConv2d,
)
from models.experimental.centernet.tt.utils import TtConvTranspose2D


class TtIdentity:
    """Identity layer for pass-through operations."""

    def __init__(self):
        pass

    def __call__(self, x):
        return x


class TtIDAUp:
    """TTNN implementation of IDAUp (Iterative Deep Aggregation Upsampling)."""

    def __init__(self, node_kernel, out_dim, channels, up_factors, parameters, layer_args, device):
        self.intermediates = {}
        self.device = device
        self.channels = channels
        self.out_dim = out_dim
        self.up_factors = up_factors
        self.parameters = parameters
        self.layer_args = layer_args
        self.projs = []
        for i, c in enumerate(channels):
            try:
                proj_params = getattr(parameters, f"proj_{i}")
                proj = TtConv2d(
                    self._make_conv_config(
                        proj_params,
                        batch_size=1,
                        input_height=64,
                        input_width=64,
                        in_channels=c,
                        out_channels=out_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    ),
                    device,
                )
                self.projs.append(proj)
            except KeyError:
                self.projs.append(TtIdentity())

        self.upsampling_layers = []
        for i in range(len(channels)):
            try:
                up_params = getattr(parameters, f"up_{i}")
                up_args = getattr(layer_args, f"up_{i}")
                upsample_layer = TtConvTranspose2D(up_args, up_params, device)
                self.upsampling_layers.append(upsample_layer)
            except KeyError:
                self.upsampling_layers.append(None)

        for i in range(1, len(channels)):
            node_name = f"node_{i}"
            node_params = getattr(parameters, node_name)
            node_args = getattr(layer_args, node_name)
            node = TtConv2d(
                self._make_conv_config(
                    node_params,
                    batch_size=node_args["0"].batch_size,
                    input_height=node_args["0"].input_height,
                    input_width=node_args["0"].input_width,
                    in_channels=node_args["0"].in_channels,
                    out_channels=node_args["0"].out_channels,
                    kernel_size=node_args["0"].kernel_size,
                    stride=node_args["0"].stride,
                    padding=node_args["0"].padding,
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                ),
                device,
            )
            setattr(self, node_name, node)

    @staticmethod
    def _ensure_tuple(param):
        """Helper method to ensure parameter is a tuple."""
        if isinstance(param, int):
            return (param, param)
        elif not isinstance(param, tuple):
            return tuple(param)
        return param

    def _make_conv_config(
        self,
        params,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation,
    ):
        """Helper method to create Conv2dConfiguration from parameters."""
        from models.tt_cnn.tt.builder import Conv2dConfiguration

        kernel_size = self._ensure_tuple(kernel_size)
        padding = self._ensure_tuple(padding)
        stride = self._ensure_tuple(stride)

        if "conv_config" in params:
            return params["conv_config"]

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight=params["weight"],
            bias=params["bias"],
            activation=activation,
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
        )

    def __call__(self, layers):
        """Make TtIDAUp callable."""
        return self.forward(layers)

    def _apply_projection(self, l, i):
        """Apply projection to layer i."""
        n, h, w, c = l.shape
        proj = self.projs[i]
        if isinstance(proj, TtIdentity):
            return l

        proj_out = proj(l)
        proj_n, proj_h, proj_w, proj_c = proj_out.shape
        if proj_h != h or proj_w != w:
            total_elements = proj_n * proj_h * proj_w * proj_c
            expected_elements = n * h * w * self.out_dim

            if total_elements == expected_elements:
                if proj_h == 1 and proj_w == h * w:
                    # Format is [N, 1, H*W, C] -> [N, H, W, C]
                    proj_out = ttnn.reshape(proj_out, [n, h, w, self.out_dim])
                elif proj_h == self.out_dim and proj_c == h * w:
                    # Format is [N, C, H*W, 1] -> [N, H, W, C]
                    proj_out = ttnn.reshape(proj_out, [n, self.out_dim, h, w])
                    proj_out = ttnn.permute(proj_out, (0, 2, 3, 1))
                else:
                    # Fallback: direct reshape
                    proj_out = ttnn.reshape(proj_out, [n, h, w, self.out_dim])

        return proj_out

    def _apply_upsampling(self, proj_out, i):
        """Apply upsampling to projected output."""
        up_factor = self.up_factors[i]

        if up_factor <= 1:
            return proj_out
        else:
            up_layer_args = getattr(self.layer_args, f"up_{i}")
            proj_out = ttnn.reshape(
                proj_out, (proj_out.shape[0], up_layer_args.input_height, up_layer_args.input_width, proj_out.shape[3])
            )

        n_proj, h_proj, w_proj, c_proj = proj_out.shape
        target_h = h_proj * up_factor
        target_w = w_proj * up_factor

        # Pad input to tile-aligned dimensions if needed
        pad_h = (32 - (h_proj % 32)) % 32
        pad_w = (32 - (w_proj % 32)) % 32

        if pad_h > 0 or pad_w > 0:
            padded_h = h_proj + pad_h
            padded_w = w_proj + pad_w
            proj_out_padded = ttnn.pad(
                proj_out, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            proj_out_padded = proj_out
            padded_h, padded_w = h_proj, w_proj

        # Ensure input is in ROW_MAJOR layout
        if proj_out_padded.layout != ttnn.ROW_MAJOR_LAYOUT:
            proj_out_padded = ttnn.to_layout(proj_out_padded, ttnn.ROW_MAJOR_LAYOUT)

        # Use conv_transpose2d for upsampling
        up_out_padded = self.upsampling_layers[i](proj_out_padded)

        # Reshape if needed: [N, 1, H*W, C] -> [N, H, W, C]
        actual_out_n, actual_out_h, actual_out_w, actual_out_c = up_out_padded.shape
        expected_out_h = padded_h * up_factor
        expected_out_w = padded_w * up_factor
        if actual_out_h == 1 and actual_out_w == expected_out_h * expected_out_w:
            up_out_padded = ttnn.reshape(up_out_padded, [actual_out_n, expected_out_h, expected_out_w, actual_out_c])

        # Unpad to target dimensions if needed
        if pad_h > 0 or pad_w > 0:
            up_out = ttnn.slice(up_out_padded, (0, 0, 0, 0), (n_proj, target_h, target_w, actual_out_c))
        else:
            up_out = up_out_padded
        return up_out

    def _apply_node_convolution(self, x, layer_i, i):
        """Apply node convolution after concatenation."""
        x = ttnn.reshape(x, layer_i.shape)
        concat_out = ttnn.concat([x, layer_i], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        node_name = f"node_{i}"
        node = getattr(self, node_name)

        # Reshape for conv2d: [N, H, W, C] -> [N, 1, H*W, C]
        n, h, w, c = concat_out.shape
        reshaped = ttnn.reshape(concat_out, [n, 1, h * w, c])
        x = node(reshaped)
        # Reshape back: [N, 1, H*W, C] -> [N, H, W, C]
        x = ttnn.reshape(x, [n, h, w, self.out_dim])

        return x

    def forward(self, layers):
        """Forward pass of TtIDAUp."""
        assert len(self.channels) == len(layers), f"{len(self.channels)} vs {len(layers)} layers"
        layers = list(layers)

        for i, l in enumerate(layers):
            proj_out = self._apply_projection(l, i)
            up_out = self._apply_upsampling(proj_out, i)
            layers[i] = up_out

        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            x = self._apply_node_convolution(x, layers[i], i)
            y.append(x)

        return x, y


class TtDLAUp:
    """TTNN implementation of DLAUpsampling."""

    def __init__(
        self, channels, scales=(1, 2, 4, 8, 16), in_channels=None, parameters=None, layer_args=None, device=None
    ):
        self.device = device
        self.intermediates = {}
        self.layer_args = layer_args
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = list(scales)

        for i in range(len(channels) - 1):
            j = -i - 2
            ida_params = getattr(parameters, f"ida_{i}")
            ida_layer_args = getattr(layer_args, f"ida_{i}")
            setattr(
                self,
                f"ida_{i}",
                TtIDAUp(
                    3,
                    channels[j],
                    in_channels[j:],
                    [s // scales[j] for s in scales[j:]],
                    ida_params,
                    ida_layer_args,
                    device,
                ),
            )
            scales[j + 1 :] = [scales[j]] * len(scales[j + 1 :])
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def __call__(self, layers):
        """Make TtDLAUp callable."""
        return self.forward(layers)

    def forward(self, layers):
        """Forward pass of TtDLAUp."""
        layers = list(layers)
        assert len(layers) > 1

        for i in range(len(layers) - 1):
            ida = getattr(self, f"ida_{i}")
            x, y = ida(layers[-i - 2 :])
            layers[-i - 1 :] = y

        return x
