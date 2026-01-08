# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
)


class TtIdentity:
    """Identity layer for pass-through operations."""

    def __init__(self):
        pass

    def __call__(self, x):
        return x


def fill_up_weights_tt(up_weight):
    """TTNN version of fill_up_weights for ConvTranspose2d initialization."""
    w = up_weight
    f = math.ceil(w.shape[-2] / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)

    # Create weight pattern for upsampling
    for i in range(w.shape[-2]):
        for j in range(w.shape[-1]):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

    for c in range(1, w.shape[0]):
        w[c, 0, :, :] = w[0, 0, :, :]

    return w


class TtIDAUp:
    """TTNN implementation of IDAUp (Iterative Deep Aggregation Upsampling)."""

    def __init__(self, node_kernel, out_dim, channels, up_factors, parameters, device):
        self.intermediates = {}
        self.device = device
        self.channels = channels
        self.out_dim = out_dim
        self.up_factors = up_factors
        self.parameters = parameters

        # Initialize projection layers
        for i, c in enumerate(channels):
            proj_name = f"proj_{i}"
            try:
                proj_params = getattr(parameters, proj_name)
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
            except KeyError:
                proj = TtIdentity()

            setattr(self, f"proj_{i}", proj)

        # Initialize node layers
        for i in range(1, len(channels)):
            node_name = f"node_{i}"
            node_params = getattr(parameters, node_name)
            node = TtConv2d(
                self._make_conv_config(
                    node_params,
                    batch_size=1,
                    input_height=64,
                    input_width=64,
                    in_channels=out_dim * 2,
                    out_channels=out_dim,
                    kernel_size=node_kernel,
                    stride=1,
                    padding=node_kernel // 2,
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
        try:
            proj_params = getattr(self.parameters, f"proj_{i}")
            n, h, w, c = l.shape

            proj_config = Conv2dConfiguration(
                input_height=h,
                input_width=w,
                in_channels=c,
                out_channels=self.out_dim,
                batch_size=n,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                weight=proj_params["weight"],
                bias=proj_params["bias"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                weights_dtype=ttnn.bfloat16,
                output_dtype=ttnn.bfloat16,
            )
            proj = TtConv2d(proj_config, self.device)
            proj_out = proj(l)

            # Reshape if needed to restore spatial dimensions
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
        except KeyError:
            # Identity layer
            return l

    def _apply_upsampling(self, proj_out, i):
        """Apply upsampling to projected output."""
        up_factor = self.up_factors[i]

        if up_factor <= 1:
            return proj_out

        n_proj, h_proj, w_proj, c_proj = proj_out.shape
        target_h = h_proj * up_factor
        target_w = w_proj * up_factor

        up_params = getattr(self.parameters, f"up_{i}")
        weight_tensor = up_params["weight"]

        # Ensure weight is on host and in ROW_MAJOR layout
        if hasattr(weight_tensor, "device") and weight_tensor.device() is not None:
            weight_tensor = ttnn.from_device(weight_tensor)

        if weight_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            weight_tensor = ttnn.to_layout(weight_tensor, ttnn.ROW_MAJOR_LAYOUT)

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
        up_out_padded = ttnn.conv_transpose2d(
            input_tensor=proj_out_padded,
            weight_tensor=weight_tensor,
            bias_tensor=None,
            in_channels=self.out_dim,
            out_channels=self.out_dim,
            kernel_size=(up_factor * 2, up_factor * 2),
            stride=(up_factor, up_factor),
            padding=(up_factor // 2, up_factor // 2),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=self.out_dim,
            device=self.device,
            batch_size=n_proj,
            input_height=padded_h,
            input_width=padded_w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mirror_kernel=True,
        )

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
        # Ensure spatial dimensions match
        x_n, x_h, x_w, x_c = x.shape
        layer_n, layer_h, layer_w, layer_c = layer_i.shape

        if x_h != layer_h or x_w != layer_w:
            # Upsample the smaller one to match
            if x_h < layer_h or x_w < layer_w:
                scale_h = max(1, layer_h // x_h) if x_h > 0 else 1
                scale_w = max(1, layer_w // x_w) if x_w > 0 else 1
                if scale_h > 1 or scale_w > 1:
                    x = ttnn.upsample(x, (scale_h, scale_w))
            elif layer_h < x_h or layer_w < x_w:
                scale_h = max(1, x_h // layer_h) if layer_h > 0 else 1
                scale_w = max(1, x_w // layer_w) if layer_w > 0 else 1
                if scale_h > 1 or scale_w > 1:
                    layer_i = ttnn.upsample(layer_i, (scale_h, scale_w))

        # Concatenate along channel dimension
        concat_out = ttnn.concat([x, layer_i], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply node convolution
        n, h, w, c = concat_out.shape
        node_params = getattr(self.parameters, f"node_{i}")

        node_config = Conv2dConfiguration(
            input_height=h,
            input_width=w,
            in_channels=c,
            out_channels=self.out_dim,
            batch_size=n,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            weight=node_params["weight"],
            bias=node_params["bias"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
        )
        node = TtConv2d(node_config, self.device)

        # Reshape for conv2d: [N, H, W, C] -> [N, 1, H*W, C]
        reshaped = ttnn.reshape(concat_out, [n, 1, h * w, c])
        x = node(reshaped)
        # Reshape back: [N, 1, H*W, C] -> [N, H, W, C]
        x = ttnn.reshape(x, [n, h, w, self.out_dim])

        return x

    def forward(self, layers):
        """Forward pass of TtIDAUp."""
        assert len(self.channels) == len(layers), f"{len(self.channels)} vs {len(layers)} layers"
        layers = list(layers)

        # Step 1: Apply projection and upsampling to each layer
        for i, l in enumerate(layers):
            proj_out = self._apply_projection(l, i)
            up_out = self._apply_upsampling(proj_out, i)
            layers[i] = up_out

        # Step 2: Apply node convolutions
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            x = self._apply_node_convolution(x, layers[i], i)
            y.append(x)

        return x, y


class TtDLAUp:
    """TTNN implementation of DLAUpsampling."""

    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None, parameters=None, device=None):
        self.device = device
        self.intermediates = {}

        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = list(scales)

        for i in range(len(channels) - 1):
            j = -i - 2
            ida_params = getattr(parameters, f"ida_{i}")
            setattr(
                self,
                f"ida_{i}",
                TtIDAUp(3, channels[j], in_channels[j:], [s // scales[j] for s in scales[j:]], ida_params, device),
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
