import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from .RHAG import TTPatchEmbed, TTPatchUnEmbed, TTRHAG
from .upsample import TTUpsample


class TTHAT(LightweightModule):
    """TTNN Hybrid Attention Transformer base class"""

    def __init__(
        self,
        device,
        parameters,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        memory_config=None,
        h=64,
        w=64,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.parameters = parameters
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.h = h
        self.w = w
        self.ape = ape
        self.layers = []
        num_feat = 64

        self.patch_embed = TTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,  # Set to 0 as in original
            embed_dim=180,
            norm_layer=1,
            device=device,
            parameters=parameters["patch_embed"],
            memory_config=memory_config,
        )

        for i_layer in range(self.num_layers):
            layer = TTRHAG(
                device=device,
                parameters=self.parameters[f"layers.{i_layer}"],
                dim=embed_dim,
                input_resolution=(64, 64),
                depth=6,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=overlap_ratio,
                mlp_ratio=mlp_ratio,
                img_size=max(64, 64),
                patch_size=4,
                resi_connection=resi_connection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.layers.append(layer)

        self.patch_unembed = TTPatchUnEmbed(
            mesh_device=device, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        self.upsample = TTUpsample(upscale, num_feat, device)

        # Mean normalization values
        if in_chans == 3:
            rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
            self.mean = ttnn.from_torch(rgb_mean, dtype=ttnn.bfloat16, device=device)
        else:
            self.mean = ttnn.zeros((1, 1, 1, 1), dtype=ttnn.bfloat16, device=device)

    def calculate_mask(self, x_size):
        """Calculate attention mask for SW-MSA with proper padding"""
        h, w = x_size

        # Calculate padding needed to make dimensions divisible by window_size
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size

        # Use padded dimensions
        padded_h = h + pad_h
        padded_w = w + pad_w

        # Create mask with padded dimensions
        img_mask = torch.zeros((1, padded_h, padded_w, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        # Convert to TTNN tensor
        return ttnn.from_torch(img_mask, dtype=ttnn.bfloat16, device=self.device)

    def forward_features(self, x):
        """Forward pass through transformer layers"""
        x_size = (self.h, self.w)

        # Patch embedding
        x = self.patch_embed(x)

        # Add absolute position embedding if enabled
        if self.ape and hasattr(self.parameters, "absolute_pos_embed"):
            x = ttnn.add(x, self.parameters.absolute_pos_embed, memory_config=self.memory_config)
        # Apply transformer layers
        x = ttnn.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
        for i in range(self.num_layers):
            x = self.layers[i](x, x_size, self.parameters["forward_params"])

        # Layer normalization
        x = ttnn.layer_norm(
            x,
            weight=self.parameters.norm.weight,
            bias=self.parameters.norm.bias,
            memory_config=self.memory_config,
        )

        # Patch unembedding
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        """Main forward pass"""
        # Normalize input
        x = ttnn.subtract(x, self.mean, memory_config=self.memory_config)
        x = ttnn.multiply(x, self.img_range, memory_config=self.memory_config)

        if self.upsampler == "pixelshuffle":
            # Shallow feature extraction
            x = ttnn.conv2d(
                x,
                self.parameters.conv_first.weight,
                bias=self.parameters.conv_first.bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                device=self.device,
                memory_config=self.memory_config,
            )

            # Deep feature extraction with residual connection
            features = self.forward_features(x)
            # Residual connection after body
            if hasattr(self.parameters, "conv_after_body"):
                features = ttnn.conv2d(
                    features,
                    self.parameters.conv_after_body.weight,
                    bias=self.parameters.conv_after_body.bias,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    device=self.device,
                    memory_config=self.memory_config,
                )

            x = ttnn.add(x, features, memory_config=self.memory_config)

            # Pre-upsample convolution
            x = ttnn.conv2d(
                x,
                self.parameters.conv_before_upsample.weight,
                bias=self.parameters.conv_before_upsample.bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                device=self.device,
                memory_config=self.memory_config,
            )

            # LeakyReLU activation
            x = ttnn.leaky_relu(x, negative_slope=0.01)

            # Upsampling
            x = self.parameters.upsample(x)

            # Final convolution
            x = ttnn.conv2d(
                x,
                self.parameters.conv_last.weight,
                bias=self.parameters.conv_last.bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                device=self.device,
                memory_config=self.memory_config,
            )

        # Denormalize output
        x = ttnn.divide(x, self.img_range, memory_config=self.memory_config)
        x = ttnn.add(x, self.mean, memory_config=self.memory_config)

        return x


class TTTileRefinement(TTHAT):
    """TTNN Tile Refinement Module

    Outputs both feature and final upsampled image, following the same pattern
    as the PyTorch TileRefinement class.
    """

    def forward(self, x):
        """Forward pass that returns both output and features"""
        # Normalize input
        batch_size = x.shape[0]
        self.mean = ttnn.to_layout(self.mean, ttnn.TILE_LAYOUT)
        x = ttnn.subtract(x, self.mean, memory_config=self.memory_config)
        x = ttnn.multiply(x, self.img_range, memory_config=self.memory_config)

        if self.upsampler == "pixelshuffle":
            # Shallow feature extraction
            x = ttnn.permute(x, (0, 2, 3, 1))  # (batch_size, embed_dim, num_patches)
            self.conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                activation="",
                output_layout=ttnn.TILE_LAYOUT,
                deallocate_activation=True,  # Free input memory after use
                reallocate_halo_output=True,  # Reduce memory fragmentation
                act_block_h_override=32,  # Use smaller activation blocks
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # Use height sharding
            )
            self.compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.parameters["conv_first"]["weight"],
                bias_tensor=self.parameters["conv_first"]["bias"],
                in_channels=3,
                out_channels=180,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # slice_config=slice_config,
            )
            x = ttnn.reshape(x, [batch_size, 64, 64, 180])  # TODO
            x = ttnn.permute(x, (0, 3, 1, 2))

            # Deep feature extraction - store as fea
            fea = self.forward_features(x)

            # Residual connection after body (using fea, not x like in HAT)
            self.conv_afterbody_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                activation="",
                output_layout=ttnn.TILE_LAYOUT,
                deallocate_activation=False,  # Free input memory after use
                reallocate_halo_output=True,  # Reduce memory fragmentation
                act_block_h_override=32,  # Use smaller activation blocks
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # Use height sharding
            )
            fea = ttnn.permute(fea, (0, 2, 3, 1))
            x_after_body = ttnn.conv2d(
                input_tensor=fea,
                weight_tensor=self.parameters["conv_after_body"]["weight"],
                bias_tensor=self.parameters["conv_after_body"]["bias"],
                in_channels=180,
                out_channels=180,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=fea.shape[0],
                input_height=fea.shape[1],
                input_width=fea.shape[2],
                conv_config=self.conv_afterbody_config,
                compute_config=self.compute_config,
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x_after_body = ttnn.reshape(x_after_body, [batch_size, 64, 64, 180])  # TODO
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.add(x, x_after_body, memory_config=self.memory_config)

            # Pre-upsample convolution
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.parameters.conv_before_upsample.weight,
                bias_tensor=self.parameters.conv_before_upsample.bias,
                in_channels=180,
                out_channels=64,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=self.memory_config,
                dtype=ttnn.bfloat16,
                return_weights_and_bias=False,
            )

            # LeakyReLU activation
            x = ttnn.leaky_relu(x, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, [batch_size, 64, 64, 64])  # TODO

            # Upsampling
            x = self.upsample(x, self.parameters["upsample"])

            # Final convolution
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.parameters.conv_last.weight,
                bias_tensor=self.parameters.conv_last.bias,
                in_channels=64,
                out_channels=3,
                device=self.device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=self.memory_config,
                dtype=ttnn.bfloat16,
                return_weights_and_bias=False,
            )

            x = ttnn.reshape(x, [batch_size, 256, 256, 3])  # TODO
        # Denormalize output
        x = ttnn.divide(x, self.img_range, memory_config=self.memory_config)
        self.mean = ttnn.permute(self.mean, (0, 2, 3, 1))
        x = ttnn.add(x, self.mean, memory_config=self.memory_config)
        self.mean = ttnn.permute(self.mean, (0, 3, 1, 2))

        return x, fea
