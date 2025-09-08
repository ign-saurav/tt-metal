# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.SSR.tt.patch_embed import TTPatchEmbed
from models.experimental.SSR.tt.basic_block import TTBasicLayer
from models.experimental.SSR.tt.mlp import TTMlp
from models.experimental.SSR.tt.mask_token_inference import TTMaskTokenInference
from models.common.lightweightmodule import LightweightModule


class TTTileSelection(LightweightModule):
    def __init__(self, device, parameters, args, num_cls, memory_config=None):
        super().__init__()
        self.device = device
        self.token_size = args.token_size
        self.num_layers = int(math.log2((args.imgsz // args.patchsz) // args.token_size))
        self.num_cls = num_cls
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Initialize patch embedding using existing TTPatchEmbed
        self.patch_embed = TTPatchEmbed(
            img_size=args.imgsz,
            patch_size=args.patchsz,
            in_chans=3,
            embed_dim=args.dim,
            device=device,
            parameters=parameters["patch_embed"],
            memory_config=memory_config,
        )

        # Initialize encoder layers using existing TTBasicLayer
        self.layers = []
        patches_resolution = (args.imgsz // args.patchsz, args.imgsz // args.patchsz)

        for i_layer in range(self.num_layers):
            layer = TTBasicLayer(
                device=device,
                parameters=parameters[f"layers.{i_layer}"],
                dim=int(args.dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer), patches_resolution[1] // (2**i_layer)),
                depth=2,
                num_heads=3,
                window_size=7,
                mlp_ratio=4.0,
                downsample=True if i_layer < self.num_layers - 1 else False,
                memory_config=memory_config,
            )
            self.layers.append(layer)

        # Layer norm parameters for different scales
        self.norm3_weight = parameters["norm3"]["weight"]
        self.norm3_bias = parameters["norm3"]["bias"]
        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]

        # Mask token embedding
        self.mask_token_weight = parameters["mask_token"]["weight"]

        # Initialize MLPs using existing TTMlp
        final_dim = 96 * (2**self.num_layers)

        self.fea_mlp3 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=final_dim,
            hidden_features=final_dim,
            out_features=final_dim,
            parameters=parameters["fea_mlp3"],
        )

        self.fea_mlp2 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=96 * (2 ** (self.num_layers - 1)),
            hidden_features=final_dim,
            out_features=final_dim,
            parameters=parameters["fea_mlp2"],
        )

        self.fea_mlp1 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=96 * (2 ** (self.num_layers - 2)),
            hidden_features=final_dim,
            out_features=final_dim,
            parameters=parameters["fea_mlp1"],
        )
        # Initialize mask token inference modules
        self.mask_pre3 = TTMaskTokenInference(
            device=device, parameters=parameters["mask_pre3"], dim=final_dim, num_heads=1
        )

        self.mask_pre2 = TTMaskTokenInference(
            device=device, parameters=parameters["mask_pre2"], dim=final_dim, num_heads=1
        )

        self.mask_pre1 = TTMaskTokenInference(
            device=device, parameters=parameters["mask_pre1"], dim=final_dim, num_heads=1
        )

        # MLP norm parameters
        self.mlp_norm3_weight = parameters["mlp_norm3"]["weight"]
        self.mlp_norm3_bias = parameters["mlp_norm3"]["bias"]
        self.mlp_norm2_weight = parameters["mlp_norm2"]["weight"]
        self.mlp_norm2_bias = parameters["mlp_norm2"]["bias"]
        self.mlp_norm1_weight = parameters["mlp_norm1"]["weight"]
        self.mlp_norm1_bias = parameters["mlp_norm1"]["bias"]

        # Classification MLPs
        self.mlp3 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=final_dim,
            hidden_features=96,
            out_features=96,
            parameters=parameters["mlp3"],
        )

        self.mlp2 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=final_dim,
            hidden_features=96,
            out_features=96,
            parameters=parameters["mlp2"],
        )

        self.mlp1 = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=final_dim,
            hidden_features=96,
            out_features=96,
            parameters=parameters["mlp1"],
        )

        # Linear classification layers
        self.linear3_weight = parameters["linear3"]["weight"]
        self.linear3_bias = parameters["linear3"]["bias"]
        self.linear2_weight = parameters["linear2"]["weight"]
        self.linear2_bias = parameters["linear2"]["bias"]
        self.linear1_weight = parameters["linear1"]["weight"]
        self.linear1_bias = parameters["linear1"]["bias"]

    def __call__(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Patch embedding using existing TTPatchEmbed
        x = self.patch_embed(x)

        # Encoder using existing TTBasicLayer components
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        # Apply layer normalization to different scale features
        x3 = ttnn.layer_norm(x, weight=self.norm3_weight, bias=self.norm3_bias)
        x2 = ttnn.layer_norm(x_downsample[-1], weight=self.norm2_weight, bias=self.norm2_bias)
        x1 = ttnn.layer_norm(x_downsample[-2], weight=self.norm1_weight, bias=self.norm1_bias)

        # Get mask tokens and expand for batch
        mask_tokens = ttnn.unsqueeze(self.mask_token_weight, 0)
        mask_tokens = ttnn.expand(mask_tokens, [B, -1, -1])

        # Process scale 3 (finest scale)
        fea_3_processed = self.fea_mlp3(x3)
        mask_tokens = ttnn.to_layout(mask_tokens, ttnn.ROW_MAJOR_LAYOUT)
        fea_3_processed = ttnn.to_layout(fea_3_processed, ttnn.ROW_MAJOR_LAYOUT)
        fea_3 = ttnn.concat([mask_tokens, fea_3_processed], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fea_3 = ttnn.to_layout(fea_3, ttnn.TILE_LAYOUT)

        mask_tokens = ttnn.slice(fea_3, [0, 0, 0], [B, 1, fea_3.shape[-1]])
        mask_3 = self.mask_pre3(fea_3)
        mask_3 = ttnn.layer_norm(mask_3, weight=self.mlp_norm3_weight, bias=self.mlp_norm3_bias)
        mask_3 = self.mlp3(mask_3)
        mask_3 = ttnn.linear(mask_3, self.linear3_weight, bias=self.linear3_bias)
        mask_3 = self._reshape_output(mask_3, B, self.token_size, self.token_size)

        # Process scale 2
        fea_2_processed = self.fea_mlp2(x2)
        mask_tokens = ttnn.to_layout(mask_tokens, ttnn.ROW_MAJOR_LAYOUT)
        fea_2_processed = ttnn.to_layout(fea_2_processed, ttnn.ROW_MAJOR_LAYOUT)
        fea_2 = ttnn.concat([mask_tokens, fea_2_processed], dim=1)
        fea_2 = ttnn.to_layout(fea_2, ttnn.TILE_LAYOUT)

        mask_tokens = ttnn.slice(fea_2, [0, 0, 0], [B, 1, fea_2.shape[-1]])
        mask_2 = self.mask_pre2(fea_2)
        mask_2 = ttnn.layer_norm(mask_2, weight=self.mlp_norm2_weight, bias=self.mlp_norm2_bias)
        mask_2 = self.mlp2(mask_2)
        mask_2 = ttnn.linear(mask_2, self.linear2_weight, bias=self.linear2_bias)
        mask_2 = self._reshape_output(mask_2, B, self.token_size * 2, self.token_size * 2)

        # Process scale 1 (coarsest scale)
        fea_1_processed = self.fea_mlp1(x1)
        mask_tokens = ttnn.to_layout(mask_tokens, ttnn.ROW_MAJOR_LAYOUT)
        fea_1_processed = ttnn.to_layout(fea_1_processed, ttnn.ROW_MAJOR_LAYOUT)
        fea_1 = ttnn.concat([mask_tokens, fea_1_processed], dim=1)
        fea_1 = ttnn.to_layout(fea_1, ttnn.TILE_LAYOUT)

        mask_1 = self.mask_pre1(fea_1)
        mask_1 = ttnn.layer_norm(mask_1, weight=self.mlp_norm1_weight, bias=self.mlp_norm1_bias)
        mask_1 = self.mlp1(mask_1)
        mask_1 = ttnn.linear(mask_1, self.linear1_weight, bias=self.linear1_bias)
        mask_1 = self._reshape_output(mask_1, B, self.token_size * 4, self.token_size * 4)

        return mask_3, mask_2, mask_1

    def _reshape_output(self, mask_output, B, H, W):
        """Reshape output to spatial dimensions"""
        # mask_output shape: [B, N, C] where C is num_cls
        N, C = mask_output.shape[1], mask_output.shape[2]

        # Transpose and reshape: [B, N, C] -> [B, C, N] -> [B, C, H, W]
        mask_output = ttnn.transpose(mask_output, 1, 2)  # [B, C, N]
        mask_output = ttnn.reshape(mask_output, [B, C, H, W])

        return mask_output
