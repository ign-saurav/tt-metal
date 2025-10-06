# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.transfuser.tt.gpt_block import TTGptBlock


class TTGpt(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        n_head,
        n_layer,
        use_velocity,
        img_vert_anchors,
        img_horz_anchors,
        lidar_vert_anchors,
        lidar_horz_anchors,
        seq_len,
        n_embd,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    ):
        self.device = device
        self.parameters = parameters
        self.n_head = n_head
        self.n_layer = n_layer
        self.use_velocity = use_velocity
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                seq_len * img_vert_anchors * img_horz_anchors + seq_len * lidar_vert_anchors * lidar_horz_anchors,
                n_embd,
            )
        )
        if self.use_velocity:
            self.vel_emb = nn.Linear(1, n_embd)
        self.tt_blocks = []
        for i in range(n_layer):
            self.tt_blocks.append(
                TTGptBlock(device, parameters[f"blocks_{i}"], n_head, dtype=dtype, memory_config=memory_config)
            )
        self.dtype = dtype
        self.memory_config = memory_config

    # def forward(self, token_embeddings, velocity):
    #     # self.pos_emb = ttnn.from_torch(self.pos_emb, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype, memory_config=self.memory_config)
    #     if self.use_velocity == True:
    #         velocity_embeddings = self.vel_emb(velocity)  # (B, C)
    #         # add (learnable) positional embedding and velocity embedding for all tokens
    #         x = self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)  # (B, an * T, C)
    #     else:
    #         x = self.pos_emb + token_embeddings
    #     x = ttnn.from_torch(
    #         x, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype, memory_config=self.memory_config
    #     )
    #     for i in range(self.n_layer):
    #         x = self.tt_blocks[i](x)

    #     x = ttnn.layer_norm(x, weight=self.parameters["ln_f_weight"], bias=self.parameters["ln_f_bias"])
    #     return x
    def __call__(self, token_embeddings, velocity):
        # Convert positional embeddings to TTNN format once
        pos_emb_tt = ttnn.from_torch(
            self.pos_emb.data,  # Use .data to get the tensor without gradient tracking
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        # Convert token embeddings to TTNN
        token_emb_tt = ttnn.from_torch(
            token_embeddings,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

        if self.use_velocity:
            # Velocity is already a TTNN tensor from the test
            velocity_torch = ttnn.to_torch(velocity)
            velocity_embeddings = self.vel_emb(velocity_torch)  # (B, C)
            velocity_emb_tt = ttnn.from_torch(
                velocity_embeddings,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.dtype,
                memory_config=self.memory_config,
            )
            # Broadcast velocity embeddings across sequence length
            # velocity_emb_tt shape: (B, C) -> (B, 1, C) for broadcasting
            velocity_emb_tt = ttnn.unsqueeze(velocity_emb_tt, dim=1)

            # Add positional + token + velocity embeddings
            x = ttnn.add(pos_emb_tt, token_emb_tt)
            x = ttnn.add(x, velocity_emb_tt)
        else:
            # Add positional + token embeddings
            x = ttnn.add(pos_emb_tt, token_emb_tt)

        # Pass through transformer blocks
        for i in range(self.n_layer):
            x = self.tt_blocks[i](x)

        # Final layer normalization
        x = ttnn.layer_norm(x, weight=self.parameters["ln_f_weight"], bias=self.parameters["ln_f_bias"])

        return x
