# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.transfuser.tt.gpt_block import TTGptBlock


def generate_token_embeddings_tt(image_tensor, lidar_tensor, seq_len, n_embd):
    """
    Generate token embeddings from NCHW format tensors.

    Args:
        image_tensor: (batch, channels, height, width) - e.g., (1, 72, 5, 22)
        lidar_tensor: (batch, channels, height, width) - e.g., (1, 72, 8, 8)
        seq_len: sequence length (should be 1)
        n_embd: embedding dimension (should be 72)

    Returns:
        token_embeddings: (batch, total_tokens, n_embd)
        Additional metadata for post-processing
    """
    bz = image_tensor.shape[0]
    img_c = image_tensor.shape[1]  # Should be 72
    img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]  # 5, 22

    lidar_c = lidar_tensor.shape[1]  # Should be 72
    lidar_h, lidar_w = lidar_tensor.shape[2], lidar_tensor.shape[3]  # 8, 8

    # Permute from NCHW to NHWC format
    # (batch, channels, height, width) -> (batch, height, width, channels)
    image_tokens = ttnn.permute(image_tensor, (0, 2, 3, 1))  # (1, 5, 22, 72)
    image_tokens = ttnn.reshape(image_tokens, (bz, img_h * img_w, n_embd))  # (1, 110, 72)

    lidar_tokens = ttnn.permute(lidar_tensor, (0, 2, 3, 1))  # (1, 8, 8, 72)
    lidar_tokens = ttnn.reshape(lidar_tokens, (bz, lidar_h * lidar_w, n_embd))  # (1, 64, 72)

    # Concatenate image and lidar tokens along sequence dimension
    token_embeddings = ttnn.concat([image_tokens, lidar_tokens], dim=1)  # (1, 174, 72)

    return token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w


def post_process_output_tt(
    x,
    bz,
    seq_len,
    img_vert_anchors,
    img_horz_anchors,
    lidar_vert_anchors,
    lidar_horz_anchors,
    n_embed,
    img_h,
    img_w,
    lidar_h,
    lidar_w,
):
    # Reshape to [bz, total_seq, n_embed]
    total_seq = seq_len * img_vert_anchors * img_horz_anchors + seq_len * lidar_vert_anchors * lidar_horz_anchors
    x = ttnn.reshape(x, (bz, total_seq, n_embed))

    # Split image and lidar tensors
    img_seq_len = seq_len * img_vert_anchors * img_horz_anchors

    # Slice image tensor
    image_tensor = x[:, :img_seq_len, :]
    image_tensor_out = ttnn.reshape(image_tensor, (bz * seq_len, -1, img_h, img_w))

    # Slice lidar tensor
    lidar_tensor = x[:, img_seq_len:, :]
    lidar_tensor_out = ttnn.reshape(lidar_tensor, (bz * seq_len, -1, lidar_h, lidar_w))

    return image_tensor_out, lidar_tensor_out


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
        self.pos_emb = parameters["pos_emb"]

        if self.use_velocity:
            # Store velocity embedding weights and bias as TTNN tensors
            self.vel_emb_weight = parameters["vel_emb_weight"]
            self.vel_emb_bias = parameters["vel_emb_bias"]

        self.tt_blocks = []
        for i in range(n_layer):
            self.tt_blocks.append(
                TTGptBlock(device, parameters[f"blocks_{i}"], n_head, dtype=dtype, memory_config=memory_config)
            )
        self.dtype = dtype
        self.memory_config = memory_config
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.seq_len = seq_len

    def __call__(self, tt_image_input, tt_lidar_input, velocity, n_embed):
        token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w = generate_token_embeddings_tt(
            tt_image_input, tt_lidar_input, self.seq_len, n_embed
        )

        if self.use_velocity:
            # Convert velocity to TTNN
            velocity_torch = velocity if isinstance(velocity, torch.Tensor) else ttnn.to_torch(velocity)
            velocity_embeddings = self.vel_emb(velocity_torch)
            velocity_embeddings = ttnn.from_torch(
                velocity_embeddings.unsqueeze(1),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.dtype,
                memory_config=self.memory_config,
            )
            # Now all tensors are TTNN tensors
            x = ttnn.add(self.pos_emb, token_embeddings)
            x = ttnn.add(x, velocity_embeddings)
        else:
            x = ttnn.add(self.pos_emb, token_embeddings)

        # Continue with transformer blocks
        for i in range(self.n_layer):
            x = self.tt_blocks[i](x)

        x = ttnn.layer_norm(x, weight=self.parameters["ln_f_weight"], bias=self.parameters["ln_f_bias"])
        tt_image_output, tt_lidar_output = post_process_output_tt(
            x,
            bz,
            seq_len,
            self.img_vert_anchors,
            self.img_horz_anchors,
            self.lidar_vert_anchors,
            self.lidar_horz_anchors,
            n_embed,
            img_h,
            img_w,
            lidar_h,
            lidar_w,
        )
        return tt_image_output, tt_lidar_output
