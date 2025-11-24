# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ttnn
import torch


class TTSiglipVisionEmbeddings:
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self._unfold = torch.nn.Unfold(
            kernel_size=(config.patch_size, config.patch_size), stride=(config.patch_size, config.patch_size)
        )
        self.patch_embedding_weight = parameters["patch_embedding"]["weight"]
        self.patch_embedding_bias = parameters["patch_embedding"]["bias"]

    def __call__(self, pixel_values, position_embeddings):
        # Use PyTorch unfold to extract patches on host
        # Input: [B, 3, H, W] -> [B, 588, 4900] where 588 = 14*14*3, 4900 = 70*70 patches
        x = self._unfold(pixel_values)  # [B, 588, 4900]
        x = x.permute(0, 2, 1)  # [B, 4900, 588]

        # Pad last dimension to nearest 32 for TILE_LAYOUT: 588 -> 608
        from models.common.utility_functions import nearest_32

        pad_len = nearest_32(x.shape[-1]) - x.shape[-1]
        if pad_len > 0:
            padding = torch.zeros((x.shape[0], x.shape[1], pad_len), dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=-1)  # [B, 4900, 608]

        # Convert to TTNN tensor with ReplicateTensorToMesh
        x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Linear projection (replaces Conv2d)
        # Remove bias to avoid shape issues with batched matmul
        patch_embeds = ttnn.linear(
            x,
            self.patch_embedding_weight,
            bias=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        # Add bias via elementwise add if available (avoids matmul bias limitation with batched inputs)
        if self.patch_embedding_bias is not None:
            bias_tt = ttnn.reshape(self.patch_embedding_bias, (1, 1, self.patch_embedding_bias.shape[-1]))  # [1,1,1152]
            patch_embeds = ttnn.add(patch_embeds, bias_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # Add position embeddings
        patch_embeds = ttnn.add(
            patch_embeds, position_embeddings, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )

        return patch_embeds
