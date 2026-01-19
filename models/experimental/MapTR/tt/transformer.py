# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MapTR Perception Transformer Implementation
"""

import ttnn
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from torchvision.transforms.functional import rotate as torch_rotate


class TtConvFuser(nn.Module):
    """TTNN Convolution-based feature fuser for multi-modal fusion."""

    def __init__(self, params: dict, device: ttnn.Device):
        super().__init__()
        self.device = device
        self.params = params

        # Extract conv parameters
        conv_params = params["conv"]
        self.conv_weight = conv_params["weight"]
        self.conv_bias = conv_params.get("bias")

        # Extract batch norm parameters
        bn_params = params["batch_norm"]
        self.bn_weight = bn_params["weight"]
        self.bn_bias = bn_params["bias"]
        self.bn_running_mean = bn_params["running_mean"]
        self.bn_running_var = bn_params["running_var"]
        self.bn_eps = bn_params.get("eps", 1e-5)
        self.bn_momentum = bn_params.get("momentum", 0.1)

    def forward(self, inputs: List[ttnn.Tensor]) -> ttnn.Tensor:
        # Concatenate inputs along channel dimension
        concatenated = ttnn.concat(inputs, dim=1)

        # Apply convolution
        conv_output = ttnn.conv2d(
            input=concatenated,
            weight=self.conv_weight,
            bias=self.conv_bias,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            groups=1,
        )

        # Apply batch normalization
        bn_output = ttnn.batch_norm(
            conv_output,
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
            weight=self.bn_weight,
            bias=self.bn_bias,
            epsilon=self.bn_eps,
            momentum=self.bn_momentum,
            training=False,
        )

        # Apply ReLU activation
        output = ttnn.relu(bn_output)

        return output


class TtMapTRPerceptionTransformer(nn.Module):
    """TTNN MapTR Perception Transformer Implementation."""

    def __init__(
        self,
        params: dict,
        device: ttnn.Device,
        encoder: nn.Module,
        decoder: nn.Module,
        embed_dims: int = 256,
        num_feature_levels: int = 1,
        num_cams: int = 6,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        len_can_bus: int = 18,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        rotate_center: List[int] = None,
        fuser: Optional[TtConvFuser] = None,
    ):
        super().__init__()

        if rotate_center is None:
            rotate_center = [100, 100]

        self.device = device
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center
        self.fuser = fuser

        # Check if using attention-based BEV encoder
        self.use_attn_bev = hasattr(encoder, "layers")

        # Initialize TTNN parameters
        self._init_ttnn_layers()

    def _init_ttnn_layers(self):
        """Initialize TTNN layers and parameters."""
        # Level embeddings
        self.level_embeds = self.params["level_embeds"]

        # Camera embeddings
        self.cams_embeds = self.params["cams_embeds"]

        # Reference points linear layer
        ref_params = self.params["reference_points"]
        self.reference_points_weight = ref_params["weight"]
        self.reference_points_bias = ref_params["bias"]

        # CAN bus MLP parameters
        # nn.Sequential structure: [0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU, [4/norm]=LayerNorm
        can_bus_params = self.params["can_bus_mlp"]
        self.can_bus_mlp_weight1 = can_bus_params["0"]["weight"]  # First Linear at index 0
        self.can_bus_mlp_bias1 = can_bus_params["0"]["bias"]
        self.can_bus_mlp_weight2 = can_bus_params["2"]["weight"]  # Second Linear at index 2 (after ReLU)
        self.can_bus_mlp_bias2 = can_bus_params["2"]["bias"]

        if self.can_bus_norm:
            self.can_bus_norm_weight = can_bus_params["norm"]["weight"]
            self.can_bus_norm_bias = can_bus_params["norm"]["bias"]
            self.can_bus_norm_eps = 1e-5

        # Feature projection layer (optional, for when feat dims != embed_dims)
        self.feat_proj_weight = self.params.get("feat_proj", {}).get("weight", None)
        self.feat_proj_bias = self.params.get("feat_proj", {}).get("bias", None)

    def _linear(self, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor = None) -> ttnn.Tensor:
        """TTNN linear layer implementation."""
        return ttnn.linear(input_tensor, weight, bias=bias)

    def _layer_norm(
        self, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float = 1e-5
    ) -> ttnn.Tensor:
        """TTNN layer norm implementation."""
        return ttnn.layer_norm(input_tensor, epsilon=eps, weight=weight, bias=bias)

    def _relu(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """TTNN ReLU activation."""
        return ttnn.relu(input_tensor)

    def _can_bus_mlp(self, can_bus_torch: torch.Tensor) -> ttnn.Tensor:
        """CAN bus MLP implementation in TTNN.

        Uses padding to make tensors tile-compatible (multiples of 32).
        Input can_bus is [bs, len_can_bus] where len_can_bus=18.

        Args:
            can_bus_torch: PyTorch tensor [bs, len_can_bus] - passed as torch for padding

        Returns:
            TTNN tensor [bs, embed_dims]
        """
        bs = can_bus_torch.shape[0]
        in_features = can_bus_torch.shape[1]  # 18

        # Pad input from [bs, 18] to [bs, 32] for tile compatibility
        padded_size = 32
        if in_features < padded_size:
            padding = torch.zeros(bs, padded_size - in_features, dtype=can_bus_torch.dtype)
            can_bus_padded = torch.cat([can_bus_torch, padding], dim=1)
        else:
            can_bus_padded = can_bus_torch

        # Convert padded input to TTNN
        can_bus = ttnn.from_torch(can_bus_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Get weights as torch tensors for padding
        weight1_torch = ttnn.to_torch(self.can_bus_mlp_weight1)  # [18, 128] (stored as T)
        bias1_torch = ttnn.to_torch(self.can_bus_mlp_bias1)  # [128]

        # Pad weight1 from [18, 128] to [32, 128]
        if weight1_torch.shape[0] < padded_size:
            w_padding = torch.zeros(
                padded_size - weight1_torch.shape[0], weight1_torch.shape[1], dtype=weight1_torch.dtype
            )
            weight1_padded = torch.cat([weight1_torch, w_padding], dim=0)
        else:
            weight1_padded = weight1_torch

        # Convert padded weight to TTNN
        weight1 = ttnn.from_torch(weight1_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        bias1 = ttnn.from_torch(bias1_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # First linear: [bs, 32] @ [32, 128] = [bs, 128] + ReLU
        hidden = ttnn.linear(can_bus, weight1, bias=bias1)
        hidden = ttnn.relu(hidden)

        # Clean up first layer tensors
        ttnn.deallocate(can_bus)
        ttnn.deallocate(weight1)
        ttnn.deallocate(bias1)

        # Second linear: [bs, 128] @ [128, 256] = [bs, 256] + ReLU
        # weight2 is [128, 256] - already tile-compatible
        output = ttnn.linear(hidden, self.can_bus_mlp_weight2, bias=self.can_bus_mlp_bias2)
        output = ttnn.relu(output)

        # Clean up hidden tensor
        ttnn.deallocate(hidden)

        # Layer norm if enabled
        if self.can_bus_norm:
            output = ttnn.layer_norm(
                output, weight=self.can_bus_norm_weight, bias=self.can_bus_norm_bias, epsilon=self.can_bus_norm_eps
            )

        return output

    def attn_bev_encode(
        self,
        mlvl_feats: List[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature encoding using attention-based encoder."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Get batch size from first feature
        bs = mlvl_feats[0].shape[0]

        # Expand BEV queries
        bev_queries = ttnn.unsqueeze(bev_queries, 1)
        bev_queries = ttnn.repeat(bev_queries, [1, bs, 1])

        # Flatten BEV position
        bev_pos = ttnn.reshape(bev_pos, [bs, self.embed_dims, bev_h * bev_w])
        bev_pos = ttnn.permute(bev_pos, [2, 0, 1])

        # Calculate shift from ego motion (same as PyTorch version)
        img_metas = kwargs.get("img_metas", [{}])
        delta_x = np.array([each.get("can_bus", np.zeros(18))[0] for each in img_metas])
        delta_y = np.array([each.get("can_bus", np.zeros(18))[1] for each in img_metas])
        ego_angle = np.array([each.get("can_bus", np.zeros(18))[-2] / np.pi * 180 for each in img_metas])

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        # Convert shift to TTNN tensor [bs, 2] matching PyTorch reference
        # PyTorch ref: shift = shift.permute(1, 0) converts [2, bs] -> [bs, 2]
        shift_array = np.stack([shift_x, shift_y], axis=0)  # [2, bs]
        shift_torch = torch.from_numpy(shift_array).permute(1, 0).float()  # [bs, 2]
        shift = ttnn.from_torch(shift_torch, dtype=ttnn.bfloat16, device=self.device)

        # Handle previous BEV features
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = ttnn.permute(prev_bev, [1, 0, 2])

            if self.rotate_prev_bev:
                # Rotate previous BEV features based on ego motion
                # Fallback to PyTorch
                prev_bev_torch = ttnn.to_torch(prev_bev)  # (bev_h*bev_w, bs, embed_dims)
                for i in range(bs):
                    rotation_angle = img_metas[i].get("can_bus", np.zeros(18))[-1]
                    tmp_prev_bev = prev_bev_torch[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = torch_rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev_torch[:, i] = tmp_prev_bev[:, 0]
                prev_bev = ttnn.from_torch(prev_bev_torch, dtype=ttnn.bfloat16, device=self.device)

        # Process CAN bus signals
        can_bus_list = [torch.from_numpy(each.get("can_bus", np.zeros(18))).float() for each in img_metas]
        can_bus_torch = torch.stack(can_bus_list)
        can_bus_torch = can_bus_torch[:, : self.len_can_bus]
        # Pass torch tensor directly to _can_bus_mlp for padding
        can_bus_processed = self._can_bus_mlp(can_bus_torch)
        can_bus_processed = ttnn.unsqueeze(can_bus_processed, 0)

        # Add CAN bus to BEV queries
        if self.use_can_bus:
            bev_queries = ttnn.add(bev_queries, can_bus_processed)

        # Process multi-level features
        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # Flatten and permute features [num_cam, bs, h*w, c]
            feat = ttnn.reshape(feat, [bs, num_cam, c, h * w])
            feat = ttnn.permute(feat, [1, 0, 3, 2])

            # Project features to embed_dims if dimension doesn't match
            if c != self.embed_dims and self.feat_proj_weight is not None:
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                feat = ttnn.linear(feat, self.feat_proj_weight, bias=self.feat_proj_bias)

            # Add camera embeddings
            # feat shape: [num_cam, bs, h*w, c] = [6, 1, 1400, 256]
            # cams_embeds shape: [num_cam, c] = [6, 256]
            if self.use_cams_embeds:
                cams_embed_expanded = ttnn.unsqueeze(ttnn.unsqueeze(self.cams_embeds, 1), 1)
                # cams_embed_expanded: [6, 1, 1, 256]
                # Repeat to match feat: [num_cam, bs, h*w, c] = [6, 1, 1400, 256]
                cams_embed_expanded = ttnn.repeat(cams_embed_expanded, [1, bs, h * w, 1])
                feat = ttnn.add(feat, cams_embed_expanded)
                ttnn.deallocate(cams_embed_expanded)

            # Add level embeddings
            # level_embeds shape: [num_levels, c], we take one level: [1, 256]
            level_embed = self.level_embeds[lvl : lvl + 1]
            level_embed_expanded = ttnn.unsqueeze(ttnn.unsqueeze(level_embed, 0), 0)
            # level_embed_expanded: [1, 1, 1, 256]
            # Repeat to match feat: [num_cam, bs, h*w, c] = [6, 1, 1400, 256]
            level_embed_expanded = ttnn.repeat(level_embed_expanded, [num_cam, bs, h * w, 1])
            feat = ttnn.add(feat, level_embed_expanded)
            ttnn.deallocate(level_embed_expanded)

            feat_flatten.append(feat)

        # Concatenate all features
        feat_flatten = ttnn.concat(feat_flatten, dim=2)
        feat_flatten = ttnn.permute(feat_flatten, [0, 2, 1, 3])

        # Calculate level start index (cumulative sum of spatial sizes)
        # PyTorch ref: level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # spatial_shapes is still a list of tuples at this point
        spatial_prods = [h * w for h, w in spatial_shapes]
        level_start_index_list = [0] + list(np.cumsum(spatial_prods)[:-1]) if len(spatial_prods) > 1 else [0]
        level_start_index_torch = torch.tensor(level_start_index_list, dtype=torch.long)
        level_start_index = ttnn.from_torch(level_start_index_torch, dtype=ttnn.int32, device=self.device)

        # Convert spatial shapes to tensor (after using it for level_start_index calculation)
        spatial_shapes_torch = torch.tensor(spatial_shapes, dtype=torch.long)
        spatial_shapes_ttnn = ttnn.from_torch(spatial_shapes_torch, dtype=ttnn.int32, device=self.device)

        # Call encoder
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes_ttnn,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )

        return bev_embed

    def lss_bev_encode(
        self,
        mlvl_feats: List[ttnn.Tensor],
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature encoding using LSS-based encoder."""
        assert len(mlvl_feats) == 1, "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs.get("img_metas", [])

        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape

        bev_embed = ttnn.reshape(bev_embed, [bs, c, -1])
        bev_embed = ttnn.permute(bev_embed, [0, 2, 1])

        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature extraction."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            bev_embed = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs,
            )

        # Fuse with LiDAR features if available
        if lidar_feat is not None and self.fuser is not None:
            bs = mlvl_feats[0].shape[0]

            # Reshape BEV features for fusion
            bev_embed = ttnn.reshape(bev_embed, [bs, bev_h, bev_w, -1])
            bev_embed = ttnn.permute(bev_embed, [0, 3, 1, 2])

            # Process LiDAR features
            lidar_feat = ttnn.permute(lidar_feat, [0, 1, 3, 2])
            lidar_feat = ttnn.interpolate(lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False)

            # Fuse features
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = ttnn.reshape(fused_bev, [bs, -1, bev_h * bev_w])
            fused_bev = ttnn.permute(fused_bev, [0, 2, 1])
            bev_embed = fused_bev

        return bev_embed

    def forward(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        object_query_embed: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        reg_branches: Optional[nn.ModuleList] = None,
        cls_branches: Optional[nn.ModuleList] = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """TTNN forward pass."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Get BEV features
        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        # Get batch size
        bs = mlvl_feats[0].shape[0]

        # Split object query embeddings into position and content
        object_query_embed = ttnn.to_layout(object_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_pos, query = ttnn.split(object_query_embed, self.embed_dims, dim=1)

        # Expand queries for batch dimension
        query_pos = ttnn.unsqueeze(query_pos, 0)
        query_pos = ttnn.expand(query_pos, (bs, -1, -1))
        query_pos = ttnn.to_layout(query_pos, layout=ttnn.TILE_LAYOUT)

        query = ttnn.unsqueeze(query, 0)
        query = ttnn.expand(query, (bs, -1, -1))

        # Calculate reference points
        reference_points = ttnn.linear(query_pos, self.reference_points_weight, bias=self.reference_points_bias)
        reference_points = ttnn.sigmoid(reference_points)
        init_reference_out = reference_points

        # Permute for decoder input
        query = ttnn.permute(query, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        bev_embed = ttnn.permute(bev_embed, (1, 0, 2))

        # Create spatial shapes and level start index tensors
        spatial_shapes = ttnn.from_torch(
            torch.tensor([[bev_h, bev_w]], dtype=torch.int32), dtype=ttnn.int32, device=self.device
        )
        level_start_index = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, device=self.device)

        # Call decoder
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
