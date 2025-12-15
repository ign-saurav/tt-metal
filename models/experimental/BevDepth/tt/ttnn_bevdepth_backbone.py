# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from models.experimental.BevDepth.tt.ttnn_resnet50_backbone import ResNet50_BEVDepth
from models.experimental.BevDepth.tt.ttnn_secondfpn import SECONDFPN_TTNN
from models.experimental.BevDepth.tt.ttnn_depthnet import DepthNet_TTNN


class TtBaseLSSFPN:
    """
    TTNN implementation of BaseLSSFPN backbone.

    This class combines:
    - ResNet50_BEVDepth: Image backbone for feature extraction
    - SECONDFPN_TTNN: Feature pyramid network (neck)
    - DepthNet_TTNN: Depth estimation network

    The LSS (Lift-Splat-Shoot) transformation from camera features to BEV
    is handled using PyTorch for complex geometric operations.
    """

    def __init__(
        self,
        device,
        backbone_parameters,
        neck_parameters,
        depthnet_parameters,
        lss_conf=None,
        model_config=None,
    ):
        """
        Initialize TTNN BaseLSSFPN backbone.

        Args:
            device: TTNN device
            backbone_parameters: Parameters for ResNet50 backbone
            neck_parameters: Parameters for SECONDFPN neck
            depthnet_parameters: Parameters for DepthNet
            lss_conf: LSS configuration dict with:
                - x_bound: [min, max, step]
                - y_bound: [min, max, step]
                - z_bound: [min, max, step]
                - d_bound: [min, max, step]
                - final_dim: [H, W]
                - downsample_factor: int
                - output_channels: int
            model_config: Model configuration dict (dtype, math fidelity, etc.)
        """
        self.device = device
        self.model_config = model_config

        # Store LSS configuration
        self.lss_conf = lss_conf or {}
        self._init_lss_config()

        # Initialize TTNN components
        batch_size = self.model_config.get("batch_size", 1)

        # Image backbone: ResNet50
        self.img_backbone = ResNet50_BEVDepth(
            device=device,
            parameters=backbone_parameters,
            batch_size=batch_size,
            model_config=self.model_config,
            return_intermediate=True,
            return_block_outputs=False,
        )

        self._neck_params = neck_parameters
        img_h, img_w = self.final_dim
        neck_input_shapes = [
            (img_h // 4, img_w // 4),
            (img_h // 8, img_w // 8),
            (img_h // 16, img_w // 16),
            (img_h // 32, img_w // 32),
        ]
        self.img_neck = SECONDFPN_TTNN(
            device=device,
            parameters=neck_parameters,
            in_channels=self.model_config.get("neck_in_channels", [256, 512, 1024, 2048]),
            out_channels=self.model_config.get("neck_out_channels", [128, 128, 128, 128]),
            upsample_strides=self.model_config.get("neck_upsample_strides", [0.25, 0.5, 1, 2]),
            model_config=self.model_config,
            input_shapes=neck_input_shapes,
            batch_size=self.model_config.get("batch_size", 1),
            use_torch_fallback=self.model_config.get("use_torch_fallback", False),
        )

        # DepthNet: Depth estimation network
        self.depth_net = DepthNet_TTNN(
            device=device,
            parameters=depthnet_parameters,
            in_channels=self.model_config.get("depthnet_in_channels", 512),
            mid_channels=self.model_config.get("depthnet_mid_channels", 256),
            context_channels=self.model_config.get("depthnet_context_channels", 512),
            depth_channels=self.model_config.get("depthnet_depth_channels", 112),
            model_config=self.model_config,
        )

        # Store depth_channels for later use
        self.depth_channels = self.depth_net.depth_channels

        # Initialize voxel pooling functions
        self._init_voxel_pooling()

    def _init_lss_config(self):
        """Initialize LSS configuration buffers."""
        lss_conf = self.lss_conf

        # Extract LSS parameters
        x_bound = lss_conf.get("x_bound", [-51.2, 51.2, 0.8])
        y_bound = lss_conf.get("y_bound", [-51.2, 51.2, 0.8])
        z_bound = lss_conf.get("z_bound", [-5.0, 3.0, 0.2])
        d_bound = lss_conf.get("d_bound", [2.0, 58.0, 0.5])
        final_dim = lss_conf.get("final_dim", [256, 704])
        downsample_factor = lss_conf.get("downsample_factor", 16)
        output_channels = lss_conf.get("output_channels", 80)

        # Register buffers (as tensors, not nn.Module buffers since we're not a nn.Module)
        self.voxel_size = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
        self.voxel_coord = torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
        self.voxel_num = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])

        self.d_bound = d_bound
        self.final_dim = final_dim
        self.downsample_factor = downsample_factor
        self.output_channels = output_channels

        # Create frustum
        self.frustum = self._create_frustum()

    def _create_frustum(self):
        """Generate frustum for LSS transformation."""
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def _init_voxel_pooling(self):
        """Initialize voxel pooling functions."""
        try:
            from models.experimental.BevDepth.reference.bevdepth.ops.voxel_pooling_inference import (
                voxel_pooling_inference,
            )

            self.voxel_pooling_inference = voxel_pooling_inference
            self._voxel_pooling_available = True
        except ImportError:
            from models.experimental.BevDepth.reference.bevdepth.layers.backbones.base_lss_fpn import (
                _voxel_pooling_inference_fallback,
            )

            self.voxel_pooling_inference = _voxel_pooling_inference_fallback
            self._voxel_pooling_available = False

    def _get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord."""
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # B x N x D x H x W x 4
        points = self.frustum.to(sensor2ego_mat.device)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)

        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        return points[..., :3]

    def _get_cam_feats(self, imgs):
        """Get camera features using TTNN components."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        # Flatten for processing: [B*num_sweeps*num_cams, 3, H, W]
        imgs_flat = imgs.flatten(0, 2)  # [B*num_sweeps*num_cams, 3, H, W]

        # Process each image through backbone and neck
        img_feats_list = []
        for i in range(imgs_flat.shape[0]):
            img = imgs_flat[i]  # [3, H, W]
            # Convert to NHWC: [1, H, W, 3]
            img_nhwc = img.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]
            img_ttnn = ttnn.from_torch(
                img_nhwc,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            img_ttnn = ttnn.to_device(img_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Get backbone features
            features = self.img_backbone(img_ttnn, input_height=imH, input_width=imW)

            # Convert features to TTNN format for neck, deallocating each immediately
            layer_names = ["layer1", "layer2", "layer3", "layer4"]
            neck_inputs_ttnn = []
            for layer_name in layer_names:
                feat = features.get(layer_name)
                if feat is not None:
                    # Convert to PyTorch (copies data off device)
                    feat_torch = ttnn.to_torch(feat)
                    # Deallocate original immediately to free memory
                    ttnn.deallocate(feat, force=True)

                    # Features are in NHWC, convert to TTNN for neck
                    feat_ttnn = ttnn.from_torch(
                        feat_torch,
                        dtype=self.model_config["ACTIVATIONS_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    feat_ttnn = ttnn.to_layout(feat_ttnn, ttnn.TILE_LAYOUT)
                    neck_inputs_ttnn.append(feat_ttnn)

            if neck_inputs_ttnn:
                img_h, img_w = self.final_dim
                neck_input_shapes = [
                    (img_h // 4, img_w // 4),
                    (img_h // 8, img_w // 8),
                    (img_h // 16, img_w // 16),
                    (img_h // 32, img_w // 32),
                ]
                fresh_neck = SECONDFPN_TTNN(
                    device=self.device,
                    parameters=self._neck_params,
                    in_channels=self.model_config.get("neck_in_channels", [256, 512, 1024, 2048]),
                    out_channels=self.model_config.get("neck_out_channels", [128, 128, 128, 128]),
                    upsample_strides=self.model_config.get("neck_upsample_strides", [0.25, 0.5, 1, 2]),
                    model_config=self.model_config,
                    input_shapes=neck_input_shapes,
                    batch_size=1,
                    use_torch_fallback=self.model_config.get("use_torch_fallback", False),
                )
                neck_output = fresh_neck(neck_inputs_ttnn, batch_size=1)

                # Convert back to PyTorch
                if isinstance(neck_output, list):
                    neck_feat = neck_output[0]
                else:
                    neck_feat = neck_output

                neck_feat_torch = ttnn.to_torch(neck_feat)
                # Convert from NHWC to NCHW
                if len(neck_feat_torch.shape) == 4:
                    neck_feat_torch = neck_feat_torch.permute(0, 3, 1, 2)

                img_feats_list.append(neck_feat_torch)

        # Reshape back to [B, num_sweeps, num_cams, C, H, W]
        if img_feats_list:
            img_feats = torch.cat(img_feats_list, dim=0)
            _, C, H, W = img_feats.shape
            img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams, C, H, W)
            return img_feats
        else:
            raise RuntimeError("Failed to process images through backbone and neck")

    def _forward_depth_net(self, feat, mats_dict):
        """Forward through depth net."""
        batch_size, num_cams, C, H, W = feat.shape

        # Flatten for depth net: [B*num_cams, C, H, W]
        feat_flat = feat.reshape(batch_size * num_cams, C, H, W)

        # Convert to TTNN format
        feat_nhwc = feat_flat.permute(0, 2, 3, 1)  # NCHW -> NHWC
        feat_ttnn = ttnn.from_torch(
            feat_nhwc,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        feat_ttnn = ttnn.to_device(feat_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Process through depth net
        # DepthNet returns concatenated [depth_channels + context_channels] in NHWC format
        depth_feature = self.depth_net(feat_ttnn, batch_size=batch_size * num_cams, mats_dict=mats_dict)

        # Convert back to PyTorch
        depth_feature_torch = ttnn.to_torch(depth_feature)
        # Convert from NHWC to NCHW: [B*num_cams, H, W, depth_channels + context_channels] -> [B*num_cams, depth_channels + context_channels, H, W]
        if len(depth_feature_torch.shape) == 4:
            depth_feature_torch = depth_feature_torch.permute(0, 3, 1, 2)

        return depth_feature_torch

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False):
        """Forward function for single sweep."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        # Get camera features
        img_feats = self._get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]  # [B, num_cams, C, H, W]

        # Forward through depth net
        depth_feature = self._forward_depth_net(source_features, mats_dict)

        # Extract depth and context features
        depth_channels = self.depth_net.depth_channels
        depth = depth_feature[:, :depth_channels].softmax(dim=1, dtype=depth_feature.dtype)
        context_features = depth_feature[:, depth_channels : (depth_channels + self.output_channels)].contiguous()

        # Get geometry
        geom_xyz = self._get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )

        # Convert geometry to voxel coordinates
        geom_xyz = (
            (geom_xyz - (self.voxel_coord.to(geom_xyz.device) - self.voxel_size.to(geom_xyz.device) / 2.0))
            / self.voxel_size.to(geom_xyz.device)
        ).int()

        # Voxel pooling (inference mode)
        if self._voxel_pooling_available:
            voxel_num_device = self.voxel_num.to(context_features.device)
            feature_map = self.voxel_pooling_inference(
                geom_xyz,
                depth,
                context_features,
                voxel_num_device,
            )
        else:
            feature_map = self.voxel_pooling_inference(
                geom_xyz,
                depth,
                context_features,
                self.voxel_num,
            )

        if is_return_depth:
            return feature_map.contiguous(), depth_feature[:, :depth_channels].softmax(dim=1)
        return feature_map.contiguous()

    def __call__(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        """
        Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps, num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor, optional): Timestamps with shape of (B, num_sweeps, num_cameras).
            is_return_depth(bool, optional): Whether to return depth. Default: False.

        Returns:
            Tensor: BEV feature map, or tuple (feature_map, depth) if is_return_depth=True.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        # Process key frame (sweep_index=0)
        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_depth=is_return_depth
        )

        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index, sweep_imgs[:, sweep_index : sweep_index + 1, ...], mats_dict, is_return_depth=False
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            # Concatenate features from all sweeps along channel dimension (matching reference)
            return torch.cat(ret_feature_list, 1), key_frame_res[1]

        # Concatenate features from all sweeps along channel dimension (matching reference)
        return torch.cat(ret_feature_list, 1)
