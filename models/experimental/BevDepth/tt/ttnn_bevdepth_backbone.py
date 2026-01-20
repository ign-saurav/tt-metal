# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch

from models.experimental.BevDepth.tt.ttnn_resnet50_backbone import TtResNet50Backbone
from models.experimental.BevDepth.tt.ttnn_secondfpn import TtSecondFpnBackbone
from models.experimental.BevDepth.tt.ttnn_depthnet import TtDepthNet
from models.experimental.BevDepth.reference.base_lss_fpn import _voxel_pooling_inference_fallback


class TtBaseLSSFPN:
    """
    TTNN implementation of BaseLSSFPN backbone."""

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
        """
        self.device = device
        self.model_config = model_config
        self.use_torch_fallback_secondfpn = os.environ.get("FALLBACK_ON_SECONDFPN", "1") == "1"

        self.lss_conf = lss_conf or {}
        self._init_lss_config()

        # Initialize TTNN components
        batch_size = self.model_config.get("batch_size", 1)

        # Image backbone: ResNet50
        self.img_backbone = TtResNet50Backbone(
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
        self.img_neck = TtSecondFpnBackbone(
            device=device,
            parameters=neck_parameters,
            in_channels=self.model_config.get("neck_in_channels", [256, 512, 1024, 2048]),
            out_channels=self.model_config.get("neck_out_channels", [128, 128, 128, 128]),
            upsample_strides=self.model_config.get("neck_upsample_strides", [0.25, 0.5, 1, 2]),
            model_config=self.model_config,
            input_shapes=neck_input_shapes,
            batch_size=self.model_config.get("batch_size", 1),
            use_torch_fallback=self.use_torch_fallback_secondfpn,
        )

        # DepthNet: Depth estimation network
        self.depth_net = TtDepthNet(
            device=device,
            parameters=depthnet_parameters,
            in_channels=self.model_config.get("depthnet_in_channels", 512),
            mid_channels=self.model_config.get("depthnet_mid_channels", 256),
            context_channels=self.model_config.get("depthnet_context_channels", 512),
            depth_channels=self.model_config.get("depthnet_depth_channels", 112),
            model_config=self.model_config,
        )

        self.depth_channels = self.depth_net.depth_channels

        self.voxel_pooling_inference = _voxel_pooling_inference_fallback

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

        imgs_flat = imgs.flatten(0, 2)

        img_feats_list = []
        for i in range(imgs_flat.shape[0]):
            img = imgs_flat[i]
            img_nhwc = img.permute(1, 2, 0).unsqueeze(0)
            img_ttnn = ttnn.from_torch(
                img_nhwc,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            img_ttnn = ttnn.to_device(img_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

            features = self.img_backbone(img_ttnn, input_height=imH, input_width=imW)

            layer_names = ["layer1", "layer2", "layer3", "layer4"]
            neck_inputs_ttnn = []
            for layer_name in layer_names:
                feat = features.get(layer_name)
                if feat is not None:
                    feat_torch = ttnn.to_torch(feat)
                    ttnn.deallocate(feat, force=True)

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
                fresh_neck = TtSecondFpnBackbone(
                    device=self.device,
                    parameters=self._neck_params,
                    in_channels=self.model_config.get("neck_in_channels", [256, 512, 1024, 2048]),
                    out_channels=self.model_config.get("neck_out_channels", [128, 128, 128, 128]),
                    upsample_strides=self.model_config.get("neck_upsample_strides", [0.25, 0.5, 1, 2]),
                    model_config=self.model_config,
                    input_shapes=neck_input_shapes,
                    batch_size=1,
                    use_torch_fallback=self.use_torch_fallback_secondfpn,
                )
                neck_output = fresh_neck(neck_inputs_ttnn, batch_size=1)

                if isinstance(neck_output, list):
                    neck_feat = neck_output[0]
                else:
                    neck_feat = neck_output

                img_feats_list.append(neck_feat)

        if img_feats_list:
            for i, feat in enumerate(img_feats_list):
                if feat.is_sharded():
                    img_feats_list[i] = ttnn.sharded_to_interleaved(feat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                img_feats_list[i] = ttnn.to_memory_config(img_feats_list[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)

            img_feats_ttnn = ttnn.concat(img_feats_list, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            img_feats_torch = ttnn.to_torch(img_feats_ttnn)

            if len(img_feats_torch.shape) == 4:
                img_feats_torch = img_feats_torch.permute(0, 3, 1, 2)

            _, C, H, W = img_feats_torch.shape
            img_feats = img_feats_torch.reshape(batch_size, num_sweeps, num_cams, C, H, W)

            ttnn.deallocate(img_feats_ttnn)
            for feat in img_feats_list:
                ttnn.deallocate(feat)

            return img_feats
        else:
            raise RuntimeError("Failed to process images through backbone and neck")

    def _forward_depth_net(self, feat, mats_dict, return_ttnn=False):
        """Forward through depth net."""
        batch_size, num_cams, C, H, W = feat.shape

        feat_flat = feat.reshape(batch_size * num_cams, C, H, W)

        feat_nhwc = feat_flat.permute(0, 2, 3, 1)
        feat_ttnn = ttnn.from_torch(
            feat_nhwc,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        feat_ttnn = ttnn.to_device(feat_ttnn, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        depth_feature = self.depth_net(feat_ttnn, batch_size=batch_size * num_cams, mats_dict=mats_dict)

        if return_ttnn:
            return depth_feature

        depth_feature_torch = ttnn.to_torch(depth_feature)
        if len(depth_feature_torch.shape) == 4:
            depth_feature_torch = depth_feature_torch.permute(0, 3, 1, 2)

        return depth_feature_torch

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False):
        """Forward function for single sweep."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        img_feats = self._get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]  # [B, num_cams, C, H, W]

        depth_feature_ttnn = self._forward_depth_net(source_features, mats_dict, return_ttnn=True)

        depth_channels = self.depth_net.depth_channels
        total_channels = depth_feature_ttnn.shape[-1]

        depth_ttnn = ttnn.slice(
            depth_feature_ttnn,
            [0, 0, 0, 0],
            [depth_feature_ttnn.shape[0], depth_feature_ttnn.shape[1], depth_feature_ttnn.shape[2], depth_channels],
        )
        depth_ttnn = ttnn.softmax(depth_ttnn, dim=-1)

        context_start = depth_channels
        context_end = depth_channels + self.output_channels
        context_features_ttnn = ttnn.slice(
            depth_feature_ttnn,
            [0, 0, 0, context_start],
            [depth_feature_ttnn.shape[0], depth_feature_ttnn.shape[1], depth_feature_ttnn.shape[2], context_end],
        )

        depth_torch = ttnn.to_torch(depth_ttnn)
        if len(depth_torch.shape) == 4:
            depth_torch = depth_torch.permute(0, 3, 1, 2)
        depth = depth_torch

        context_features_torch = ttnn.to_torch(context_features_ttnn)
        if len(context_features_torch.shape) == 4:
            context_features_torch = context_features_torch.permute(0, 3, 1, 2)
        context_features = context_features_torch.contiguous()

        ttnn.deallocate(depth_ttnn)
        ttnn.deallocate(context_features_ttnn)
        ttnn.deallocate(depth_feature_ttnn)

        geom_xyz = self._get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )

        geom_xyz = (
            (geom_xyz - (self.voxel_coord.to(geom_xyz.device) - self.voxel_size.to(geom_xyz.device) / 2.0))
            / self.voxel_size.to(geom_xyz.device)
        ).int()

        feature_map = self.voxel_pooling_inference(
            geom_xyz,
            depth,
            context_features,
            self.voxel_num,
        )

        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def __call__(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        """Forward function for BEVDepth backbone."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

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

        if len(ret_feature_list) > 1:
            ret_feature_list_ttnn = []
            for feat in ret_feature_list:
                feat_nhwc = feat.permute(0, 2, 3, 1) if len(feat.shape) == 4 else feat
                feat_ttnn = ttnn.from_torch(
                    feat_nhwc,
                    dtype=self.model_config.get("ACTIVATIONS_DTYPE", ttnn.bfloat16),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                ret_feature_list_ttnn.append(ttnn.to_memory_config(feat_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG))

            ret_feature_ttnn = ttnn.concat(ret_feature_list_ttnn, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ret_feature_torch = ttnn.to_torch(ret_feature_ttnn)

            if len(ret_feature_torch.shape) == 4:
                ret_feature_torch = ret_feature_torch.permute(0, 3, 1, 2)

            for feat in ret_feature_list_ttnn:
                ttnn.deallocate(feat)
            ttnn.deallocate(ret_feature_ttnn)

            if is_return_depth:
                return ret_feature_torch.contiguous(), key_frame_res[1]
            return ret_feature_torch.contiguous()

        if is_return_depth:
            return key_frame_feature.contiguous(), key_frame_res[1]
        return key_frame_feature.contiguous()
