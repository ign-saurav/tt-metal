# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/layers/backbones/base_lss_fpn.py
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn.functional as F

# from mmcv.cnn import build_conv_layer
from models.experimental.BevDepth.reference.bevdepth.layers.heads.conv import build_conv_layer

# from mmdet3d.models import build_neck
# from mmdet.models import build_backbone
# from mmdet.models.backbones.resnet import BasicBlock
from models.experimental.BevDepth.reference.bevdepth.layers.heads.resnet import BasicBlock
from torch import nn

# from torch.cuda.amp.autocast_mode import autocast

from models.experimental.BevDepth.reference.bevdepth.layers.heads.builder import build_backbone, build_neck

try:
    from models.experimental.BevDepth.reference.bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from models.experimental.BevDepth.reference.bevdepth.ops.voxel_pooling_train import voxel_pooling_train

    _VOXEL_POOLING_AVAILABLE = True
except ImportError:
    print("Import VoxelPooling fail. Using PyTorch fallback.")
    _VOXEL_POOLING_AVAILABLE = False
    voxel_pooling_inference = None
    voxel_pooling_train = None


def _voxel_pooling_inference_fallback(
    geom_xyz: torch.Tensor,
    depth: torch.Tensor,
    context_features: torch.Tensor,
    voxel_num: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch replacement for the CUDA voxel pooling op.

    This function handles flexible input sizes by inferring spatial dimensions
    from the actual tensor shapes rather than assuming they match geom_xyz.
    """
    device = context_features.device
    if isinstance(voxel_num, torch.Tensor):
        voxel_sizes = voxel_num.detach().cpu().tolist()
    else:
        voxel_sizes = voxel_num if isinstance(voxel_num, (list, tuple)) else [voxel_num]
    num_voxel_x, num_voxel_y, num_voxel_z = [int(v) for v in voxel_sizes]

    B, num_cams, num_depth, num_height, num_width, _ = geom_xyz.shape
    channels = context_features.shape[1]
    depth_channels = depth.shape[1]

    # depth and context_features come in as (B*num_cams, C, H, W)
    # Infer actual spatial dimensions from tensor shapes (flexible for different input sizes)
    if len(depth.shape) == 4:
        # depth: (B*num_cams, depth_channels, H, W)
        _, _, actual_height, actual_width = depth.shape
    else:
        # Fallback: infer from total elements
        depth_total_elements = depth.numel()
        spatial_size = depth_total_elements // (B * num_cams * depth_channels)
        # Try to match geom_xyz aspect ratio, then adjust
        if spatial_size == num_height * num_width:
            actual_height, actual_width = num_height, num_width
        else:
            # Infer from spatial size - try to maintain aspect ratio
            aspect_ratio = num_width / num_height if num_height > 0 else 1.0
            actual_width = int((spatial_size * aspect_ratio) ** 0.5)
            actual_height = spatial_size // actual_width
            # Ensure valid dimensions
            while actual_height * actual_width != spatial_size and actual_width > 0:
                actual_width -= 1
                actual_height = spatial_size // actual_width if actual_width > 0 else 1

    # Reshape depth and context_features from (B*num_cams, ...) to (B, num_cams, ...)
    # Use actual spatial dimensions from tensor shapes
    depth = depth.view(B, num_cams, depth_channels, actual_height, actual_width)
    # Extract first num_depth channels (depth_channels should be >= num_depth)
    if depth_channels >= num_depth:
        depth = depth[:, :, :num_depth, :, :]
    else:
        # If fewer channels, pad or handle gracefully
        raise ValueError(f"depth_channels ({depth_channels}) < num_depth ({num_depth})")

    # context_features: (B*num_cams, channels, H, W)
    if len(context_features.shape) == 4:
        _, _, ctx_height, ctx_width = context_features.shape
        if ctx_height != actual_height or ctx_width != actual_width:
            # Reshape to match depth spatial dimensions if needed
            context_features = context_features.view(B, num_cams, channels, ctx_height, ctx_width)
            # Interpolate if dimensions don't match
            if ctx_height != actual_height or ctx_width != actual_width:
                context_features = torch.nn.functional.interpolate(
                    context_features.view(B * num_cams, channels, ctx_height, ctx_width),
                    size=(actual_height, actual_width),
                    mode="bilinear",
                    align_corners=False,
                ).view(B, num_cams, channels, actual_height, actual_width)
        else:
            context_features = context_features.view(B, num_cams, channels, actual_height, actual_width)
    else:
        context_features = context_features.view(B, num_cams, channels, actual_height, actual_width)

    context = context_features.permute(0, 1, 3, 4, 2).contiguous().unsqueeze(2).expand(-1, -1, num_depth, -1, -1, -1)

    # Reshape geom_xyz to match actual feature map dimensions if they differ
    if num_height != actual_height or num_width != actual_width:
        # Reshape geom_xyz from (B, num_cams, num_depth, num_height, num_width, 3)
        # to (B, num_cams, num_depth, actual_height, actual_width, 3)
        # First flatten spatial dimensions, then reshape
        geom_xyz_flat = geom_xyz.view(B, num_cams, num_depth, num_height * num_width, 3)
        # If total elements match, we can reshape directly
        if num_height * num_width == actual_height * actual_width:
            geom_xyz = geom_xyz_flat.view(B, num_cams, num_depth, actual_height, actual_width, 3)
        else:
            # If dimensions don't match, we need to interpolate the coordinates
            # Interpolate each coordinate channel separately
            # Convert to float for interpolation (interpolate requires float)
            geom_xyz_reshaped = geom_xyz.view(B, num_cams, num_depth, num_height, num_width, 3).float()
            # Permute to (B, num_cams, num_depth, 3, num_height, num_width) for interpolation
            geom_xyz_perm = geom_xyz_reshaped.permute(0, 1, 2, 5, 3, 4).contiguous()
            geom_xyz_interp = F.interpolate(
                geom_xyz_perm.view(B * num_cams * num_depth, 3, num_height, num_width),
                size=(actual_height, actual_width),
                mode="bilinear",
                align_corners=False,
            )
            # Reshape back to (B, num_cams, num_depth, actual_height, actual_width, 3)
            # Keep as float for now, will convert to long when extracting coordinates
            geom_xyz = (
                geom_xyz_interp.view(B, num_cams, num_depth, 3, actual_height, actual_width)
                .permute(0, 1, 2, 4, 5, 3)
                .contiguous()
            )

    geom = geom_xyz.long()
    x = geom[..., 0]
    y = geom[..., 1]
    z = geom[..., 2]

    valid_mask = (x >= 0) & (x < num_voxel_x) & (y >= 0) & (y < num_voxel_y) & (z >= 0) & (z < num_voxel_z)
    valid = valid_mask.to(depth.dtype)

    depth = depth.unsqueeze(-1)
    contributions = depth * context * valid.unsqueeze(-1)

    batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
    batch_indices = batch_indices.expand_as(depth[..., 0])

    x = x.clamp(0, num_voxel_x - 1)
    y = y.clamp(0, num_voxel_y - 1)

    flat_index = batch_indices * (num_voxel_y * num_voxel_x) + y * num_voxel_x + x

    bev = torch.zeros(B * num_voxel_y * num_voxel_x, channels, device=device, dtype=context_features.dtype)
    bev.index_add_(0, flat_index.view(-1).long(), contributions.view(-1, channels))
    bev = bev.view(B, num_voxel_y, num_voxel_x, channels).permute(0, 3, 1, 2).contiguous()
    return bev


def _voxel_pooling_train_fallback(
    geom_xyz: torch.Tensor,
    img_feat_with_depth: torch.Tensor,
    voxel_num: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch replacement for the CUDA voxel pooling train op."""
    device = img_feat_with_depth.device
    if isinstance(voxel_num, torch.Tensor):
        voxel_sizes = voxel_num.detach().cpu().tolist()
    else:
        voxel_sizes = voxel_num if isinstance(voxel_num, (list, tuple)) else [voxel_num]
    num_voxel_x, num_voxel_y, num_voxel_z = [int(v) for v in voxel_sizes]

    # img_feat_with_depth shape: (B, num_cams, num_depth, H, W, channels)
    B, num_cams, num_depth, num_height, num_width, channels = img_feat_with_depth.shape

    # Reshape to match expected format: (B, N, C) where N = num_cams * num_depth * H * W
    img_feat_flat = img_feat_with_depth.reshape(B, -1, channels)
    geom_flat = geom_xyz.reshape(B, -1, 3)

    geom = geom_flat.long()
    x = geom[..., 0]
    y = geom[..., 1]
    z = geom[..., 2]

    valid_mask = (x >= 0) & (x < num_voxel_x) & (y >= 0) & (y < num_voxel_y) & (z >= 0) & (z < num_voxel_z)
    valid = valid_mask.to(img_feat_with_depth.dtype).unsqueeze(-1)

    contributions = img_feat_flat * valid

    batch_indices = torch.arange(B, device=device).view(B, -1)

    x = x.clamp(0, num_voxel_x - 1)
    y = y.clamp(0, num_voxel_y - 1)

    flat_index = batch_indices * (num_voxel_y * num_voxel_x) + y * num_voxel_x + x

    bev = torch.zeros(B * num_voxel_y * num_voxel_x, channels, device=device, dtype=img_feat_with_depth.dtype)
    bev.index_add_(0, flat_index.view(-1).long(), contributions.view(-1, channels))
    bev = bev.view(B, num_voxel_y, num_voxel_x, channels).permute(0, 3, 1, 2).contiguous()
    return bev


__all__ = ["BaseLSSFPN"]

# At the top, after imports
from models.experimental.BevDepth.reference.bevdepth.layers.heads.builder import BACKBONES, MODELS, NECKS

# Import necks to ensure they are registered
from models.experimental.BevDepth.reference.bevdepth.layers.necks.second_fpn import SECONDFPN  # noqa: F401


# Register the class
@BACKBONES.register_module()
@NECKS.register_module()
@MODELS.register_module()
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm
        )
        self.aspp3 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm
        )
        self.aspp4 = _ASPPModule(
            inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(
                cfg=dict(
                    type="DCN",
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128,
                )
            ),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict["intrin_mats"][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict["ida_mats"][:, 0:1, ...]
        sensor2ego = mats_dict["sensor2ego_mats"][:, 0:1, ..., :3, :]
        bda = mats_dict["bda_mat"].view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    # @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class BaseLSSFPN(nn.Module):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
        use_da=False,
    ):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(BaseLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer("voxel_size", torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            "voxel_coord", torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_num", torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer("frustum", self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = self._configure_depth_aggregation_net()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf["in_channels"],
            depth_net_conf["mid_channels"],
            self.output_channels,
            self.depth_channels,
        )

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels, self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4, 2
            ).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous()
            )
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
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

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(
            batch_size, num_sweeps, num_cams, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3]
        )
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
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
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(
                batch_size * num_cams, source_features.shape[2], source_features.shape[3], source_features.shape[4]
            ),
            mats_dict,
        )
        depth = depth_feature[:, : self.depth_channels].softmax(dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        if self.training or self.use_da:
            img_feat_with_depth = depth.unsqueeze(1) * depth_feature[
                :, self.depth_channels : (self.depth_channels + self.output_channels)
            ].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            # Use fallback if CUDA ops not available
            if _VOXEL_POOLING_AVAILABLE:
                voxel_num_device = self.voxel_num.to(img_feat_with_depth.device)
                feature_map = voxel_pooling_train(geom_xyz, img_feat_with_depth.contiguous(), voxel_num_device)
            else:
                feature_map = _voxel_pooling_train_fallback(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num)
        else:
            context_features = depth_feature[
                :, self.depth_channels : (self.depth_channels + self.output_channels)
            ].contiguous()
            # Use fallback if CUDA ops not available
            if _VOXEL_POOLING_AVAILABLE:
                voxel_num_device = self.voxel_num.to(context_features.device)
                feature_map = voxel_pooling_inference(
                    geom_xyz,
                    depth,
                    context_features,
                    voxel_num_device,
                )
            else:
                feature_map = _voxel_pooling_inference_fallback(
                    geom_xyz,
                    depth,
                    context_features,
                    self.voxel_num,
                )
        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return feature_map.contiguous(), depth_feature[:, : self.depth_channels].softmax(dim=1)
        return feature_map.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
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
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
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

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)
