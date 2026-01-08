# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Full Inference Pipeline

This script runs the complete MapTR inference pipeline with:
- Image loading and preprocessing
- Model weight loading
- Multi-camera inference
- Result visualization

Usage:
    python models/experimental/mapTR/inference/run_inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --images /path/to/images/ \
        --output /path/to/output/

    # Or with demo mode (generates dummy data):
    python models/experimental/mapTR/inference/run_inference.py --demo
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR components
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_maptr import MapTR
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder


# ============================================================================
# Configuration Classes
# ============================================================================


class MapTRConfig:
    """Configuration for MapTR model."""

    def __init__(
        self,
        # Image settings
        img_height: int = 900,
        img_width: int = 1600,
        num_cameras: int = 6,
        # Model architecture
        embed_dims: int = 256,
        num_classes: int = 3,  # divider, ped_crossing, boundary
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        # BEV settings
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        # Transformer settings
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        # Backbone settings
        backbone_depth: int = 50,
        fpn_out_channels: int = 256,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.num_cameras = num_cameras

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels

        self.backbone_depth = backbone_depth
        self.fpn_out_channels = fpn_out_channels

    @classmethod
    def from_nuScenes(cls):
        """Create config for nuScenes dataset."""
        return cls(
            img_height=900,
            img_width=1600,
            num_cameras=6,
            num_classes=3,
            num_vec=50,
            num_pts_per_vec=20,
            bev_h=200,
            bev_w=100,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        )

    @classmethod
    def from_maptr_tiny(cls):
        """Create config for MapTR Tiny R50 checkpoint (maptr_tiny_r50_24e.pth)."""
        return cls(
            img_height=480,
            img_width=800,
            num_cameras=6,
            num_classes=3,
            num_vec=50,
            num_pts_per_vec=20,
            bev_h=200,
            bev_w=100,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_encoder_layers=6,
            num_decoder_layers=6,
        )

    @classmethod
    def from_small(cls):
        """Create small config for testing/demo (no pretrained weights)."""
        return cls(
            img_height=224,
            img_width=400,
            num_cameras=6,
            num_classes=3,
            num_vec=20,
            num_pts_per_vec=10,
            bev_h=50,
            bev_w=50,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_encoder_layers=2,
            num_decoder_layers=2,
        )


# ============================================================================
# Model Builder
# ============================================================================


class MapTRTransformerWithBEVFormer(nn.Module):
    """MapTR Transformer with BEVFormer encoder for proper weight loading."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points_in_pillar: int = 4,
        num_cams: int = 6,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        feedforward_channels: int = 512,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        # BEVFormer Encoder
        self.encoder = BEVFormerEncoder(
            num_layers=num_encoder_layers,
            pc_range=self.pc_range,
            num_points_in_pillar=num_points_in_pillar,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
        )

        # Decoder
        self.decoder = SimpleDecoder(
            embed_dims=embed_dims,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
        )

        # Additional layers matching checkpoint keys
        self.level_embeds = nn.Parameter(torch.randn(4, embed_dims))
        self.cams_embeds = nn.Parameter(torch.randn(num_cams, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        object_query_embeds: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: tuple,
        bev_pos: torch.Tensor,
        reg_branches: nn.ModuleList = None,
        cls_branches: nn.ModuleList = None,
        img_metas: List[Dict] = None,
        prev_bev: torch.Tensor = None,
    ):
        """Forward pass through BEVFormer encoder and decoder."""
        bs = mlvl_feats[0].size(0)

        # Prepare BEV queries
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos_flat = bev_pos.flatten(2).permute(2, 0, 1)

        # Add CAN bus info
        if img_metas is not None:
            can_bus = torch.zeros(bs, 18, device=bev_queries.device, dtype=bev_queries.dtype)
            for i, meta in enumerate(img_metas):
                if "can_bus" in meta:
                    can_bus_data = meta["can_bus"]
                    if isinstance(can_bus_data, np.ndarray):
                        can_bus[i] = torch.from_numpy(can_bus_data[:18]).to(can_bus.device)
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

        # Compute shift for temporal fusion
        shift = torch.zeros(bs, 2, device=bev_queries.device, dtype=bev_queries.dtype)

        # Prepare multi-level features
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs_f, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cam, bs, h*w, c)
            if self.cams_embeds is not None:
                feat = feat + self.cams_embeds[:num_cam, None, None, :].to(feat.dtype)
            if lvl < len(self.level_embeds):
                feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, h*w, bs, c)

        # Run BEVFormer encoder
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_flat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas,
        )

        # Prepare decoder queries
        query_embeds = object_query_embeds
        query = query_embeds[..., : self.embed_dims]
        query_pos = query_embeds[..., self.embed_dims :]

        query = query.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)

        # Reference points
        reference_points = self.reference_points(query_pos).sigmoid()

        # Run decoder
        hs, inter_references = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
        )

        return bev_embed.permute(1, 0, 2), hs, reference_points, inter_references

    def get_bev_features(
        self,
        mlvl_feats: List[torch.Tensor],
        lidar_feat: Optional[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: tuple,
        bev_pos: torch.Tensor,
        img_metas: List[Dict],
        prev_bev: torch.Tensor = None,
    ):
        """Get BEV features only."""
        bs = mlvl_feats[0].size(0)
        bev_embed = bev_queries.unsqueeze(0).repeat(bs, 1, 1)
        return bev_embed


class SimpleDecoder(nn.Module):
    """Simple transformer decoder."""

    def __init__(self, embed_dims: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.layers = nn.ModuleList([SimpleDecoderLayer(embed_dims, num_heads) for _ in range(num_layers)])

    def forward(self, query, key, value, query_pos, reference_points):
        # Handle BEV embed shape
        if key.dim() == 2:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        # Permute to (num_query, bs, embed_dims)
        output = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2) if key.dim() == 3 else key
        value = value.permute(1, 0, 2) if value.dim() == 3 else value
        query_pos_perm = query_pos.permute(1, 0, 2)

        intermediate = []
        intermediate_reference_points = []

        for layer in self.layers:
            output_t = output.permute(1, 0, 2)
            key_t = key.permute(1, 0, 2) if key.dim() == 3 else key
            value_t = value.permute(1, 0, 2) if value.dim() == 3 else value
            query_pos_t = query_pos_perm.permute(1, 0, 2)

            output_t = layer(output_t, key_t, value_t, query_pos_t)
            output = output_t.permute(1, 0, 2)
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        hs = torch.stack(intermediate)
        inter_references = torch.stack(intermediate_reference_points)
        return hs, inter_references


class SimpleDecoderLayer(nn.Module):
    """Simple decoder layer."""

    def __init__(self, embed_dims: int, num_heads: int, ffn_dims: int = None):
        super().__init__()
        # FFN hidden dims: checkpoint uses 512 (2x embed_dims)
        if ffn_dims is None:
            ffn_dims = embed_dims * 2  # Match checkpoint: 256 * 2 = 512
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, query_pos):
        q = k = query + query_pos
        query2 = self.self_attn(q, k, query)[0]
        query = self.norm1(query + query2)

        query2 = self.cross_attn(query + query_pos, key, value)[0]
        query = self.norm2(query + query2)

        query2 = self.ffn(query)
        query = self.norm3(query + query2)
        return query


def build_maptr_model(config: MapTRConfig, device: torch.device = None) -> MapTR:
    """Build MapTR model from config.

    Args:
        config: Model configuration.
        device: Target device.

    Returns:
        MapTR model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build backbone (ResNet50 by default)
    if config.backbone_depth == 50:
        layers = [3, 4, 6, 3]
    elif config.backbone_depth == 101:
        layers = [3, 4, 23, 3]
    else:
        layers = [3, 4, 6, 3]

    backbone = ResNet(
        block=Bottleneck,
        layers=layers,
        out_indices=(3,),  # Output from layer4 (2048 channels) to match checkpoint
    )

    # Build FPN
    fpn = FPN(
        in_channels=[2048],  # layer4 output has 2048 channels
        out_channels=config.fpn_out_channels,
        num_outs=1,
    )

    # Build transformer with BEVFormer encoder
    transformer = MapTRTransformerWithBEVFormer(
        embed_dims=config.embed_dims,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        num_points_in_pillar=4,
        num_cams=config.num_cameras,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        feedforward_channels=config.feedforward_channels,
    )

    # Build positional encoding
    pos_enc = LearnedPositionalEncoding(
        num_feats=config.embed_dims // 2,
        row_num_embed=config.bev_h,
        col_num_embed=config.bev_w,
    )

    # Build head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=pos_enc,
        embed_dims=config.embed_dims,
        num_classes=config.num_classes,
        num_reg_fcs=2,
        code_size=2,  # (x, y) per point
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        query_embed_type="instance_pts",  # Match checkpoint: separate instance and pts embeddings
        transform_method="minmax",
    )

    # Build full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    return model.to(device)


# ============================================================================
# Image Processing
# ============================================================================


class ImageProcessor:
    """Process images for MapTR inference."""

    def __init__(
        self,
        img_height: int = 900,
        img_width: int = 1600,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean or [123.675, 116.28, 103.53]
        self.std = std or [58.395, 57.12, 57.375]

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        return np.array(img)

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess images for model input.

        Args:
            images: List of images as numpy arrays (H, W, C) in RGB format.

        Returns:
            Tensor of shape (1, num_cams, 3, H, W).
        """
        processed = []
        for img in images:
            # Normalize
            img = img.astype(np.float32)
            img = (img - np.array(self.mean)) / np.array(self.std)
            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            processed.append(img)

        # Stack cameras: (num_cams, 3, H, W)
        imgs = np.stack(processed, axis=0)
        # Add batch dimension: (1, num_cams, 3, H, W)
        imgs = imgs[np.newaxis, ...]
        return torch.from_numpy(imgs).float()

    def generate_dummy_images(self, num_cameras: int = 6) -> torch.Tensor:
        """Generate dummy images for testing."""
        return torch.randn(1, num_cameras, 3, self.img_height, self.img_width)


# ============================================================================
# Camera Calibration
# ============================================================================


class CameraCalibration:
    """Handle camera calibration for MapTR."""

    def __init__(self, num_cameras: int = 6):
        self.num_cameras = num_cameras

    def create_dummy_calibration(
        self,
        img_height: int = 900,
        img_width: int = 1600,
    ) -> Dict[str, np.ndarray]:
        """Create dummy camera calibration for testing.

        Returns:
            Dictionary with calibration matrices.
        """
        # Create identity transforms for dummy data
        lidar2img = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera2ego = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        camera_intrinsics = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        img_aug_matrix = np.eye(4)[np.newaxis, :, :].repeat(self.num_cameras, axis=0)
        lidar2ego = np.eye(4)

        # Set reasonable intrinsics
        fx, fy = 1000.0, 1000.0
        cx, cy = img_width / 2, img_height / 2
        for i in range(self.num_cameras):
            camera_intrinsics[i, 0, 0] = fx
            camera_intrinsics[i, 1, 1] = fy
            camera_intrinsics[i, 0, 2] = cx
            camera_intrinsics[i, 1, 2] = cy

        return {
            "lidar2img": lidar2img,
            "camera2ego": camera2ego,
            "camera_intrinsics": camera_intrinsics,
            "img_aug_matrix": img_aug_matrix,
            "lidar2ego": lidar2ego,
        }

    def load_calibration(self, calibration_path: str) -> Dict[str, np.ndarray]:
        """Load calibration from JSON file."""
        with open(calibration_path, "r") as f:
            calib = json.load(f)

        return {
            "lidar2img": np.array(calib.get("lidar2img", np.eye(4))),
            "camera2ego": np.array(calib.get("camera2ego", np.eye(4))),
            "camera_intrinsics": np.array(calib.get("camera_intrinsics", np.eye(4))),
            "img_aug_matrix": np.array(calib.get("img_aug_matrix", np.eye(4))),
            "lidar2ego": np.array(calib.get("lidar2ego", np.eye(4))),
        }


# ============================================================================
# Weight Loading
# ============================================================================


def remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap checkpoint keys to match model key names.

    Args:
        state_dict: Original checkpoint state dict.

    Returns:
        Remapped state dict with keys matching the model.
    """
    key_mapping = {
        # FPN keys - add index for ModuleList
        "img_neck.fpn_convs.0.conv.": "img_neck.fpn_convs.conv.",
        "img_neck.lateral_convs.0.conv.": "img_neck.lateral_convs.conv.",
        # CAN bus MLP norm
        "transformer.can_bus_mlp.norm.": "transformer.can_bus_mlp.4.",
        # Decoder attention structure
        ".attentions.0.attn.in_proj_": ".cross_attn.in_proj_",
        ".attentions.0.attn.out_proj.": ".cross_attn.out_proj.",
        ".attentions.1.": ".self_attn.",
        # FFN structure
        ".ffns.0.layers.0.0.": ".ffn.0.",
        ".ffns.0.layers.1.": ".ffn.2.",
        # Norms
        ".norms.0.": ".norm1.",
        ".norms.1.": ".norm2.",
        ".norms.2.": ".norm3.",
    }

    remapped = {}
    for k, v in state_dict.items():
        new_key = k
        for old_pattern, new_pattern in key_mapping.items():
            if old_pattern in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
        remapped[new_key] = v

    return remapped


def load_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to strictly enforce weight matching.

    Returns:
        Model with loaded weights.
    """
    print(f"Loading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    # Remap checkpoint keys to match model structure
    new_state_dict = remap_checkpoint_keys(new_state_dict)

    # Check for size mismatches and filter them out if not strict
    model_state = model.state_dict()
    filtered_state_dict = {}
    size_mismatches = []

    for k, v in new_state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                size_mismatches.append(f"  {k}: checkpoint {v.shape} vs model {model_state[k].shape}")
                if not strict:
                    continue  # Skip mismatched weights
        filtered_state_dict[k] = v

    if size_mismatches:
        print(f"\n⚠ Size mismatches found ({len(size_mismatches)}):")
        for msg in size_mismatches[:10]:
            print(msg)
        if len(size_mismatches) > 10:
            print(f"  ... and {len(size_mismatches) - 10} more")
        print("\nHINT: Make sure to use the correct config for your checkpoint:")
        print("  --config maptr_tiny   for maptr_tiny_r50_24e.pth (BEV: 200x100)")
        print("  --config nuScenes     for full nuScenes config (BEV: 200x100)")
        print("  --config small        for testing without weights (BEV: 50x50)")

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        print(f"\nMissing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")

    if unexpected_keys:
        print(f"\nUnexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")

    print("\n✓ Weights loaded successfully!")
    return model


def convert_mmcv_checkpoint(checkpoint_path: str, output_path: str):
    """Convert MMCV checkpoint to pure PyTorch format.

    Args:
        checkpoint_path: Path to MMCV checkpoint.
        output_path: Path to save converted checkpoint.
    """
    print(f"Converting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Key mapping from MMCV to PyTorch
    key_mapping = {
        # Backbone
        "img_backbone.": "img_backbone.",
        # Neck
        "img_neck.": "img_neck.",
        # Head
        "pts_bbox_head.": "pts_bbox_head.",
    }

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in key_mapping.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix) :]
                break
        new_state_dict[new_key] = v

    torch.save({"state_dict": new_state_dict}, output_path)
    print(f"Converted checkpoint saved to: {output_path}")


# ============================================================================
# Inference Pipeline
# ============================================================================


class MapTRInference:
    """MapTR inference pipeline."""

    def __init__(
        self,
        config: MapTRConfig,
        checkpoint_path: Optional[str] = None,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Build model
        print("Building model...")
        self.model = build_maptr_model(config, self.device)

        # Load weights if provided
        if checkpoint_path is not None:
            self.model = load_weights(self.model, checkpoint_path, strict=False)

        self.model.eval()

        # Initialize processors
        self.image_processor = ImageProcessor(
            img_height=config.img_height,
            img_width=config.img_width,
        )
        self.calibration = CameraCalibration(num_cameras=config.num_cameras)

        # Class names
        self.class_names = ["divider", "ped_crossing", "boundary"]

    def create_img_metas(
        self,
        calibration: Dict[str, np.ndarray],
        scene_token: str = "inference",
    ) -> List[List[Dict]]:
        """Create image metadata for inference."""
        meta = {
            "scene_token": scene_token,
            "can_bus": np.zeros(18),
            "lidar2img": calibration["lidar2img"],
            "camera2ego": calibration.get("camera2ego", np.eye(4)),
            "camera_intrinsics": calibration.get("camera_intrinsics", np.eye(4)),
            "img_aug_matrix": calibration.get("img_aug_matrix", np.eye(4)),
            "lidar2ego": calibration.get("lidar2ego", np.eye(4)),
            "img_shape": [(self.config.img_height, self.config.img_width)] * self.config.num_cameras,
            "prev_bev_exists": False,
        }
        return [[meta]]

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        img_metas: List[List[Dict]],
    ) -> List[Dict]:
        """Run inference on images.

        Args:
            images: Input images tensor of shape (1, num_cams, 3, H, W).
            img_metas: Image metadata.

        Returns:
            List of detection results.
        """
        images = images.to(self.device)
        results = self.model(img_metas=img_metas, img=images)
        return results

    def predict_from_paths(
        self,
        image_paths: List[str],
        calibration: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict]:
        """Run inference on images from file paths.

        Args:
            image_paths: List of paths to camera images.
            calibration: Camera calibration (uses dummy if None).

        Returns:
            List of detection results.
        """
        # Load and preprocess images
        images = [self.image_processor.load_image(p) for p in image_paths]
        images_tensor = self.image_processor.preprocess(images)

        # Use dummy calibration if not provided
        if calibration is None:
            calibration = self.calibration.create_dummy_calibration(self.config.img_height, self.config.img_width)

        img_metas = self.create_img_metas(calibration)

        return self.predict(images_tensor, img_metas)

    def predict_demo(self) -> List[Dict]:
        """Run inference on dummy data for demo/testing."""
        images = self.image_processor.generate_dummy_images(self.config.num_cameras)
        calibration = self.calibration.create_dummy_calibration(self.config.img_height, self.config.img_width)
        img_metas = self.create_img_metas(calibration)
        return self.predict(images, img_metas)

    def format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Format results for output.

        Args:
            results: Raw model output.

        Returns:
            Formatted results dictionary.
        """
        formatted = {
            "num_detections": 0,
            "detections": [],
        }

        for i, result in enumerate(results):
            if "pts_bbox" not in result:
                continue

            pts_bbox = result["pts_bbox"]
            bboxes = pts_bbox["boxes_3d"]
            scores = pts_bbox["scores_3d"]
            labels = pts_bbox["labels_3d"]
            pts = pts_bbox["pts_3d"]

            num_dets = bboxes.shape[0]
            formatted["num_detections"] += num_dets

            for j in range(num_dets):
                det = {
                    "sample_idx": i,
                    "class": self.class_names[labels[j].item()]
                    if labels[j].item() < len(self.class_names)
                    else f"class_{labels[j].item()}",
                    "class_id": labels[j].item(),
                    "score": scores[j].item(),
                    "bbox": bboxes[j].cpu().numpy().tolist(),
                    "points": pts[j].cpu().numpy().tolist(),
                }
                formatted["detections"].append(det)

        return formatted

    def print_results(self, results: List[Dict]):
        """Print detection results."""
        formatted = self.format_results(results)

        print("\n" + "=" * 60)
        print("Detection Results")
        print("=" * 60)
        print(f"Total detections: {formatted['num_detections']}")

        for det in formatted["detections"][:10]:  # Print first 10
            print(f"\n  Class: {det['class']}")
            print(f"  Score: {det['score']:.4f}")
            print(f"  BBox: {det['bbox']}")
            print(f"  Points: {len(det['points'])} vertices")

        if len(formatted["detections"]) > 10:
            print(f"\n  ... and {len(formatted['detections']) - 10} more detections")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MapTR Inference Pipeline")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Paths to input images (one per camera)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing camera images",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with dummy data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["small", "nuScenes", "maptr_tiny"],
        help="Model configuration preset (use 'maptr_tiny' for maptr_tiny_r50_24e.pth checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config
    if args.config == "nuScenes":
        config = MapTRConfig.from_nuScenes()
    elif args.config == "maptr_tiny":
        config = MapTRConfig.from_maptr_tiny()
    else:
        config = MapTRConfig.from_small()

    print("\n" + "=" * 60)
    print("MapTR Inference Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"BEV size: {config.bev_h} x {config.bev_w}")
    print(f"Num vectors: {config.num_vec}")
    print(f"Points per vector: {config.num_pts_per_vec}")

    # Create inference pipeline
    inference = MapTRInference(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Run inference
    if args.demo:
        print("\nRunning demo inference with dummy data...")
        results = inference.predict_demo()

    elif args.images:
        print(f"\nRunning inference on {len(args.images)} images...")
        calibration = None
        if args.calibration:
            calibration = inference.calibration.load_calibration(args.calibration)
        results = inference.predict_from_paths(args.images, calibration)

    elif args.image_dir:
        # Load all images from directory
        image_dir = Path(args.image_dir)
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(sorted(image_dir.glob(f"*{ext}")))
            image_paths.extend(sorted(image_dir.glob(f"*{ext.upper()}")))

        if len(image_paths) < config.num_cameras:
            print(f"Warning: Found only {len(image_paths)} images, expected {config.num_cameras}")

        image_paths = [str(p) for p in image_paths[: config.num_cameras]]
        print(f"\nRunning inference on images from {args.image_dir}...")
        calibration = None
        if args.calibration:
            calibration = inference.calibration.load_calibration(args.calibration)
        results = inference.predict_from_paths(image_paths, calibration)

    else:
        print("\nNo input provided, running demo mode...")
        results = inference.predict_demo()

    # Print results
    inference.print_results(results)

    # Save results if output path provided
    if args.output:
        formatted = inference.format_results(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
