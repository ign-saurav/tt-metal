# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MapTR Inference and Visualization Pipeline

Based on official MapTR repository: https://github.com/hustvl/MapTR

Usage:
    # Demo mode:
    python models/experimental/mapTR/inference/run_inference.py --demo

    # With nuScenes mini dataset:
    python models/experimental/mapTR/inference/run_inference.py \
        --checkpoint models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth \
        --nuscenes models/experimental/mapTR/resources/data/nuscenes \
        --show-dir ./work_dirs/vis_pred \
        --score-thresh 0.3
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import os.path as osp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR components from reference folder
from models.experimental.mapTR.reference.pytorch_resnet import ResNet, Bottleneck
from models.experimental.mapTR.reference.pytorch_fpn import FPN
from models.experimental.mapTR.reference.pytorch_positional_encoding import LearnedPositionalEncoding
from models.experimental.mapTR.reference.pytorch_maptr_head import MapTRHead
from models.experimental.mapTR.reference.pytorch_maptr import MapTR
from models.experimental.mapTR.reference.pytorch_bevformer_encoder import BEVFormerEncoder
from models.experimental.mapTR.reference.pytorch_transformer import MapTRPerceptionTransformer
from models.experimental.mapTR.reference.modules.decoder import MapDetectionTransformerDecoder


# ============================================================================
# Constants
# ============================================================================

# Camera order matching nuScenes/MapTR
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

COLORS_PLT = {
    0: "orange",  # divider
    1: "blue",  # ped_crossing
    2: "green",  # boundary
}

CLASS_NAMES = ["divider", "ped_crossing", "boundary"]


# ============================================================================
# Configuration
# ============================================================================


class MapTRConfig:
    """Configuration for MapTR model."""

    def __init__(
        self,
        img_height: int = 450,  # Default: 900 * 0.5 scale
        img_width: int = 800,  # Default: 1600 * 0.5 scale
        num_cameras: int = 6,
        embed_dims: int = 256,
        num_classes: int = 3,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        bev_h: int = 200,
        bev_w: int = 100,
        pc_range: List[float] = None,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_channels: int = 512,
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
    def from_maptr_tiny(cls):
        """Config for MapTR Tiny R50 checkpoint.

        Matches official config: maptr_tiny_r50_24e_bevformer.py
        https://github.com/hustvl/MapTR/blob/main/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py

        Key settings:
        - Original img_scale: (1600, 900)
        - RandomScaleImageMultiViewImage scales=[0.5] -> actual input: 800x450
        - point_cloud_range: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        - bev_h=200, bev_w=100
        - num_vec=50, num_pts_per_vec=20
        """
        return cls(
            img_height=450,  # 900 * 0.5 (RandomScaleImageMultiViewImage scales=[0.5])
            img_width=800,  # 1600 * 0.5
            num_cameras=6,
            num_classes=3,  # divider, ped_crossing, boundary
            num_vec=50,
            num_pts_per_vec=20,
            bev_h=200,
            bev_w=100,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_encoder_layers=1,
            num_decoder_layers=6,
            num_heads=8,
            feedforward_channels=512,  # _ffn_dim_ = _dim_ * 2
        )


# ============================================================================
# nuScenes Data Loader
# ============================================================================


class NuScenesLoader:
    """Load nuScenes data with proper calibration."""

    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-mini",
        img_height: int = 450,  # 900 * 0.5 scale
        img_width: int = 800,  # 1600 * 0.5 scale
    ):
        self.data_root = Path(data_root)
        self.version = version
        self.img_height = img_height
        self.img_width = img_width

        # Image normalization (ImageNet)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        # Original nuScenes image size
        self.orig_h = 900
        self.orig_w = 1600

        # Load nuScenes metadata
        self.nusc = None
        self._load_nuscenes()

    def _load_nuscenes(self):
        """Load nuScenes database."""
        try:
            from nuscenes.nuscenes import NuScenes

            self.nusc = NuScenes(
                version=self.version,
                dataroot=str(self.data_root),
                verbose=True,
            )
            print(f"✓ Loaded nuScenes {self.version}: {len(self.nusc.sample)} samples")
        except ImportError:
            print("⚠ nuscenes-devkit not installed. Install: pip install nuscenes-devkit")
            self._load_from_json()
        except Exception as e:
            print(f"⚠ Could not load nuScenes: {e}")
            self._load_from_json()

    def _load_from_json(self):
        """Fallback: Load metadata from JSON files."""
        print("Loading from JSON files...")
        meta_dir = self.data_root / self.version

        self.samples = []
        self.sample_data = {}
        self.calibrated_sensors = {}
        self.ego_poses = {}

        # Load samples
        sample_file = meta_dir / "sample.json"
        if sample_file.exists():
            with open(sample_file) as f:
                self.samples = json.load(f)
            print(f"  Loaded {len(self.samples)} samples")

        # Load sample_data
        sample_data_file = meta_dir / "sample_data.json"
        if sample_data_file.exists():
            with open(sample_data_file) as f:
                for sd in json.load(f):
                    self.sample_data[sd["token"]] = sd

        # Load calibrated_sensor
        calib_file = meta_dir / "calibrated_sensor.json"
        if calib_file.exists():
            with open(calib_file) as f:
                for cs in json.load(f):
                    self.calibrated_sensors[cs["token"]] = cs

        # Load ego_pose
        ego_file = meta_dir / "ego_pose.json"
        if ego_file.exists():
            with open(ego_file) as f:
                for ep in json.load(f):
                    self.ego_poses[ep["token"]] = ep

    def _quat_to_rot(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    def get_sample_list(self) -> List[Dict]:
        """Get list of all samples."""
        if self.nusc is not None:
            return [{"token": s["token"], "scene_token": s["scene_token"]} for s in self.nusc.sample]
        return [{"token": s["token"], "scene_token": s.get("scene_token", "")} for s in self.samples]

    def get_lidar2img(self, sample_token: str) -> np.ndarray:
        """Compute lidar2img transformation matrices for all cameras."""
        lidar2img_list = []

        if self.nusc is not None:
            sample = self.nusc.get("sample", sample_token)

            # Get LIDAR calibration
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            lidar_ego = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

            # lidar2ego
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = self._quat_to_rot(lidar_calib["rotation"])
            lidar2ego[:3, 3] = lidar_calib["translation"]

            # ego2global (at lidar timestamp)
            ego2global_lidar = np.eye(4)
            ego2global_lidar[:3, :3] = self._quat_to_rot(lidar_ego["rotation"])
            ego2global_lidar[:3, 3] = lidar_ego["translation"]

            for cam_name in CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
                cam_ego = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

                # Camera intrinsics
                intrinsic = np.eye(4)
                cam_intrinsic = np.array(cam_calib["camera_intrinsic"])

                # Scale intrinsics for resized images
                scale_w = self.img_width / self.orig_w
                scale_h = self.img_height / self.orig_h
                cam_intrinsic[0, :] *= scale_w
                cam_intrinsic[1, :] *= scale_h
                intrinsic[:3, :3] = cam_intrinsic

                # cam2ego
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = self._quat_to_rot(cam_calib["rotation"])
                cam2ego[:3, 3] = cam_calib["translation"]

                # ego2global (at camera timestamp)
                ego2global_cam = np.eye(4)
                ego2global_cam[:3, :3] = self._quat_to_rot(cam_ego["rotation"])
                ego2global_cam[:3, 3] = cam_ego["translation"]

                # Compute full transformation: lidar -> ego -> global -> ego_cam -> cam -> image
                global2ego_cam = np.linalg.inv(ego2global_cam)
                ego2cam = np.linalg.inv(cam2ego)

                lidar2global = ego2global_lidar @ lidar2ego
                global2cam = ego2cam @ global2ego_cam
                lidar2cam = global2cam @ lidar2global
                lidar2img = intrinsic @ lidar2cam

                lidar2img_list.append(lidar2img)
        else:
            # Fallback: identity matrices
            for _ in CAMERA_NAMES:
                lidar2img_list.append(np.eye(4))

        return np.stack(lidar2img_list, axis=0).astype(np.float32)

    def load_images(self, sample_token: str) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """Load all camera images for a sample."""
        images = []
        cam_images = {}

        if self.nusc is not None:
            sample = self.nusc.get("sample", sample_token)

            for cam_name in CAMERA_NAMES:
                cam_token = sample["data"][cam_name]
                cam_data = self.nusc.get("sample_data", cam_token)
                img_path = self.data_root / cam_data["filename"]

                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                img_array = np.array(img_resized)
                images.append(img_array)
                cam_images[cam_name] = img_array
        else:
            # Fallback: load from samples directory
            samples_dir = self.data_root / "samples"
            for cam_name in CAMERA_NAMES:
                cam_dir = samples_dir / cam_name
                if cam_dir.exists():
                    img_files = sorted(cam_dir.glob("*.jpg"))
                    if img_files:
                        img = Image.open(img_files[0]).convert("RGB")
                        img_resized = img.resize((self.img_width, self.img_height), Image.BILINEAR)
                        img_array = np.array(img_resized)
                        images.append(img_array)
                        cam_images[cam_name] = img_array
                    else:
                        images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                else:
                    images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))

        return images, cam_images

    def preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess images for model input."""
        processed = []
        for img in images:
            img = img.astype(np.float32)
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)
            processed.append(img)

        imgs = np.stack(processed, axis=0)
        imgs = imgs[np.newaxis, ...]
        return torch.from_numpy(imgs).float()

    def create_img_metas(self, sample_token: str) -> List[Dict]:
        """Create image metadata for inference."""
        lidar2img = self.get_lidar2img(sample_token)

        meta = {
            "sample_token": sample_token,
            "can_bus": np.zeros(18, dtype=np.float32),
            "lidar2img": lidar2img,
            "img_shape": [(self.img_height, self.img_width)] * len(CAMERA_NAMES),
        }
        return [meta]

    def load_sample(self, sample_token: str) -> Tuple[torch.Tensor, List[Dict], Dict[str, np.ndarray]]:
        """Load a complete sample for inference."""
        images, cam_images = self.load_images(sample_token)
        images_tensor = self.preprocess_images(images)
        img_metas = self.create_img_metas(sample_token)
        return images_tensor, img_metas, cam_images


# ============================================================================
# Model Builder
# ============================================================================


def build_maptr_model(config: MapTRConfig, device: torch.device = None) -> MapTR:
    """Build MapTR model using reference implementations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backbone
    backbone = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        out_indices=(3,),
    )

    # FPN
    fpn = FPN(
        in_channels=[2048],
        out_channels=config.fpn_out_channels,
        num_outs=1,
    )

    # Encoder
    encoder = BEVFormerEncoder(
        num_layers=config.num_encoder_layers,
        pc_range=config.pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=config.embed_dims,
        num_heads=4,
        feedforward_channels=config.feedforward_channels,
        ffn_dropout=0.1,
    )

    # Decoder
    decoder = MapDetectionTransformerDecoder(
        num_layers=config.num_decoder_layers,
        embed_dim=config.embed_dims,
        num_heads=config.num_heads,
    )

    # Transformer
    transformer = MapTRPerceptionTransformer(
        encoder=encoder,
        decoder=decoder,
        embed_dims=config.embed_dims,
        num_feature_levels=4,
        num_cams=config.num_cameras,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )

    # Positional Encoding
    positional_encoding = LearnedPositionalEncoding(
        num_feats=config.embed_dims // 2,
        row_num_embed=config.bev_h,
        col_num_embed=config.bev_w,
    )

    # Head
    head = MapTRHead(
        transformer=transformer,
        positional_encoding=positional_encoding,
        embed_dims=config.embed_dims,
        num_classes=config.num_classes,
        num_reg_fcs=2,
        num_cls_fcs=2,
        code_size=2,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        query_embed_type="instance_pts",
        transform_method="minmax",
        with_box_refine=True,
    )

    # Full model
    model = MapTR(
        img_backbone=backbone,
        img_neck=fpn,
        pts_bbox_head=head,
        use_grid_mask=False,
        video_test_mode=False,
    )

    return model.to(device)


def load_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from checkpoint."""
    print(f"Loading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"✓ Loaded {len(new_state_dict)} weights")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


# ============================================================================
# Ground Truth Loader (nuScenes Map API)
# ============================================================================


class MapAnnotationLoader:
    """Load ground truth map annotations from nuScenes.

    Based on: https://github.com/hustvl/MapTR/blob/main/tools/maptr/vis_pred.py
    """

    # Map classes in MapTR
    MAP_CLASSES = ["divider", "ped_crossing", "boundary"]

    # nuScenes layer names for each class
    LAYER_NAMES = {
        "divider": ["road_divider", "lane_divider"],
        "ped_crossing": ["ped_crossing"],
        "boundary": ["road_segment", "lane"],
    }

    def __init__(
        self,
        data_root: str,
        map_root: str = None,
        pc_range: List[float] = None,
        fixed_num_pts: int = 20,
    ):
        self.data_root = data_root
        self.map_root = map_root or osp.join(data_root, "maps")
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.fixed_num_pts = fixed_num_pts

        # Try to load nuScenes map expansion
        self.nusc_maps = {}
        self._load_maps()

    def _load_maps(self):
        """Load nuScenes maps."""
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap

            # Map locations in nuScenes
            map_locations = [
                "singapore-onenorth",
                "singapore-hollandvillage",
                "singapore-queenstown",
                "boston-seaport",
            ]

            # Check if map expansion JSON files exist
            maps_found = []
            for loc in map_locations:
                map_path = osp.join(self.map_root, f"{loc}.json")
                if osp.exists(map_path):
                    maps_found.append(loc)

            if not maps_found:
                print("⚠ nuScenes map expansion files not found in maps/ folder")
                print("  GT visualization requires downloading map expansion:")
                print("  https://www.nuscenes.org/nuscenes#download")
                print("  Download 'Map expansion pack (v1.3)' and extract to:")
                print(f"  {self.map_root}/")
                print("  Expected files: singapore-onenorth.json, boston-seaport.json, etc.")
                return

            for loc in maps_found:
                try:
                    self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
                except Exception as e:
                    print(f"⚠ Failed to load map {loc}: {e}")

            if self.nusc_maps:
                print(f"✓ Loaded {len(self.nusc_maps)} nuScenes maps: {list(self.nusc_maps.keys())}")
        except ImportError:
            print("⚠ nuscenes-devkit not installed, GT maps unavailable")
            print("  Install with: pip install nuscenes-devkit")

    def get_map_for_location(self, location: str):
        """Get map for a specific location."""
        for loc_name, nusc_map in self.nusc_maps.items():
            if loc_name in location.lower() or location.lower() in loc_name:
                return nusc_map
        return None

    def sample_points_from_line(self, line, num_points: int) -> np.ndarray:
        """Sample fixed number of points from a line."""
        try:
            from shapely.geometry import LineString

            if not isinstance(line, LineString):
                line = LineString(line)

            if line.length == 0:
                return np.zeros((num_points, 2))

            distances = np.linspace(0, line.length, num_points)
            points = [line.interpolate(d) for d in distances]
            return np.array([[p.x, p.y] for p in points])
        except ImportError:
            # Fallback: simple linear interpolation
            coords = np.array(line)
            if len(coords) < 2:
                return np.zeros((num_points, 2))

            total_dist = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
            if total_dist == 0:
                return np.tile(coords[0], (num_points, 1))

            indices = np.linspace(0, len(coords) - 1, num_points)
            return np.array(
                [
                    coords[int(i)] + (coords[min(int(i) + 1, len(coords) - 1)] - coords[int(i)]) * (i - int(i))
                    for i in indices
                ]
            )

    def get_ground_truth(
        self,
        nusc,
        sample_token: str,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Get ground truth map annotations for a sample.

        Returns:
            gt_points: List of (N, 2) arrays for each map element
            gt_labels: List of class labels
        """
        gt_points = []
        gt_labels = []

        if not self.nusc_maps:
            return gt_points, gt_labels

        try:
            # Get sample info
            sample = nusc.get("sample", sample_token)

            # Get ego pose from LIDAR_TOP
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = nusc.get("sample_data", lidar_token)
            ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

            # Get map location
            scene = nusc.get("scene", sample["scene_token"])
            log = nusc.get("log", scene["log_token"])
            location = log["location"]

            nusc_map = self.get_map_for_location(location)
            if nusc_map is None:
                return gt_points, gt_labels

            # Ego position
            ego_x, ego_y = ego_pose["translation"][:2]
            ego_rotation = ego_pose["rotation"]

            # Convert quaternion to yaw
            from pyquaternion import Quaternion

            q = Quaternion(ego_rotation)
            yaw = q.yaw_pitch_roll[0]

            # Define patch around ego vehicle (larger to capture more elements)
            patch_box = (ego_x, ego_y, 80, 50)  # (x, y, height, width)

            # Get map elements
            for class_idx, class_name in enumerate(self.MAP_CLASSES):
                layer_names = self.LAYER_NAMES.get(class_name, [])

                for layer_name in layer_names:
                    try:
                        records = nusc_map.get_records_in_patch(patch_box, [layer_name], mode="intersect")

                        for record_token in records.get(layer_name, []):
                            record = nusc_map.get(layer_name, record_token)

                            # Get line geometry
                            if layer_name in ["road_divider", "lane_divider"]:
                                line_token = record.get("line_token")
                                if line_token:
                                    line = nusc_map.get("line", line_token)
                                    node_tokens = line.get("node_tokens", [])
                                    if len(node_tokens) >= 2:
                                        # Get coordinates using x, y keys
                                        nodes = []
                                        for t in node_tokens:
                                            node = nusc_map.get("node", t)
                                            nodes.append([node["x"], node["y"]])
                                        nodes = np.array(nodes)
                                        # Transform to ego frame
                                        nodes_ego = self._transform_to_ego(nodes, ego_x, ego_y, yaw)
                                        # Filter to PC range
                                        if self._in_range(nodes_ego):
                                            pts = self.sample_points_from_line(nodes_ego, self.fixed_num_pts)
                                            gt_points.append(pts)
                                            gt_labels.append(class_idx)

                            elif layer_name == "ped_crossing":
                                polygon_token = record.get("polygon_token")
                                if polygon_token:
                                    # Use extract_polygon for cleaner API
                                    try:
                                        poly = nusc_map.extract_polygon(polygon_token)
                                        nodes = np.array(poly.exterior.coords)[:-1]  # Remove duplicate last point
                                        if len(nodes) >= 2:
                                            nodes_ego = self._transform_to_ego(nodes, ego_x, ego_y, yaw)
                                            if self._in_range(nodes_ego):
                                                pts = self.sample_points_from_line(nodes_ego, self.fixed_num_pts)
                                                gt_points.append(pts)
                                                gt_labels.append(class_idx)
                                    except:
                                        pass

                            elif layer_name in ["road_segment", "lane"]:
                                polygon_token = record.get("polygon_token")
                                if polygon_token:
                                    try:
                                        poly = nusc_map.extract_polygon(polygon_token)
                                        nodes = np.array(poly.exterior.coords)[:-1]
                                        if len(nodes) >= 2:
                                            nodes_ego = self._transform_to_ego(nodes, ego_x, ego_y, yaw)
                                            if self._in_range(nodes_ego):
                                                pts = self.sample_points_from_line(nodes_ego, self.fixed_num_pts)
                                                gt_points.append(pts)
                                                gt_labels.append(class_idx)
                                    except:
                                        pass
                    except Exception as e:
                        continue

        except Exception as e:
            print(f"⚠ Error loading GT: {e}")
            import traceback

            traceback.print_exc()

        return gt_points, gt_labels

    def _transform_to_ego(self, points: np.ndarray, ego_x: float, ego_y: float, yaw: float) -> np.ndarray:
        """Transform points from global to ego frame."""
        # Translate
        points = points - np.array([ego_x, ego_y])

        # Rotate (inverse of ego rotation)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        points = points @ rotation_matrix.T

        return points

    def _in_range(self, points: np.ndarray) -> bool:
        """Check if any point is within PC range."""
        x_range = (self.pc_range[0], self.pc_range[3])
        y_range = (self.pc_range[1], self.pc_range[4])

        x_in = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        y_in = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])

        return np.any(x_in & y_in)


# ============================================================================
# Visualization
# ============================================================================


class MapTRVisualizer:
    """Visualizer for MapTR predictions and ground truth.

    Based on: https://github.com/hustvl/MapTR/blob/main/tools/maptr/vis_pred.py
    """

    def __init__(self, pc_range: List[float], car_img_path: str = None):
        self.pc_range = pc_range
        self.car_img = None

        # Try to load car icon for visualization
        if car_img_path and osp.exists(car_img_path):
            try:
                self.car_img = Image.open(car_img_path)
            except:
                pass

    def _create_bev_figure(self, title: str = None):
        """Create BEV figure with proper settings (matching original MapTR)."""
        import matplotlib.pyplot as plt

        # Match original: figsize=(2, 4) for aspect ratio
        fig = plt.figure(figsize=(2, 4))
        plt.xlim(self.pc_range[0], self.pc_range[3])
        plt.ylim(self.pc_range[1], self.pc_range[4])
        plt.axis("off")

        return fig

    def visualize_ground_truth(
        self,
        gt_points: List[np.ndarray],
        gt_labels: List[int],
        output_path: str,
    ):
        """Visualize ground truth map elements.

        Matches original MapTR: GT_fixednum_pts_MAP.png
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig = self._create_bev_figure()

        for pts, label in zip(gt_points, gt_labels):
            color = COLORS_PLT.get(label, "gray")
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=1, alpha=0.8, zorder=-1)
            plt.scatter(x, y, color=color, s=2, alpha=0.8, zorder=-1)

        # Draw car icon
        if self.car_img is not None:
            plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        else:
            # Draw simple ego vehicle rectangle
            ego_rect = plt.Rectangle(
                (-0.8, -1.5), 1.6, 3.0, fill=True, facecolor="gray", edgecolor="black", linewidth=1, zorder=10
            )
            plt.gca().add_patch(ego_rect)

        plt.savefig(output_path, bbox_inches="tight", format="png", dpi=1200)
        plt.close()
        print(f"✓ Saved GT: {output_path}")

    def visualize_predictions(
        self,
        results: Dict,
        output_path: str,
        score_thresh: float = 0.3,
        title: str = "MapTR Prediction",
    ):
        """Visualize predictions in BEV.

        Matches original MapTR: PRED_MAP_plot.png
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        fig = self._create_bev_figure()

        # Extract predictions
        bboxes = results.get("boxes_3d", results.get("bboxes"))
        scores = results.get("scores_3d", results.get("scores"))
        labels = results.get("labels_3d", results.get("labels"))
        pts = results.get("pts_3d", results.get("pts"))

        if scores is None:
            plt.close()
            return

        # Convert to numpy
        if torch.is_tensor(bboxes):
            bboxes = bboxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if pts is not None and torch.is_tensor(pts):
            pts = pts.cpu().numpy()

        keep = scores > score_thresh

        # Draw predictions (matching original style)
        for pred_score, pred_bbox, pred_label, pred_pts in zip(
            scores[keep], bboxes[keep], labels[keep], pts[keep] if pts is not None else [None] * keep.sum()
        ):
            label_int = int(pred_label)
            color = COLORS_PLT.get(label_int, "gray")

            if pred_pts is not None:
                pts_x = pred_pts[:, 0]
                pts_y = pred_pts[:, 1]
                plt.plot(pts_x, pts_y, color=color, linewidth=1, alpha=0.8, zorder=-1)
                plt.scatter(pts_x, pts_y, color=color, s=1, alpha=0.8, zorder=-1)

        # Draw car icon
        if self.car_img is not None:
            plt.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        else:
            ego_rect = plt.Rectangle(
                (-0.8, -1.5), 1.6, 3.0, fill=True, facecolor="gray", edgecolor="black", linewidth=1, zorder=10
            )
            plt.gca().add_patch(ego_rect)

        plt.savefig(output_path, bbox_inches="tight", format="png", dpi=1200)
        plt.close()
        print(f"✓ Saved Pred: {output_path}")

    def visualize_comparison(
        self,
        gt_points: List[np.ndarray],
        gt_labels: List[int],
        results: Dict,
        output_path: str,
        score_thresh: float = 0.3,
        title: str = "GT vs Prediction",
    ):
        """Create side-by-side GT and prediction visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 12))

        pc_range = self.pc_range

        # === Ground Truth (left) ===
        ax_gt = axes[0]
        ax_gt.set_xlim(pc_range[0], pc_range[3])
        ax_gt.set_ylim(pc_range[1], pc_range[4])
        ax_gt.set_aspect("equal")
        ax_gt.set_facecolor("#f5f5f5")
        ax_gt.grid(True, alpha=0.3, linestyle="--")

        gt_counts = {0: 0, 1: 0, 2: 0}
        for pts, label in zip(gt_points, gt_labels):
            color = COLORS_PLT.get(label, "gray")
            gt_counts[label] = gt_counts.get(label, 0) + 1
            x = pts[:, 0]
            y = pts[:, 1]
            ax_gt.plot(x, y, color=color, linewidth=2, alpha=0.8)
            ax_gt.scatter(x, y, color=color, s=8, alpha=0.8)

        # Ego vehicle
        ego_rect = plt.Rectangle(
            (-0.8, -1.5), 1.6, 3.0, fill=True, facecolor="yellow", edgecolor="black", linewidth=2, zorder=10
        )
        ax_gt.add_patch(ego_rect)
        ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")

        # Legend for GT
        legend_handles = []
        for label_id, color in COLORS_PLT.items():
            count = gt_counts.get(label_id, 0)
            patch = mpatches.Patch(color=color, label=f"{CLASS_NAMES[label_id]} ({count})")
            legend_handles.append(patch)
        ax_gt.legend(handles=legend_handles, loc="upper right", fontsize=8)

        # === Predictions (right) ===
        ax_pred = axes[1]
        ax_pred.set_xlim(pc_range[0], pc_range[3])
        ax_pred.set_ylim(pc_range[1], pc_range[4])
        ax_pred.set_aspect("equal")
        ax_pred.set_facecolor("#f5f5f5")
        ax_pred.grid(True, alpha=0.3, linestyle="--")

        # Extract predictions
        scores = results.get("scores_3d", results.get("scores"))
        labels = results.get("labels_3d", results.get("labels"))
        pts = results.get("pts_3d", results.get("pts"))

        pred_counts = {0: 0, 1: 0, 2: 0}

        if scores is not None:
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if pts is not None and torch.is_tensor(pts):
                pts = pts.cpu().numpy()

            keep = scores > score_thresh

            for pred_label, pred_pts in zip(labels[keep], pts[keep] if pts is not None else [None] * keep.sum()):
                label_int = int(pred_label)
                color = COLORS_PLT.get(label_int, "gray")
                pred_counts[label_int] = pred_counts.get(label_int, 0) + 1

                if pred_pts is not None:
                    ax_pred.plot(pred_pts[:, 0], pred_pts[:, 1], color=color, linewidth=2, alpha=0.8)
                    ax_pred.scatter(pred_pts[:, 0], pred_pts[:, 1], color=color, s=8, alpha=0.8)

        # Ego vehicle
        ego_rect2 = plt.Rectangle(
            (-0.8, -1.5), 1.6, 3.0, fill=True, facecolor="yellow", edgecolor="black", linewidth=2, zorder=10
        )
        ax_pred.add_patch(ego_rect2)
        ax_pred.set_title("Prediction", fontsize=14, fontweight="bold")

        # Legend for predictions
        legend_handles = []
        for label_id, color in COLORS_PLT.items():
            count = pred_counts.get(label_id, 0)
            patch = mpatches.Patch(color=color, label=f"{CLASS_NAMES[label_id]} ({count})")
            legend_handles.append(patch)
        ax_pred.legend(handles=legend_handles, loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved comparison: {output_path}")

    def create_camera_grid(
        self,
        cam_images: Dict[str, np.ndarray],
        output_path: str,
    ):
        """Create grid of camera images (surround view)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Top row: front cameras (matching original order)
        cam_order_top = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        cam_order_bottom = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

        for i, cam_name in enumerate(cam_order_top):
            if cam_name in cam_images:
                axes[0, i].imshow(cam_images[cam_name])
            axes[0, i].set_title(cam_name.replace("CAM_", ""), fontsize=10)
            axes[0, i].axis("off")

        for i, cam_name in enumerate(cam_order_bottom):
            if cam_name in cam_images:
                axes[1, i].imshow(cam_images[cam_name])
            axes[1, i].set_title(cam_name.replace("CAM_", ""), fontsize=10)
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {output_path}")


# ============================================================================
# Inference
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

        print(f"Device: {self.device}")
        print("Building model...")

        self.model = build_maptr_model(config, self.device)

        if checkpoint_path:
            self.model = load_weights(self.model, checkpoint_path)

        self.model.eval()
        self.visualizer = MapTRVisualizer(pc_range=config.pc_range)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, img_metas: List[Dict]) -> List[Dict]:
        """Run inference."""
        images = images.to(self.device)
        results = self.model(img_metas=[img_metas], img=images)
        return results

    def print_results(self, results: List[Dict], score_thresh: float = 0.3):
        """Print detection summary."""
        print("\n" + "=" * 50)
        print("Detection Results")
        print("=" * 50)

        for result in results:
            if "pts_bbox" not in result:
                continue

            pts_bbox = result["pts_bbox"]
            scores = pts_bbox["scores_3d"]
            labels = pts_bbox["labels_3d"]

            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            keep = scores > score_thresh
            print(f"Detections (score > {score_thresh}): {keep.sum()}")

            class_counts = {}
            for label in labels[keep]:
                class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"class_{label}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MapTR Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/experimental/mapTR/resources/data/weights/maptr_tiny_r50_24e_bevformer.pth",
    )
    parser.add_argument(
        "--nuscenes",
        type=str,
        default="models/experimental/mapTR/resources/data/nuscenes",
        help="Path to nuScenes data root",
    )
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--demo", action="store_true", help="Run demo with dummy data")
    parser.add_argument("--score-thresh", default=0.3, type=float)
    parser.add_argument("--show-dir", type=str, default="./work_dirs/vis_pred")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to process")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Config
    config = MapTRConfig.from_maptr_tiny()

    print("\n" + "=" * 60)
    print("MapTR Inference Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"nuScenes: {args.nuscenes}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score threshold: {args.score_thresh}")

    os.makedirs(args.show_dir, exist_ok=True)

    # Create inference pipeline
    inference = MapTRInference(
        config=config,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        device=device,
    )

    if args.demo:
        # Demo mode with dummy data
        print("\nRunning demo with dummy data...")
        dummy_images = torch.randn(1, 6, 3, config.img_height, config.img_width)
        dummy_metas = [
            {
                "can_bus": np.zeros(18, dtype=np.float32),
                "lidar2img": np.eye(4)[np.newaxis].repeat(6, axis=0),
                "img_shape": [(config.img_height, config.img_width)] * 6,
            }
        ]

        results = inference.predict(dummy_images, dummy_metas)
        inference.print_results(results, args.score_thresh)

        if results and "pts_bbox" in results[0]:
            output_path = osp.join(args.show_dir, "demo_pred.png")
            inference.visualizer.visualize_predictions(
                results[0]["pts_bbox"], output_path, args.score_thresh, "Demo Prediction"
            )
    else:
        # nuScenes mode
        print(f"\nLoading nuScenes from: {args.nuscenes}")
        loader = NuScenesLoader(
            data_root=args.nuscenes,
            version=args.version,
            img_height=config.img_height,
            img_width=config.img_width,
        )

        samples = loader.get_sample_list()
        if not samples:
            print("No samples found!")
            return

        # Initialize ground truth loader
        gt_loader = MapAnnotationLoader(
            data_root=args.nuscenes,
            pc_range=config.pc_range,
            fixed_num_pts=config.num_pts_per_vec,
        )

        start_idx = args.sample_idx
        end_idx = min(start_idx + args.num_samples, len(samples))
        print(f"Processing samples {start_idx} to {end_idx - 1}")

        for i in range(start_idx, end_idx):
            sample = samples[i]
            sample_token = sample["token"]
            print(f"\n--- Sample {i}: {sample_token[:20]}... ---")

            # Load sample
            images, img_metas, cam_images = loader.load_sample(sample_token)
            images = images.to(device)

            # Run inference
            results = inference.predict(images, img_metas)
            inference.print_results(results, args.score_thresh)

            # Load ground truth
            gt_points, gt_labels = gt_loader.get_ground_truth(loader.nusc, sample_token)
            print(f"Ground truth: {len(gt_points)} elements")
            if gt_labels:
                gt_counts = {}
                for label in gt_labels:
                    name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"class_{label}"
                    gt_counts[name] = gt_counts.get(name, 0) + 1
                for name, count in gt_counts.items():
                    print(f"  GT {name}: {count}")

            # Save visualizations
            sample_dir = osp.join(args.show_dir, f"sample_{i:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save surround view camera images
            cam_grid_path = osp.join(sample_dir, "surroud_view.jpg")
            inference.visualizer.create_camera_grid(cam_images, cam_grid_path)

            # Save GT map (matching original: GT_fixednum_pts_MAP.png)
            if gt_points:
                gt_path = osp.join(sample_dir, "GT_fixednum_pts_MAP.png")
                inference.visualizer.visualize_ground_truth(gt_points, gt_labels, gt_path)

            # Save prediction map (matching original: PRED_MAP_plot.png)
            if results and "pts_bbox" in results[0]:
                pred_path = osp.join(sample_dir, "PRED_MAP_plot.png")
                inference.visualizer.visualize_predictions(
                    results[0]["pts_bbox"], pred_path, args.score_thresh, f"Sample {i} Prediction"
                )

                # Save side-by-side comparison
                if gt_points:
                    comparison_path = osp.join(sample_dir, "comparison.png")
                    inference.visualizer.visualize_comparison(
                        gt_points,
                        gt_labels,
                        results[0]["pts_bbox"],
                        comparison_path,
                        args.score_thresh,
                    )

        print(f"\n✓ Results saved to: {args.show_dir}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
