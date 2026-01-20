# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import cv2
import numpy as np
import os
import sys
import time
import torch
import ttnn
from loguru import logger
from typing import Dict, List, Optional, Tuple

from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars, PointPillarsPreprocessor
from models.experimental.pointpillars.reference.pointpillars import PointPillars
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.common.utility_functions import tt2torch_tensor

# Import common utilities
from models.experimental.pointpillars.common import (
    VOXEL_SIZE,
    POINT_CLOUD_RANGE,
    MAX_NUM_POINTS,
    MAX_VOXELS,
    NCLASSES,
    LABEL2CLASSES,
    load_checkpoint,
    download_test_data,
    download_checkpoint,
)

# Import reference utilities for I/O and visualization
from models.experimental.pointpillars.reference.process import (
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    bbox3d2corners_camera,
    points_camera2image,
)
from models.experimental.pointpillars.reference.vis_o3d import vis_img_3d
from models.experimental.pointpillars.reference.io import read_points, read_calib


def point_range_filter(pts: np.ndarray, point_range: List[float] = None) -> np.ndarray:
    """Filter points that fall within the specified range."""
    if point_range is None:
        point_range = POINT_CLOUD_RANGE
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


class PointPillarsDemo:
    """Demo class for running PointPillars inference with PyTorch and TTNN."""

    def __init__(self, device: ttnn.Device):
        self.device = device
        self.torch_model = None
        self.ttnn_model = None

        # Model configuration from common
        self.nclasses = NCLASSES
        self.voxel_size = VOXEL_SIZE
        self.point_cloud_range = POINT_CLOUD_RANGE
        self.max_num_points = MAX_NUM_POINTS
        self.max_voxels = MAX_VOXELS

        # Limit range for filtering
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    def setup_models(self, ckpt_path: str):
        """Setup PyTorch model. TTNN model will be created before inference."""
        logger.info(f"Loading checkpoint from {ckpt_path}")

        # Initialize PyTorch model
        self.torch_model = PointPillars(
            nclasses=self.nclasses,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )

        # Load checkpoint
        state_dict = load_checkpoint(ckpt_path)
        if state_dict is not None:
            self.torch_model.load_state_dict(state_dict)

        self.torch_model = self.torch_model.to(dtype=torch.bfloat16)
        self.torch_model.eval()
        logger.info("PyTorch model loaded successfully")

        self.ttnn_model = None

    def setup_ttnn_model(self):
        """Setup TTNN model - called right before TTNN inference to match test flow."""
        logger.info("Setting up TTNN model...")

        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
            device=self.device,
        )

        # Create preprocessor
        self.preprocessor = PointPillarsPreprocessor(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
            parameters=parameters,
            device=self.device,
        )

        # Create TTNN model
        self.tt_model = TtPointPillars(
            nclasses=self.nclasses,
            parameters=parameters,
            device=self.device,
        )

        logger.info("TTNN model loaded successfully")

    def post_process(
        self,
        bbox_cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        bbox_dir_cls_pred: torch.Tensor,
    ) -> List[Dict]:
        """Post-process predictions using PyTorch model's built-in method."""
        bbox_cls_pred = bbox_cls_pred.float()
        bbox_pred = bbox_pred.float()
        bbox_dir_cls_pred = bbox_dir_cls_pred.float()

        # Generate anchors using model's built-in anchor generator
        batch_size = bbox_cls_pred.size(0)
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.torch_model.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        # Use model's built-in post-processing
        return self.torch_model.get_predicted_bboxes(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_anchors=batched_anchors,
        )

    def run_pytorch_inference(self, pc_torch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference using PyTorch model."""
        with torch.no_grad():
            cls_pred, reg_pred, dir_pred = self.torch_model(batched_pts=[pc_torch])
        return cls_pred, reg_pred, dir_pred

    def run_ttnn_inference(self, pc_torch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference using TTNN model."""
        with torch.no_grad():
            pillar_features = self.preprocessor.forward(batched_pts=[pc_torch])
            pillar_features = ttnn.permute(pillar_features, (0, 2, 3, 1))
            pillar_features = ttnn.reshape(
                pillar_features,
                (
                    pillar_features.shape[0],
                    1,
                    pillar_features.shape[1] * pillar_features.shape[2],
                    pillar_features.shape[3],
                ),
            )
            tt_cls, tt_reg, tt_dir = self.tt_model.forward(pillar_features)

        return tt_cls, tt_reg, tt_dir

    def visualize_results(
        self,
        result: Dict,
        calib_info: Optional[Dict],
        img: Optional[np.ndarray],
        title: str = "Detections",
        save_path: Optional[str] = None,
    ):
        """Process detection results and save visualization to file."""
        if len(result["lidar_bboxes"]) == 0:
            logger.warning(f"{title}: No detections found")
            return

        # Filter by image range if calibration and image available
        if calib_info is not None and img is not None:
            tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
            r0_rect = calib_info["R0_rect"].astype(np.float32)
            P2 = calib_info["P2"].astype(np.float32)
            image_shape = img.shape[:2]
            result = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)

        # Filter by LiDAR range
        result = keep_bbox_from_lidar_range(result, self.pcd_limit_range)

        lidar_bboxes = result["lidar_bboxes"]
        labels = result["labels"]
        scores = result["scores"]

        logger.info(f"{title}: Found {len(lidar_bboxes)} detections")
        for i, (bbox, label, score) in enumerate(zip(lidar_bboxes, labels, scores)):
            class_name = LABEL2CLASSES.get(label, "Unknown")
            logger.info(f"  [{i}] {class_name}: score={score:.3f}, pos=({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f})")

        # Save visualization to image file if calibration and image available
        if calib_info is not None and img is not None and len(result.get("camera_bboxes", [])) > 0:
            P2 = calib_info["P2"].astype(np.float32)
            camera_bboxes = result["camera_bboxes"]
            bboxes_corners = bbox3d2corners_camera(camera_bboxes)
            image_points = points_camera2image(bboxes_corners, P2)

            img_vis = img.copy()
            img_vis = vis_img_3d(img_vis, image_points, labels, rt=True)

            if save_path:
                cv2.imwrite(save_path, img_vis)
                logger.info(f"Saved image visualization to {save_path}")

    def run_demo(
        self,
        pc_path: str,
        calib_path: str = "",
        img_path: str = "",
        output_dir: str = "models/experimental/pointpillars/resources/output",
    ):
        """Run the complete demo with both PyTorch and TTNN inference."""
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found: {pc_path}")

        os.makedirs(output_dir, exist_ok=True)

        # Load point cloud
        logger.info(f"Loading point cloud from {pc_path}")
        pc = read_points(pc_path)
        pc = point_range_filter(pc)
        pc_torch = torch.from_numpy(pc).to(dtype=torch.bfloat16)
        logger.info(f"Point cloud shape: {pc.shape}")

        # Load optional inputs
        calib_info = read_calib(calib_path) if calib_path and os.path.exists(calib_path) else None
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) if img_path and os.path.exists(img_path) else None

        if calib_info:
            logger.info("Calibration loaded successfully")
        if img is not None:
            logger.info(f"Image loaded: {img.shape}")

        # Run PyTorch inference
        logger.info("\n" + "=" * 60)
        logger.info("Running PyTorch Inference")
        logger.info("=" * 60)

        start_time = time.time()
        pt_cls, pt_reg, pt_dir = self.run_pytorch_inference(pc_torch)
        pt_inference_time = time.time() - start_time
        logger.info(f"PyTorch inference time: {pt_inference_time * 1000:.2f} ms")

        # Post-process using model's built-in method
        start_time = time.time()
        pt_results = self.post_process(pt_cls, pt_reg, pt_dir)
        pt_postproc_time = time.time() - start_time
        logger.info(f"Post-processing time: {pt_postproc_time * 1000:.2f} ms")

        # Visualize
        self.visualize_results(
            result=pt_results[0],
            calib_info=calib_info,
            img=img,
            title="PyTorch Detections",
            save_path=os.path.join(output_dir, "pytorch_detections.jpg") if img is not None else None,
        )

        # Run TTNN inference
        logger.info("\n" + "=" * 60)
        logger.info("Running TTNN Inference")
        logger.info("=" * 60)

        # Setup TTNN model
        self.setup_ttnn_model()

        start_time = time.time()
        tt_cls, tt_reg, tt_dir = self.run_ttnn_inference(pc_torch)
        tt_cls = (
            tt2torch_tensor(tt_cls)
            .reshape(pt_cls.shape[0], pt_cls.shape[2], pt_cls.shape[3], pt_cls.shape[1])
            .permute(0, 3, 1, 2)
        )
        tt_reg = (
            tt2torch_tensor(tt_reg)
            .reshape(pt_reg.shape[0], pt_reg.shape[2], pt_reg.shape[3], pt_reg.shape[1])
            .permute(0, 3, 1, 2)
        )
        tt_dir = (
            tt2torch_tensor(tt_dir)
            .reshape(pt_dir.shape[0], pt_dir.shape[2], pt_dir.shape[3], pt_dir.shape[1])
            .permute(0, 3, 1, 2)
        )

        tt_inference_time = time.time() - start_time
        logger.info(f"TTNN inference time: {tt_inference_time * 1000:.2f} ms")

        # Post-process
        start_time = time.time()
        tt_results = self.post_process(tt_cls, tt_reg, tt_dir)
        tt_postproc_time = time.time() - start_time
        logger.info(f"Post-processing time: {tt_postproc_time * 1000:.2f} ms")

        # Visualize
        self.visualize_results(
            result=tt_results[0],
            calib_info=calib_info,
            img=img,
            title="TTNN Detections",
            save_path=os.path.join(output_dir, "ttnn_detections.jpg") if img is not None else None,
        )
        logger.info("\nDemo completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="PointPillars Demo with PyTorch and TTNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python models/experimental/pointpillars/demo/demo.py
        """,
    )
    parser.add_argument(
        "--ckpt",
        default="models/experimental/pointpillars/resources/checkpoint/epoch_160.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output", default="models/experimental/pointpillars/resources/output", help="Output directory"
    )
    parser.add_argument("--device_id", type=int, default=0, help="Tenstorrent device ID")

    args = parser.parse_args()

    resources_dir = "models/experimental/pointpillars/resources"
    checkpoint_dir = os.path.join(resources_dir, "checkpoint")

    download_test_data(resources_dir)
    download_checkpoint(checkpoint_dir)

    pc_path = os.path.join(resources_dir, "000002.bin")
    calib_path = os.path.join(resources_dir, "000002.txt")
    img_path = os.path.join(resources_dir, "000002.png")

    logger.info(f"Opening Tenstorrent device {args.device_id}")
    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=79104)

    demo = PointPillarsDemo(device)

    try:
        # Setup models
        demo.setup_models(args.ckpt)

        # Run demo
        demo.run_demo(
            pc_path=pc_path,
            calib_path=calib_path,
            img_path=img_path,
            output_dir=args.output,
        )
        return 0

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        return 1

    finally:
        logger.info("Closing device")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
