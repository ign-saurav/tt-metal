# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PointPillars Demo with PyTorch and TTNN

This demo script runs inference on point cloud data using both PyTorch and TTNN
implementations of PointPillars. It includes full post-processing with anchor
decoding, NMS, and saves visualization results to output directory.

Usage:
    python models/experimental/pointpillars/demo/demo.py \
        --pc_path models/experimental/pointpillars/data/val/000134.bin \
        --calib_path models/experimental/pointpillars/data/val/000134.txt \
        --img_path models/experimental/pointpillars/data/val/000134.png

Outputs:
    - pytorch_detections.jpg: Image with 3D bounding boxes from PyTorch model
    - ttnn_detections.jpg: Image with 3D bounding boxes from TTNN model
"""

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
from models.experimental.pointpillars.tt.pointpillars import TtPointPillars
from models.experimental.pointpillars.reference.model.pointpillars import PointPillars
from models.experimental.pointpillars.reference.model.anchors import Anchors, anchors2bboxes
from models.experimental.pointpillars.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.pointpillars.reference.ops import nms_cuda
from models.common.utility_functions import tt2torch_tensor

# Import reference utilities for I/O and visualization
from models.experimental.pointpillars.reference.utils import (
    read_points,
    read_calib,
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    vis_img_3d,
    bbox3d2corners_camera,
    points_camera2image,
    limit_period,
)


# Class mappings for KITTI dataset
CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}


def point_range_filter(pts: np.ndarray, point_range: List[float] = [0, -39.68, -3, 69.12, 39.68, 1]) -> np.ndarray:
    """
    Filter points that fall within the specified range.

    Args:
        pts: Point cloud array of shape (N, 4) with x, y, z, intensity
        point_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        Filtered points within the range
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


class PointPillarsPostProcessor:
    """
    Post-processing for PointPillars detection outputs.

    Handles anchor generation, bbox decoding, NMS, and score filtering.
    """

    def __init__(self, nclasses: int = 3):
        self.nclasses = nclasses

        # Anchor configuration for KITTI
        ranges = [
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Pedestrian
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Cyclist
            [0, -39.68, -1.78, 69.12, 39.68, -1.78],  # Car
        ]
        sizes = [
            [0.6, 0.8, 1.73],  # Pedestrian (w, l, h)
            [0.6, 1.76, 1.73],  # Cyclist
            [1.6, 3.9, 1.56],  # Car
        ]
        rotations = [0, 1.57]  # 0 and 90 degrees

        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)

        # Post-processing parameters
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(
        self,
        bbox_cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        bbox_dir_cls_pred: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Dict:
        """
        Post-process predictions for a single sample.

        Args:
            bbox_cls_pred: Classification predictions (n_anchors*3, H, W)
            bbox_pred: Regression predictions (n_anchors*7, H, W)
            bbox_dir_cls_pred: Direction predictions (n_anchors*2, H, W)
            anchors: Pre-computed anchors (H, W, 3, 2, 7)

        Returns:
            Dictionary with 'lidar_bboxes', 'labels', 'scores'
        """
        # Reshape predictions and convert to float32 (numpy/CUDA ops may not support bfloat16)
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses).float()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7).float()
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2).float()
        anchors = anchors.reshape(-1, 7).float()

        # Apply sigmoid to classification scores
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # Select top-k predictions based on max class score
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # Decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # Prepare 2D bboxes for NMS (x, y, w, l, theta)
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat(
            [bbox_pred2d_xy - bbox_pred2d_lw / 2, bbox_pred2d_xy + bbox_pred2d_lw / 2, bbox_pred[:, 6:]], dim=-1
        )

        ret_bboxes, ret_labels, ret_scores = [], [], []

        for i in range(self.nclasses):
            # Filter by score threshold
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            # Apply NMS
            keep_inds = nms_cuda(
                boxes=cur_bbox_pred2d,
                scores=cur_bbox_cls_pred,
                thresh=self.nms_thr,
                pre_maxsize=None,
                post_max_size=None,
            )

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]

            # Adjust heading angle based on direction classification
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred)
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # Handle empty results
        if len(ret_bboxes) == 0:
            return {
                "lidar_bboxes": np.array([]).reshape(0, 7),
                "labels": np.array([], dtype=np.int64),
                "scores": np.array([]),
            }

        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)

        # Keep top max_num predictions
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]

        return {
            "lidar_bboxes": ret_bboxes.detach().cpu().numpy(),
            "labels": ret_labels.detach().cpu().numpy(),
            "scores": ret_scores.detach().cpu().numpy(),
        }

    def get_predicted_bboxes(
        self,
        bbox_cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        bbox_dir_cls_pred: torch.Tensor,
        batched_anchors: List[torch.Tensor],
    ) -> List[Dict]:
        """
        Post-process predictions for a batch.

        Args:
            bbox_cls_pred: Classification predictions (B, n_anchors*3, H, W)
            bbox_pred: Regression predictions (B, n_anchors*7, H, W)
            bbox_dir_cls_pred: Direction predictions (B, n_anchors*2, H, W)
            batched_anchors: List of anchor tensors for each sample

        Returns:
            List of result dictionaries
        """
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(
                bbox_cls_pred=bbox_cls_pred[i],
                bbox_pred=bbox_pred[i],
                bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                anchors=batched_anchors[i],
            )
            results.append(result)
        return results

    def process(
        self,
        bbox_cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        bbox_dir_cls_pred: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> List[Dict]:
        """
        Full post-processing pipeline.

        Args:
            bbox_cls_pred: Classification predictions (B, n_anchors*3, H, W)
            bbox_pred: Regression predictions (B, n_anchors*7, H, W)
            bbox_dir_cls_pred: Direction predictions (B, n_anchors*2, H, W)
            device: Torch device for anchor generation

        Returns:
            List of result dictionaries with bboxes, labels, and scores
        """
        # Generate anchors based on feature map size
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)

        # Create batched anchors
        batch_size = bbox_cls_pred.size(0)
        batched_anchors = [anchors for _ in range(batch_size)]

        # Run post-processing
        results = self.get_predicted_bboxes(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_anchors=batched_anchors,
        )
        return results


class PointPillarsDemo:
    """
    Demo class for running PointPillars inference with PyTorch and TTNN.
    """

    def __init__(self, device: ttnn.Device):
        self.device = device
        self.torch_model = None
        self.ttnn_model = None

        # Model configuration
        self.nclasses = 3
        self.voxel_size = [0.16, 0.16, 4]
        self.point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.max_num_points = 32
        self.max_voxels = (16000, 40000)

        # Limit range for filtering
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

        # Post-processor
        self.post_processor = PointPillarsPostProcessor(nclasses=self.nclasses)

    def setup_models(self, ckpt_path: str):
        """
        Setup PyTorch model. TTNN model will be created  before inference.

        Args:
            ckpt_path: Path to the checkpoint file
        """
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
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.torch_model.load_state_dict(state_dict)
        self.torch_model = self.torch_model.to(dtype=torch.bfloat16)
        self.torch_model.eval()
        logger.info("PyTorch model loaded successfully")

        # TTNN model will be created lazily in setup_ttnn_model()
        self.ttnn_model = None

    def setup_ttnn_model(self):
        """Setup TTNN model - called right before TTNN inference to match test flow."""
        logger.info("Setting up TTNN model...")

        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
            device=self.device,
        )

        self.ttnn_model = TtPointPillars(
            nclasses=self.nclasses,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
            parameters=parameters,
            device=self.device,
        )
        logger.info("TTNN model loaded successfully")

    def run_pytorch_inference(self, pc_torch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference using PyTorch model.

        Args:
            pc_torch: Point cloud tensor of shape (N, 4)

        Returns:
            Tuple of (cls_pred, reg_pred, dir_pred) tensors
        """
        with torch.no_grad():
            cls_pred, reg_pred, dir_pred = self.torch_model(batched_pts=[pc_torch])
        return cls_pred, reg_pred, dir_pred

    def run_ttnn_inference(self, pc_torch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference using TTNN model.

        Args:
            pc_torch: Point cloud tensor of shape (N, 4)

        Returns:
            Tuple of (cls_pred, reg_pred, dir_pred) tensors converted back to PyTorch
        """
        with torch.no_grad():
            tt_cls, tt_reg, tt_dir = self.ttnn_model.forward(batched_pts=[pc_torch])

        # Convert TTNN outputs to PyTorch and permute from NHWC to NCHW
        cls_pred = tt2torch_tensor(tt_cls).permute(0, 3, 1, 2)
        reg_pred = tt2torch_tensor(tt_reg).permute(0, 3, 1, 2)
        dir_pred = tt2torch_tensor(tt_dir).permute(0, 3, 1, 2)

        return cls_pred, reg_pred, dir_pred

    def visualize_results(
        self,
        pc: np.ndarray,
        result: Dict,
        calib_info: Optional[Dict],
        img: Optional[np.ndarray],
        title: str = "Detections",
        save_path: Optional[str] = None,
    ):
        """
        Process detection results and save visualization to file.

        Args:
            pc: Point cloud array
            result: Detection result dictionary
            calib_info: Calibration information (optional)
            img: Camera image (optional)
            title: Title for logging
            save_path: Path to save visualization (optional)
        """
        # Skip if no detections
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
        """
        Run the complete demo with both PyTorch and TTNN inference.

        Args:
            pc_path: Path to point cloud file (.bin)
            calib_path: Path to calibration file (.txt)
            img_path: Path to image file (.png/.jpg)
            output_dir: Directory to save outputs
        """
        # Validate input
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found: {pc_path}")

        # Create output directory
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

        # Post-process
        start_time = time.time()
        pt_results = self.post_processor.process(pt_cls, pt_reg, pt_dir)
        pt_postproc_time = time.time() - start_time
        logger.info(f"Post-processing time: {pt_postproc_time * 1000:.2f} ms")

        # Visualize
        self.visualize_results(
            pc=pc,
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

        # Setup TTNN model right before inference (matches test flow)
        # This ensures parameters are loaded to device in the correct order
        self.setup_ttnn_model()

        start_time = time.time()
        tt_cls, tt_reg, tt_dir = self.run_ttnn_inference(pc_torch)
        tt_inference_time = time.time() - start_time
        logger.info(f"TTNN inference time: {tt_inference_time * 1000:.2f} ms")

        # Post-process
        start_time = time.time()
        tt_results = self.post_processor.process(tt_cls, tt_reg, tt_dir)
        tt_postproc_time = time.time() - start_time
        logger.info(f"Post-processing time: {tt_postproc_time * 1000:.2f} ms")

        # Visualize
        self.visualize_results(
            pc=pc,
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
    python models/experimental/pointpillars/demo/demo.py \\
        --pc_path models/experimental/pointpillars/data/val/000134.bin \\
        --calib_path models/experimental/pointpillars/data/val/000134.txt \\
        --img_path models/experimental/pointpillars/data/val/000134.png
        """,
    )
    parser.add_argument(
        "--ckpt",
        default="models/experimental/pointpillars/reference/model/epoch_160.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument("--pc_path", required=True, help="Path to point cloud file (.bin)")
    parser.add_argument("--calib_path", default="", help="Path to calibration file (.txt)")
    parser.add_argument("--img_path", default="", help="Path to image file (.png/.jpg)")
    parser.add_argument(
        "--output", default="models/experimental/pointpillars/resources/output", help="Output directory"
    )
    parser.add_argument("--device_id", type=int, default=0, help="Tenstorrent device ID")

    args = parser.parse_args()

    # Initialize device with same method as the working test (CreateDevice, not open_device)
    logger.info(f"Opening Tenstorrent device {args.device_id}")
    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=79104)

    demo = PointPillarsDemo(device)

    try:
        # Setup models
        demo.setup_models(args.ckpt)

        # Run demo
        demo.run_demo(
            pc_path=args.pc_path,
            calib_path=args.calib_path,
            img_path=args.img_path,
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
