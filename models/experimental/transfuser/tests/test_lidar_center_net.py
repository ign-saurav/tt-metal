# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import ttnn
import os
import json
from PIL import Image
import cv2


# Use environment variables for configuration instead of pytest options
# This avoids conflicts with pytest's argument parsing

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet
from models.experimental.transfuser.tt.lidar_center_net import LidarCenterNet as TtLidarCenterNet
from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor

from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


# Define lidar_to_histogram_features locally to avoid import issues
def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
        ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.3]
    above = lidar[lidar[..., 2] > -2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1, 2)).copy()
    return features


class TransfuserInputProcessor:
    """Processes raw data into Transfuser backbone inputs"""

    def __init__(
        self, config_values: dict = None, save_debug_images: bool = False, debug_output_dir: str = "debug_images"
    ):
        """
        Initialize the processor with configuration values

        Args:
            config_values: Dictionary containing configuration parameters
            save_debug_images: Whether to save debug images before/after preprocessing
            debug_output_dir: Directory to save debug images
        """
        # Default configuration values (matching GlobalConfig)
        self.config = {
            "img_resolution": (160, 704),  # (height, width)
            "scale": 1,
            "img_width": 320,
            "lidar_resolution_width": 256,
            "lidar_resolution_height": 256,
            "aug_degrees": [0],  # No augmentation for inference
            "use_target_point_image": False,
            "use_point_pillars": False,
        }

        # Update with provided config values
        if config_values:
            self.config.update(config_values)

        # Debug image saving settings
        self.save_debug_images = save_debug_images
        self.debug_output_dir = debug_output_dir

        if self.save_debug_images:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print(f"Debug images will be saved to: {self.debug_output_dir}")

    def process_image(self, rgb_image: Image.Image, frame_id: str = "0000") -> torch.Tensor:
        """
        Process RGB image to Transfuser format

        Args:
            rgb_image: PIL Image object
            frame_id: Frame identifier for debug image naming

        Returns:
            torch.Tensor: Processed image tensor of shape (1, 3, 160, 704)
        """
        print(f"Original image size: {rgb_image.size}")  # (width, height)

        # Save original image if debug mode is enabled
        if self.save_debug_images:
            original_path = os.path.join(self.debug_output_dir, f"{frame_id}_01_original.png")
            rgb_image.save(original_path)
            print(f"Saved original image: {original_path}")

        # Convert to numpy array
        rgb_array = np.array(rgb_image)
        print(f"RGB array shape: {rgb_array.shape}")

        # Convert RGB to BGR for CARLA compatibility
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        print(f"BGR array shape: {rgb_bgr.shape}")

        # Save BGR image if debug mode is enabled
        if self.save_debug_images:
            bgr_path = os.path.join(self.debug_output_dir, f"{frame_id}_02_bgr.png")
            cv2.imwrite(bgr_path, rgb_bgr)
            print(f"Saved BGR image: {bgr_path}")

        # Simulate multi-camera setup (using same image for all cameras)
        # This replicates the tick() method behavior
        rgb_cameras = []
        for i, pos in enumerate(["left", "front", "right"]):
            # Convert BGR back to RGB for processing
            rgb_pos = cv2.cvtColor(rgb_bgr[:, :, :3], cv2.COLOR_BGR2RGB)

            # Scale and crop each camera view
            rgb_pos = self._scale_crop(
                Image.fromarray(rgb_pos),
                self.config["scale"],
                0,
                self.config["img_resolution"][1],  # width
                0,
                self.config["img_resolution"][0],  # height
            )
            rgb_cameras.append(rgb_pos)

            # Save individual camera views if debug mode is enabled
            if self.save_debug_images:
                camera_path = os.path.join(self.debug_output_dir, f"{frame_id}_03_camera_{pos}.png")
                Image.fromarray(rgb_pos).save(camera_path)
                print(f"Saved {pos} camera view: {camera_path}")

        # Concatenate all camera views horizontally
        rgb_concatenated = np.concatenate(rgb_cameras, axis=1)
        print(f"Concatenated cameras shape: {rgb_concatenated.shape}")

        # Save concatenated image if debug mode is enabled
        if self.save_debug_images:
            concat_path = os.path.join(self.debug_output_dir, f"{frame_id}_04_concatenated.png")
            Image.fromarray(rgb_concatenated).save(concat_path)
            print(f"Saved concatenated image: {concat_path}")

        # Apply final processing (replicates prepare_image method)
        image_degrees = []
        for degree in self.config["aug_degrees"]:
            crop_shift = degree / 60 * self.config["img_width"]
            processed_image = self._shift_x_scale_crop(
                Image.fromarray(rgb_concatenated),
                scale=self.config["scale"],
                crop=self.config["img_resolution"],
                crop_shift=crop_shift,
            )
            # Convert to tensor and add batch dimension
            rgb_tensor = torch.from_numpy(processed_image).unsqueeze(0)
            image_degrees.append(rgb_tensor.to("cpu", dtype=torch.float32))

            # Save final processed image if debug mode is enabled
            if self.save_debug_images:
                final_path = os.path.join(self.debug_output_dir, f"{frame_id}_05_final_processed.png")
                # Convert tensor back to image for saving
                final_image_array = processed_image.transpose(1, 2, 0)  # CHW to HWC
                Image.fromarray(final_image_array).save(final_path)
                print(f"Saved final processed image: {final_path}")

        # Concatenate along batch dimension
        final_image = torch.cat(image_degrees, dim=0)
        print(f"Final image tensor shape: {final_image.shape}")

        return final_image

    def process_lidar(self, lidar_array: np.ndarray, frame_id: str = "0000") -> torch.Tensor:
        """
        Process LiDAR point cloud to Transfuser format

        Args:
            lidar_array: Raw LiDAR data from .npy file
            frame_id: Frame identifier for debug image naming

        Returns:
            torch.Tensor: Processed LiDAR tensor of shape (1, 2, 256, 256)
        """
        print(f"Original LiDAR array shape: {lidar_array.shape}")

        # Extract point cloud (same format as inference.py)
        pointcloud = lidar_array[1]  # Get the actual point cloud data
        print(f"Point cloud shape: {pointcloud.shape}")

        # Take only x, y, z coordinates (first 3 columns)
        pointcloud_xyz = pointcloud[:, :3]
        print(f"Point cloud XYZ shape: {pointcloud_xyz.shape}")

        # Save point cloud visualization if debug mode is enabled
        if self.save_debug_images:
            self._save_point_cloud_visualization(pointcloud_xyz, frame_id)

        # Apply histogram feature conversion (replicates prepare_lidar method)
        lidar_transformed = pointcloud_xyz.copy()
        lidar_transformed[:, 1] *= -1  # invert y-axis

        # Convert to histogram features
        histogram_features = lidar_to_histogram_features(lidar_transformed)
        histogram_tensor = torch.from_numpy(histogram_features).unsqueeze(0)

        print(f"Histogram features shape: {histogram_tensor.shape}")

        # Save histogram features visualization if debug mode is enabled
        if self.save_debug_images:
            self._save_histogram_visualization(histogram_features, frame_id)

        return histogram_tensor

    def process_lidar_bev(
        self, lidar_array: np.ndarray, target_point_image: torch.Tensor, frame_id: str = "0000"
    ) -> torch.Tensor:
        """
        Process LiDAR point cloud to final LiDAR BEV format (with target point image)

        Args:
            lidar_array: Raw LiDAR data from .npy file
            target_point_image: Target point image tensor (1, 1, 256, 256)
            frame_id: Frame identifier for debug image naming

        Returns:
            torch.Tensor: Final LiDAR BEV tensor of shape (1, 3, 256, 256)
        """
        # Get the 2-channel histogram features
        lidar_histogram = self.process_lidar(lidar_array, frame_id)

        # Concatenate with target point image to create 3-channel LiDAR BEV
        lidar_bev = torch.cat([lidar_histogram, target_point_image], dim=1)

        print(f"Final LiDAR BEV shape: {lidar_bev.shape}")

        # Save final LiDAR BEV visualization if debug mode is enabled
        if self.save_debug_images:
            self._save_lidar_bev_visualization(lidar_bev, frame_id)

        return lidar_bev

    def process_velocity(self, measurements: dict) -> torch.Tensor:
        """
        Process velocity from measurements

        Args:
            measurements: Dictionary containing measurement data

        Returns:
            torch.Tensor: Velocity tensor of shape (1, 1)
        """
        speed = measurements.get("speed", 0.0)
        velocity = torch.tensor([[speed]], dtype=torch.float32)
        print(f"Velocity tensor shape: {velocity.shape}, value: {speed}")
        return velocity

    def process_target_point(self, measurements: dict) -> tuple:
        """
        Process target point for waypoint prediction

        Args:
            measurements: Dictionary containing measurement data

        Returns:
            Tuple of (target_point, target_point_image) tensors
        """
        # For simplicity, use current position as target point
        # In real scenario, this would come from route planning
        target_point = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        # Create target point image using the same function as the original code
        target_point_image = self._draw_target_point(target_point[0].numpy())
        target_point_image = torch.from_numpy(target_point_image).unsqueeze(0)  # Add batch dimension

        print(f"Target point shape: {target_point.shape}")
        print(f"Target point image shape: {target_point_image.shape}")

        return target_point, target_point_image

    def _scale_crop(
        self, image: Image.Image, scale: int, start_x: int, crop_x: int, start_y: int, crop_y: int
    ) -> np.ndarray:
        """Scale and crop image (replicates scale_crop method)"""
        width, height = image.width // scale, image.height // scale
        if scale != 1:
            image = image.resize((width, height))

        image_array = np.asarray(image)
        cropped_image = image_array[start_y : start_y + crop_y, start_x : start_x + crop_x]
        return cropped_image

    def _shift_x_scale_crop(self, image: Image.Image, scale: int, crop: tuple, crop_shift: int = 0) -> np.ndarray:
        """Shift, scale and crop image (replicates shift_x_scale_crop method)"""
        crop_h, crop_w = crop
        width, height = int(image.width // scale), int(image.height // scale)
        im_resized = image.resize((width, height))
        image_array = np.array(im_resized)

        start_y = height // 2 - crop_h // 2
        start_x = width // 2 - crop_w // 2

        # Only shift in x direction
        start_x += int(crop_shift // scale)
        cropped_image = image_array[start_y : start_y + crop_h, start_x : start_x + crop_w]
        cropped_image = np.transpose(cropped_image, (2, 0, 1))  # HWC to CHW
        return cropped_image

    def _save_point_cloud_visualization(self, pointcloud: np.ndarray, frame_id: str):
        """Save point cloud visualization as PNG"""
        try:
            # Create a 2D scatter plot of the point cloud
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Top view (x, y)
            ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], c=pointcloud[:, 2], cmap="viridis", s=0.1, alpha=0.6)
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.set_title("Point Cloud - Top View")
            ax1.set_aspect("equal")
            ax1.grid(True)

            # Side view (x, z)
            ax2.scatter(pointcloud[:, 0], pointcloud[:, 2], c=pointcloud[:, 1], cmap="plasma", s=0.1, alpha=0.6)
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Z (m)")
            ax2.set_title("Point Cloud - Side View")
            ax2.set_aspect("equal")
            ax2.grid(True)

            plt.tight_layout()
            pc_path = os.path.join(self.debug_output_dir, f"{frame_id}_06_point_cloud.png")
            plt.savefig(pc_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved point cloud visualization: {pc_path}")

        except ImportError:
            print("Matplotlib not available for point cloud visualization")
        except Exception as e:
            print(f"Error saving point cloud visualization: {e}")

    def _save_histogram_visualization(self, histogram_features: np.ndarray, frame_id: str):
        """Save histogram features visualization as PNG"""
        try:
            import matplotlib.pyplot as plt

            # histogram_features shape: (2, 256, 256) - above and below ground
            above_ground = histogram_features[0]  # (256, 256)
            below_ground = histogram_features[1]  # (256, 256)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Above ground level
            im1 = ax1.imshow(above_ground, cmap="hot", origin="lower")
            ax1.set_title("Above Ground Level")
            ax1.set_xlabel("X (pixels)")
            ax1.set_ylabel("Y (pixels)")
            plt.colorbar(im1, ax=ax1)

            # Below ground level
            im2 = ax2.imshow(below_ground, cmap="hot", origin="lower")
            ax2.set_title("Below Ground Level")
            ax2.set_xlabel("X (pixels)")
            ax2.set_ylabel("Y (pixels)")
            plt.colorbar(im2, ax=ax2)

            # Combined (max of both)
            combined = np.maximum(above_ground, below_ground)
            im3 = ax3.imshow(combined, cmap="hot", origin="lower")
            ax3.set_title("Combined (Max)")
            ax3.set_xlabel("X (pixels)")
            ax3.set_ylabel("Y (pixels)")
            plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            hist_path = os.path.join(self.debug_output_dir, f"{frame_id}_07_histogram_features.png")
            plt.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved histogram features visualization: {hist_path}")

        except ImportError:
            print("Matplotlib not available for histogram visualization")
        except Exception as e:
            print(f"Error saving histogram visualization: {e}")

    def _draw_target_point(self, target_point: np.ndarray, color: tuple = (255, 255, 255)) -> np.ndarray:
        """Draw target point on 256x256 image (replicates draw_target_point function)"""
        image = np.zeros((256, 256), dtype=np.uint8)
        target_point = target_point.copy()

        # convert to lidar coordinate
        target_point[1] += 1.3
        point = target_point * 8.0
        point[1] *= -1
        point[1] = 256 - point[1]
        point[0] += 128
        point = point.astype(np.int32)
        point = np.clip(point, 0, 256)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        image = image.reshape(1, 256, 256)
        return image.astype(np.float32) / 255.0

    def _save_lidar_bev_visualization(self, lidar_bev: torch.Tensor, frame_id: str):
        """Save final LiDAR BEV visualization as PNG"""
        try:
            import matplotlib.pyplot as plt

            # lidar_bev shape: (1, 3, 256, 256) - histogram + target point
            histogram_above = lidar_bev[0, 0].numpy()  # (256, 256)
            histogram_below = lidar_bev[0, 1].numpy()  # (256, 256)
            target_point = lidar_bev[0, 2].numpy()  # (256, 256)

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

            # Above ground level
            im1 = ax1.imshow(histogram_above, cmap="hot", origin="lower")
            ax1.set_title("Above Ground Level")
            ax1.set_xlabel("X (pixels)")
            ax1.set_ylabel("Y (pixels)")
            plt.colorbar(im1, ax=ax1)

            # Below ground level
            im2 = ax2.imshow(histogram_below, cmap="hot", origin="lower")
            ax2.set_title("Below Ground Level")
            ax2.set_xlabel("X (pixels)")
            ax2.set_ylabel("Y (pixels)")
            plt.colorbar(im2, ax=ax2)

            # Target point
            im3 = ax3.imshow(target_point, cmap="hot", origin="lower")
            ax3.set_title("Target Point")
            ax3.set_xlabel("X (pixels)")
            ax3.set_ylabel("Y (pixels)")
            plt.colorbar(im3, ax=ax3)

            # Combined (all channels)
            combined = np.stack([histogram_above, histogram_below, target_point], axis=-1)
            ax4.imshow(combined, origin="lower")
            ax4.set_title("Combined LiDAR BEV (3 channels)")
            ax4.set_xlabel("X (pixels)")
            ax4.set_ylabel("Y (pixels)")

            plt.tight_layout()
            bev_path = os.path.join(self.debug_output_dir, f"{frame_id}_08_lidar_bev_final.png")
            plt.savefig(bev_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved final LiDAR BEV visualization: {bev_path}")

        except ImportError:
            print("Matplotlib not available for LiDAR BEV visualization")
        except Exception as e:
            print(f"Error saving LiDAR BEV visualization: {e}")

    def process_all_inputs(self, data_root: str, frame: str) -> dict:
        """
        Process all inputs from data root for a specific frame

        Args:
            data_root: Path to data directory
            frame: Frame number (e.g., "0120")

        Returns:
            Dictionary containing all processed inputs
        """
        print(f"Processing frame {frame} from {data_root}")
        print("=" * 50)

        # Load RGB image
        rgb_path = os.path.join(data_root, "rgb", f"{frame}.png")
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")

        rgb_image = Image.open(rgb_path).convert("RGB")
        processed_image = self.process_image(rgb_image, frame)

        # Load LiDAR data
        lidar_path = os.path.join(data_root, "lidar", f"{frame}.npy")
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"LiDAR data not found: {lidar_path}")

        lidar_array = np.load(lidar_path, allow_pickle=True)

        # Load measurements
        meas_path = os.path.join(data_root, "measurements", f"{frame}.json")
        if not os.path.exists(meas_path):
            raise FileNotFoundError(f"Measurements not found: {meas_path}")

        with open(meas_path, "r") as f:
            measurements = json.load(f)

        processed_velocity = self.process_velocity(measurements)
        target_point, target_point_image = self.process_target_point(measurements)

        # Process LiDAR BEV (3-channel: histogram + target point) - this is what TransfuserBackbone expects
        processed_lidar_bev = self.process_lidar_bev(lidar_array, target_point_image, frame)

        # Prepare inputs in the format expected by TransfuserBackbone
        inputs = {
            "image": processed_image,  # (1, 3, 160, 704)
            "lidar": processed_lidar_bev,  # (1, 3, 256, 256) - histogram + target point
            "velocity": processed_velocity,  # (1, 1)
            "target_point": target_point,  # (1, 2)
            "target_point_image": target_point_image,  # (1, 1, 256, 256)
        }

        print("=" * 50)
        print("Final processed inputs:")
        for key, value in inputs.items():
            print(f"  {key}: {value.shape} ({value.dtype})")

        return inputs


def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        # Process each head's parameters
        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            if head_name == "heatmap_head":
                weight_dtype = ttnn.float32
            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)

                # Get output channels for this head
                out_channels = head[2].weight.shape[0]  # From second conv layer

                # Note: We cannot use prepare_conv_weights here because we need
                # the full conv2d parameters (batch_size, input_height, etc.)
                # which are only available at runtime, not during preprocessing.
                # So we keep weights in PyTorch format and convert at runtime.
                parameters[head_name] = {}

                # Store weights in PyTorch format - will be prepared during first forward pass
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

        return parameters

    return custom_preprocessor


def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


def compare_boxes_pcc(ref_boxes, torch_boxes):
    """
    Compare all reference boxes with all torch boxes using PCC.
    Returns the top len(ref_boxes) PCC scores with their indices.
    """
    pcc_scores = []

    print("Computing PCC between all pairs of boxes...")

    # Compare each reference box with all torch boxes
    for i, bbox_ref in enumerate(ref_boxes):
        # Handle different data structures
        bbox_ref_array = bbox_ref[0] if isinstance(bbox_ref, tuple) else bbox_ref

        for j, bbox_torch in enumerate(torch_boxes):
            # Handle different data structures
            bbox_torch_array = bbox_torch[0] if isinstance(bbox_torch, tuple) else bbox_torch

            does_pass, pcc_value = check_with_pcc(
                bbox_ref_array, bbox_torch_array, 0.0
            )  # Use 0.0 threshold to get raw PCC
            print(f"PCC value: {pcc_value}")
            print(f"PCC passed: {does_pass}")
            pcc_scores.append((i, j, pcc_value))

    # Sort by PCC descending (best first)
    pcc_scores.sort(key=lambda x: x[2], reverse=True)

    # Take top len(ref_boxes) scores
    top_pcc = pcc_scores[: len(ref_boxes)]

    return top_pcc, pcc_scores


def print_results(top_pcc, all_pcc_scores):
    """
    Print the results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("TOP PCC SCORES (Top len(ref_boxes) matches)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Ref_Idx':<8} {'Torch_Idx':<10} {'PCC_Score':<12}")
    print("-" * 60)

    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        # Convert pcc_val to float if it's a string
        try:
            pcc_float = float(pcc_val)
            print(f"{rank:<6} {ref_idx:<8} {torch_idx:<10} {pcc_float:<12.6f}")
        except (ValueError, TypeError):
            print(f"{rank:<6} {ref_idx:<8} {torch_idx:<10} {str(pcc_val):<12}")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total comparisons: {len(all_pcc_scores)}")
    print(f"Top matches shown: {len(top_pcc)}")

    if all_pcc_scores:
        all_pcc_values = [float(score[2]) for score in all_pcc_scores]
        print(f"Best PCC score: {max(all_pcc_values):.6f}")
        print(f"Worst PCC score: {min(all_pcc_values):.6f}")
        print(f"Average PCC score: {np.mean(all_pcc_values):.6f}")
        print(f"Median PCC score: {np.median(all_pcc_values):.6f}")

    print("\n" + "=" * 60)
    print("DETAILED TOP MATCHES")
    print("=" * 60)
    for rank, (ref_idx, torch_idx, pcc_val) in enumerate(top_pcc, 1):
        # Convert pcc_val to float if it's a string
        try:
            pcc_float = float(pcc_val)
            print(f"Rank {rank}: Ref box {ref_idx} ↔ Torch box {torch_idx} (PCC: {pcc_float:.6f})")
        except (ValueError, TypeError):
            print(f"Rank {rank}: Ref box {ref_idx} ↔ Torch box {torch_idx} (PCC: {str(pcc_val)})")


def load_data_from_args_or_fallback(
    data_root=None, frame=None, save_debug_images=False, debug_output_dir="debug_images"
):
    """
    Load data from command line arguments or fallback to random data

    Args:
        data_root: Path to data directory
        frame: Frame number (e.g., "0120")
        save_debug_images: Whether to save debug images
        debug_output_dir: Directory to save debug images

    Returns:
        Dictionary containing processed inputs
    """
    # Check if data loading is requested and data exists
    use_real_data = False
    if data_root and frame:
        # Check if the data directory exists
        if os.path.exists(data_root):
            rgb_path = os.path.join(data_root, "rgb", f"{frame}.png")
            lidar_path = os.path.join(data_root, "lidar", f"{frame}.npy")
            meas_path = os.path.join(data_root, "measurements", f"{frame}.json")

            if all(os.path.exists(p) for p in [rgb_path, lidar_path, meas_path]):
                use_real_data = True
                print(f"Using real data from {data_root}, frame {frame}")
            else:
                print(f"Data directory exists but missing required files. Using random data instead.")
        else:
            print(f"Data directory {data_root} not found. Using random data instead.")

    if use_real_data:
        # Use TransfuserInputProcessor to load real data
        processor = TransfuserInputProcessor(save_debug_images=save_debug_images, debug_output_dir=debug_output_dir)
        inputs = processor.process_all_inputs(data_root, frame)
    else:
        # Generate random data as fallback
        print("Generating random data for testing...")
        inputs = {
            "image": torch.randn(1, 3, 160, 704, dtype=torch.float32),
            "lidar": torch.randn(1, 3, 256, 256, dtype=torch.float32),
            "velocity": torch.randn(1, 1, dtype=torch.float32),
            "target_point": torch.randn(1, 2, dtype=torch.float32),
            "target_point_image": torch.randn(1, 1, 256, 256, dtype=torch.float32),
        }
        print("Random data generated:")
        for key, value in inputs.items():
            print(f"  {key}: {value.shape} ({value.dtype})")

    return inputs


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, target_point_image_shape, img_shape, lidar_bev_shape",
    [
        ("regnety_032", "regnety_032", 4, False, (1, 1, 256, 256), (1, 3, 160, 704), (1, 3, 256, 256)),
    ],  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_lidar_center_net(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    target_point_image_shape,
    img_shape,
    lidar_bev_shape,
    input_dtype,
    weight_dtype,
):
    # Get configuration from environment variables
    data_root = os.environ.get("TRANSFUSER_DATA_ROOT", None)
    frame = os.environ.get("TRANSFUSER_FRAME", None)
    save_debug_images = os.environ.get("TRANSFUSER_SAVE_DEBUG_IMAGES", "false").lower() == "true"
    debug_output_dir = os.environ.get("TRANSFUSER_DEBUG_OUTPUT_DIR", "debug_images")

    # Load data from arguments or fallback to random data
    inputs = load_data_from_args_or_fallback(
        data_root=data_root, frame=frame, save_debug_images=save_debug_images, debug_output_dir=debug_output_dir
    )

    # Extract each component
    image = inputs["image"]  # RGB camera image tensor
    lidar_bev = inputs["lidar"]  # LiDAR BEV tensor
    velocity = inputs["velocity"]  # Ego velocity tensor
    target_point = inputs["target_point"]  # Target point tensor

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    # setting machine to avoid loading files
    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    config.use_target_point_image = True

    ref_layer = LidarCenterNet(
        config,
        backbone="transFuser",
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=use_velocity,
    ).eval()

    ref_feature, pred_wp, ref_head_results, ref_boxes, ref_rotated_bboxes = ref_layer.forward_ego(
        image, lidar_bev, target_point, velocity
    )

    # Unpack list outputs (each contains one tensor since we have single scale)
    (
        ref_center_heatmap_list,
        ref_wh_list,
        ref_offset_list,
        ref_yaw_class_list,
        ref_yaw_res_list,
        ref_velocity_list,
        ref_brake_list,
    ) = ref_head_results

    # Extract single tensors from lists
    ref_center_heatmap = ref_center_heatmap_list[0]
    ref_wh = ref_wh_list[0]
    ref_offset = ref_offset_list[0]
    ref_yaw_class = ref_yaw_class_list[0]
    ref_yaw_res = ref_yaw_res_list[0]
    ref_velocity = ref_velocity_list[0]
    ref_brake = ref_brake_list[0]
    torch_model = ref_layer._model

    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    gpt1_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer1,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer1"] = gpt1_parameters
    gpt2_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer2,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer2"] = gpt2_parameters
    gpt3_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer3,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer3"] = gpt3_parameters
    gpt4_parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.transformer4,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16),
        device=device,
    )
    parameters["transformer4"] = gpt4_parameters

    # Preprocess model parameters
    parameters["head"] = preprocess_model_parameters(
        initialize_model=lambda: ref_layer.head,
        custom_preprocessor=create_lidar_center_net_head_preprocessor(device, weight_dtype),
        device=device,
    )

    tt_layer = TtLidarCenterNet(
        device,
        parameters,
        config,
        backbone="transFuser",
    )

    # Convert input to TTNN format
    tt_image_input = ttnn.from_torch(
        image.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_lidar_input = ttnn.from_torch(
        lidar_bev.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_velocity_input = ttnn.from_torch(
        velocity,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_image = ttnn.to_device(tt_image_input, device)
    tt_lidar_bev = ttnn.to_device(tt_lidar_input, device)
    tt_velocity = ttnn.to_device(tt_velocity_input, device)

    tt_features, tt_pred_wp = tt_layer.forward_ego(tt_image, tt_lidar_bev, tt_velocity, target_point)

    torch_feature = ttnn.to_torch(tt_features[0], device=device, dtype=torch.float32)
    # Permute NHWC -> NCHW
    torch_feature = torch_feature.permute(0, 3, 1, 2)

    pcc_passed, pcc_msg = check_with_pcc(ref_feature, torch_feature, pcc=0.95)
    logger.info(f"Feature PCC: {pcc_msg}")
    assert pcc_passed, f"Feature PCC check failed: {pcc_msg}"

    does_pass, pred_wp_pcc_message = check_with_pcc(pred_wp, tt_pred_wp, 0.80)
    logger.info(f"pred wp PCC: {pred_wp_pcc_message}")
    assert does_pass, f"pred wp PCC check failed: {pred_wp_pcc_message}"

    torch_results = ref_layer.head([torch_feature])

    # import pdb; pdb.set_trace()
    does_pass, results_pcc_message = check_with_pcc(ref_head_results[0][0], torch_results[0][0], 0.80)
    logger.info(f"results PCC: {results_pcc_message}")
    assert does_pass, f"results PCC check failed: {results_pcc_message}"

    # Unpack list outputs
    (
        torch_center_heatmap_list,
        torch_wh_list,
        torch_offset_list,
        torch_yaw_class_list,
        torch_yaw_res_list,
        torch_velocity_list,
        torch_brake_list,
    ) = torch_results

    # Extract single tensors from lists
    torch_center_heatmap = torch_center_heatmap_list[0]
    torch_wh = torch_wh_list[0]
    torch_offset = torch_offset_list[0]
    torch_yaw_class = torch_yaw_class_list[0]
    torch_yaw_res = torch_yaw_res_list[0]
    torch_velocity = torch_velocity_list[0]
    torch_brake = torch_brake_list[0]

    # # After the pred_wp PCC check, add bbox post-processing for TTNN outputs

    # Convert TTNN outputs to torch for get_bboxes (it expects torch tensors)
    tt_preds_torch = (
        [torch_center_heatmap],
        [torch_wh],
        [torch_offset],
        [torch_yaw_class],
        [torch_yaw_res],
        [torch_velocity],
        [torch_brake],
    )

    # # Call get_bboxes on the reference head (reusing the same logic)
    torch_boxes = ref_layer.head.get_bboxes(*tt_preds_torch)
    does_pass, box_pcc_message = check_with_pcc(ref_boxes[0][0], torch_boxes[0][0], 0.80)
    logger.info(f"box PCC: {box_pcc_message}")
    torch_bboxes, _ = torch_boxes[0]

    # Filter by confidence threshold
    torch_bboxes = torch_bboxes[torch_bboxes[:, -1] > config.bb_confidence_threshold]

    # Convert to metric coordinates
    torch_rotated_bboxes = []
    for bbox in torch_bboxes.detach().cpu().numpy():
        bbox_metric = ref_layer.get_bbox_local_metric(bbox)
        torch_rotated_bboxes.append(bbox_metric)

    # Compare bbox counts
    logger.info(f"Reference bboxes count: {len(ref_rotated_bboxes)}")
    logger.info(f"TTNN bboxes count: {len(torch_rotated_bboxes)}")

    box_match = len(ref_rotated_bboxes) == len(torch_rotated_bboxes)
    logger.info(f"Box match: {box_match}")

    top_pcc, all_pcc_scores = compare_boxes_pcc(ref_rotated_bboxes, torch_rotated_bboxes)

    print_results(top_pcc, all_pcc_scores)

    does_pass, wh_pcc_message = check_with_pcc(ref_wh, torch_wh, 0.80)
    logger.info(f"WH PCC: {wh_pcc_message}")

    does_pass, offset_pcc_message = check_with_pcc(ref_offset, torch_offset, 0.80)
    logger.info(f"Offset PCC: {offset_pcc_message}")

    does_pass, yaw_class_pcc_message = check_with_pcc(ref_yaw_class, torch_yaw_class, 0.80)
    logger.info(f"Yaw Class PCC: {yaw_class_pcc_message}")

    does_pass, yaw_res_pcc_message = check_with_pcc(ref_yaw_res, torch_yaw_res, 0.80)
    logger.info(f"Yaw Residual PCC: {yaw_res_pcc_message}")

    does_pass, velocity_pcc_message = check_with_pcc(ref_velocity, torch_velocity, 0.80)
    logger.info(f"Velocity PCC: {velocity_pcc_message}")

    does_pass, brake_pcc_message = check_with_pcc(ref_brake, torch_brake, 0.80)
    logger.info(f"Brake PCC: {brake_pcc_message}")

    does_pass, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, torch_center_heatmap, 0.80)
    logger.info(f"Center Heatmap PCC: {heatmap_pcc_message}")

    assert does_pass, f"Center Heatmap PCC Failed! PCC: {heatmap_pcc_message}"

    if does_pass:
        try:
            print("SEED: ", torch.seed())
        except:
            pass
        logger.info("LidarCenterNet Passed!")
    else:
        logger.warning("LidarCenterNet Failed!")
