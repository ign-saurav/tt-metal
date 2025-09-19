# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import logging
from collections import Counter
from config import DemoConfig
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Post-processing class
# ---------------------------------------------------------------------
class PostProcessing:
    """
    Post-processing for Panoptic-DeepLab
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self.center_threshold = self.config.center_threshold
        self.nms_kernel = self.config.nms_kernel
        self.top_k_instance = self.config.top_k_instances
        self.thing_classes = self.config.thing_classes
        self.stuff_classes = self.config.stuff_classes
        self.stuff_area_threshold = self.config.stuff_area_threshold
        self.label_divisor = self.config.label_divisor
        self.instance_score_threshold = self.config.instance_score_threshold

        # Create lookup sets for faster checking
        self.thing_set = set(self.thing_classes)
        self.stuff_set = set(self.stuff_classes)

    def find_instance_center(self, center_heatmap, threshold=0.1, nms_kernel=7, top_k=200):
        """
        Find the center points from the center heatmap.
        """
        # Normalize shape to [N, C, H, W] (N usually 1, C usually 1)
        if center_heatmap.dim() == 2:
            center_heatmap = center_heatmap.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif center_heatmap.dim() == 3:
            # could be [C,H,W] or [1,H,W]; make it [1,C,H,W]
            center_heatmap = center_heatmap.unsqueeze(0)
        elif center_heatmap.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported center_heatmap dim: {center_heatmap.dim()}")

        # Apply threshold; below threshold become -1
        center_heatmap = F.threshold(center_heatmap, threshold, -1.0)

        # Stronger NMS via max pooling with larger kernel
        nms_padding = (nms_kernel - 1) // 2
        center_heatmap_max_pooled = F.max_pool2d(center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding)

        # Keep only strong local maxima
        center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1.0

        # Additional filtering: remove weak peaks
        max_val = center_heatmap.max()
        min_peak_value = max_val * 0.3  # Only keep peaks that are at least 30% of max
        center_heatmap[center_heatmap < min_peak_value] = -1.0

        # Continue with rest of the method...
        center_heatmap = center_heatmap.squeeze()
        if center_heatmap.dim() != 2:
            center_heatmap = center_heatmap[0] if center_heatmap.dim() > 2 else center_heatmap

        all_centers = torch.nonzero(center_heatmap > 0.0)
        if all_centers.size(0) == 0:
            return all_centers

        if top_k is not None and all_centers.size(0) > top_k:
            scores = center_heatmap[all_centers[:, 0], all_centers[:, 1]]
            _, top_indices = torch.topk(scores, min(top_k, scores.size(0)))
            return all_centers[top_indices]

        return all_centers

    def group_pixels(self, center_points, offsets):
        """Group pixels with stricter distance threshold."""
        height, width = offsets.size()[1:]

        # Generate coordinate map
        y_coord, x_coord = torch.meshgrid(
            torch.arange(height, dtype=offsets.dtype, device=offsets.device),
            torch.arange(width, dtype=offsets.dtype, device=offsets.device),
            indexing="ij",
        )
        coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

        # Predicted centers for each pixel
        center_loc = coord + offsets
        center_loc = center_loc.flatten(1).T.unsqueeze_(0)
        center_points = center_points.unsqueeze(1).float()

        # Compute distances
        distance = torch.norm(center_points - center_loc, dim=-1)
        MAX_DISTANCE = (
            self.config.max_distance
        )  # Any pixel farther than this from all centers is not assigned to any instance.
        min_dist, instance_id = torch.min(distance, dim=0)
        instance_id = instance_id.reshape((1, height, width)) + 1
        instance_id[min_dist.reshape((1, height, width)) > MAX_DISTANCE] = 0

        return instance_id

    def get_instance_segmentation(
        self, sem_seg, center_heatmap, offsets, thing_seg, thing_ids, threshold=0.1, nms_kernel=3, top_k=None
    ):
        """
        Post-processing for instance segmentation, gets class agnostic instance id.
        """
        center_points = self.find_instance_center(
            center_heatmap, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k
        )
        logger.info(f"Found {center_points.size(0)} instance centers")
        if center_points.size(0) == 0:
            return torch.zeros_like(sem_seg), center_points.unsqueeze(0)
        ins_seg = self.group_pixels(center_points, offsets)
        return thing_seg * ins_seg, center_points.unsqueeze(0)

    def merge_semantic_and_instance(
        self, sem_seg, ins_seg, semantic_thing_seg, label_divisor, thing_ids, stuff_area, void_label
    ):
        """
        Post-processing for panoptic segmentation, by merging semantic segmentation
        label and class agnostic instance segmentation label.
        """
        # In case thing mask does not align with semantic prediction.
        pan_seg = torch.zeros_like(sem_seg) + void_label
        is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

        # Keep track of instance id for each class.
        class_id_tracker = Counter()

        # Paste thing by majority voting.
        instance_ids = torch.unique(ins_seg)
        for ins_id in instance_ids:
            if ins_id == 0:
                continue
            # Make sure only do majority voting within `semantic_thing_seg`.
            thing_mask = (ins_seg == ins_id) & is_thing
            if torch.nonzero(thing_mask).size(0) == 0:
                continue
            class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
            class_id_tracker[class_id.item()] += 1
            new_ins_id = class_id_tracker[class_id.item()]
            pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

        unique_pan = torch.unique(pan_seg)
        thing_instances = sum(1 for p in unique_pan if (p % label_divisor) > 0 and (p // label_divisor) in thing_ids)
        logger.info(f"Created {thing_instances} thing instances")

        class_ids = torch.unique(sem_seg)
        for class_id in class_ids:
            if class_id.item() in thing_ids:
                continue
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
            if stuff_mask.sum().item() >= stuff_area:
                pan_seg[stuff_mask] = class_id * label_divisor
        # Clamp all values to be >= 0 (removes any negative/void labels)
        pan_seg = torch.clamp(pan_seg, min=0)

        road_class_id = 0
        if road_class_id in self.stuff_set:
            # Find all pixels semantically classified as road
            road_semantic_mask = sem_seg == road_class_id

            if road_semantic_mask.sum() > 1000:  # If significant road area exists
                # Find road pixels not already assigned to panoptic segments
                unassigned_road_mask = road_semantic_mask & (pan_seg == void_label)

                if unassigned_road_mask.sum() > 500:  # If significant unassigned road area
                    logger.info(f"Force-assigning {unassigned_road_mask.sum()} road pixels to panoptic")
                    # Assign road class ID (0) with label_divisor to create panoptic ID
                    pan_seg[unassigned_road_mask] = road_class_id * label_divisor

                # Also force-assign any road pixels that might be void
                all_road_mask = road_semantic_mask & (pan_seg <= void_label)
                if all_road_mask.sum() > 500:
                    logger.info(f"Force-assigning additional {all_road_mask.sum()} void road pixels")
                    pan_seg[all_road_mask] = road_class_id * label_divisor

                # Debug: Check final road assignment
                final_road_panoptic = (pan_seg // label_divisor) == road_class_id
                logger.info(
                    f"Final road panoptic assignment: {final_road_panoptic.sum()} pixels with ID {road_class_id * label_divisor}"
                )

        return pan_seg

    def get_panoptic_segmentation(
        self,
        sem_seg,
        center_heatmap,
        offsets,
        thing_ids=[11, 12, 13, 14, 15, 16, 17, 18],
        label_divisor=256,
        stuff_area=2048,
        void_label=0,
        threshold=0.05,
        nms_kernel=7,
        top_k=200,
        foreground_mask=None,
    ):
        """
        Post-processing for panoptic segmentation.
        """
        # --- normalize sem_seg to [1,H,W]
        if sem_seg.dim() == 4 and sem_seg.shape[0] == 1:
            sem_seg = torch.argmax(sem_seg, dim=1)
        elif sem_seg.dim() == 4 and sem_seg.shape[0] != 1:
            # If batched, pick first item
            sem_seg = torch.argmax(sem_seg, dim=1)[0].unsqueeze(0)
        elif sem_seg.dim() == 3 and sem_seg.shape[0] == 1:
            # already [1,H,W]
            pass
        elif sem_seg.dim() == 3:
            # [C,H,W] ? we expected [1,H,W] - take first batch
            sem_seg = sem_seg[0].unsqueeze(0)
        elif sem_seg.dim() == 2:
            sem_seg = sem_seg.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected sem_seg shape: {sem_seg.shape}")

        # --- normalize center_heatmap to [1,1,H,W]
        if center_heatmap.dim() == 2:
            center_heatmap = center_heatmap.unsqueeze(0).unsqueeze(0)
        elif center_heatmap.dim() == 3:
            # if [1,H,W] -> [1,1,H,W], if [C,H,W] -> [1,C,H,W]
            if center_heatmap.shape[0] == 1:
                center_heatmap = center_heatmap.unsqueeze(1)
            else:
                center_heatmap = center_heatmap.unsqueeze(0)
        elif center_heatmap.dim() == 4:
            # keep as-is (N,C,H,W)
            pass
        else:
            raise ValueError(f"Unexpected center_heatmap shape: {center_heatmap.shape}")

        # --- normalize offsets to [2,H,W] (instance offsets expected channel-first)
        if offsets.dim() == 4:
            # [N,C,H,W] -> squeeze batch if single
            if offsets.shape[0] == 1:
                offsets = offsets.squeeze(0)  # [C,H,W]
            else:
                offsets = offsets[0]
        elif offsets.dim() == 3:
            # keep as [C,H,W]
            pass
        elif offsets.dim() == 2:
            # [H,W] -> not valid for offsets but we will unsqueeze channel
            offsets = offsets.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected offsets shape: {offsets.shape}")
        # Create thing mask
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in thing_ids:
            thing_seg[sem_seg == thing_class] = 1

        # Get instances
        instance, center = self.get_instance_segmentation(
            sem_seg,
            center_heatmap,
            offsets,
            thing_seg,
            thing_ids,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )

        # Merge semantic and instance
        panoptic = self.merge_semantic_and_instance(
            sem_seg, instance, thing_seg, label_divisor, thing_ids, stuff_area, void_label
        )

        return panoptic, center

    def postprocess_outputs(
        self,
        torch_outputs: torch.Tensor,
        torch_outputs_2: torch.Tensor,
        torch_outputs_3: torch.Tensor,
        ttnn_outputs,
        ttnn_outputs_2,
        ttnn_outputs_3,
        original_size: Tuple[int, int],
        ttnn_device,
        output_mesh_composer: Optional[Any],
    ) -> Dict[str, Dict]:
        """Process outputs from both PyTorch and TTNN pipelines."""
        results = {"torch": {}, "ttnn": {}}

        # Process PyTorch outputs
        if torch_outputs is not None:
            logger.info("Processing PyTorch outputs...")
            try:
                # Run panoptic fusion
                panoptic_pred, _ = self.get_panoptic_segmentation(torch_outputs, torch_outputs_3, torch_outputs_2)
                void_pixels = (panoptic_pred == -1).sum()
                total_pixels = panoptic_pred.numel()
                logger.info(f"Void pixels: {void_pixels}/{total_pixels} ({100*void_pixels/total_pixels:.1f}%)")

                # Extract individual outputs
                semantic_pred = torch.argmax(torch_outputs, dim=1)

                # Convert to numpy and resize
                results["torch"]["semantic_pred"] = cv2.resize(
                    semantic_pred[0].cpu().numpy().astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                )

                results["torch"]["panoptic_pred"] = cv2.resize(
                    panoptic_pred[0].cpu().numpy().astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                )

                # save center heatmap
                center_np = torch_outputs_3[0, 0].cpu().numpy()
                results["torch"]["center_heatmap"] = cv2.resize(
                    center_np, original_size, interpolation=cv2.INTER_LINEAR
                )

                # save offset map
                offset_np = torch_outputs_2[0].cpu().numpy()
                results["torch"]["offset_map"] = np.stack(
                    [
                        cv2.resize(offset_np[0], original_size, interpolation=cv2.INTER_LINEAR),
                        cv2.resize(offset_np[1], original_size, interpolation=cv2.INTER_LINEAR),
                    ]
                )

            except Exception as e:
                logger.error(f"Error processing PyTorch outputs: {e}")
                import traceback

                traceback.print_exc()

        # Process TTNN outputs (similar structure)
        if ttnn_outputs is not None:
            logger.info("Processing TTNN outputs...")
            import ttnn

            try:
                # Convert to PyTorch tensors
                semantic_logits = ttnn.to_torch(ttnn_outputs, device=ttnn_device, mesh_composer=output_mesh_composer)
                offset_map = ttnn.to_torch(ttnn_outputs_2, device=ttnn_device, mesh_composer=output_mesh_composer)
                center_heatmap = ttnn.to_torch(ttnn_outputs_3, device=ttnn_device, mesh_composer=output_mesh_composer)

                # Reshape if needed (handle TTNN's NHWC format)
                semantic_logits = self._reshape_ttnn_output(semantic_logits, "semantic_logits")
                offset_map = self._reshape_ttnn_output(offset_map, "offset_map")
                center_heatmap = self._reshape_ttnn_output(center_heatmap, "center_heatmap")

                # Run panoptic fusion
                panoptic_pred, _ = self.get_panoptic_segmentation(semantic_logits, center_heatmap, offset_map)

                # Extract outputs
                semantic_pred = torch.argmax(semantic_logits, dim=1)

                # Convert and resize
                results["ttnn"]["semantic_pred"] = cv2.resize(
                    semantic_pred[0].cpu().numpy().astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                )

                results["ttnn"]["panoptic_pred"] = cv2.resize(
                    panoptic_pred[0].cpu().numpy().astype(np.int64), original_size, interpolation=cv2.INTER_NEAREST
                )

                # outputs
                center_np = center_heatmap[0, 0].cpu().float().numpy()
                results["ttnn"]["center_heatmap"] = cv2.resize(center_np, original_size, interpolation=cv2.INTER_LINEAR)

                offset_np = offset_map[0].cpu().float().numpy()
                results["ttnn"]["offset_map"] = np.stack(
                    [
                        cv2.resize(offset_np[0], original_size, interpolation=cv2.INTER_LINEAR),
                        cv2.resize(offset_np[1], original_size, interpolation=cv2.INTER_LINEAR),
                    ]
                )

            except Exception as e:
                logger.error(f"Error processing TTNN outputs: {e}")
                import traceback

                traceback.print_exc()

        return results

    def _reshape_ttnn_output(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Handle TTNN's NHWC to NCHW conversion."""
        if not isinstance(tensor, torch.Tensor):
            return tensor

        if tensor.dim() != 4:
            return tensor

        N, d1, d2, d3 = tensor.shape

        if key == "semantic_logits":
            expected_c = 19  # set this to dataset's num_classes
            if d1 == expected_c:
                return tensor  # assume already NCHW
            if d3 == expected_c:
                return tensor.permute(0, 3, 1, 2)

        elif key == "center_heatmap":
            # Center head is single channel
            if d1 == 1:
                return tensor
            if d3 == 1:
                return tensor.permute(0, 3, 1, 2)

        elif key == "offset_map":
            if d1 == 2:
                return tensor
            if d3 == 2:
                return tensor.permute(0, 3, 1, 2)

        return tensor


# ---------------------------------------------------------------------
# Panoptic visualizer class
# ---------------------------------------------------------------------
class PanopticVisualizer:
    """Visualizer that adds labels to panoptic segmentation results."""

    def __init__(self, config, alpha: float = 0.5):
        self.config = config
        self.colors = config._get_cityscapes_colors()
        self.class_names = config.class_names
        self.thing_classes = set(config.thing_classes)
        self.stuff_classes = set(config.stuff_classes)
        self.label_divisor = config.label_divisor
        self.alpha = alpha  # transparency level (0.5 = 50% transparent)

    # ---------------------------------------------------------------------
    # Create transparent overlay
    # ---------------------------------------------------------------------
    def create_transparent_overlay(
        self, original_image: np.ndarray, panoptic_pred: np.ndarray, alpha: Optional[float] = None
    ) -> np.ndarray:
        """Create a transparent colored overlay on the original image."""

        if alpha is None:
            alpha = self.alpha

        if original_image.shape[-1] != 3:
            raise ValueError("Original image must be RGB")

        overlay = original_image.copy()
        output = original_image.copy()

        unique_ids = np.unique(panoptic_pred)

        road_pixels_found = False
        for pan_id in unique_ids:
            if pan_id > 0:
                semantic_class = int(pan_id // self.label_divisor)
                if semantic_class == 0:
                    road_pixels_found = True
                    mask = panoptic_pred == pan_id

        # Process all panoptic IDs
        for pan_id in unique_ids:
            if pan_id == 0:  # Skip void/background
                continue

            semantic_class = int(pan_id // self.label_divisor)
            instance_id = int(pan_id % self.label_divisor)
            mask = panoptic_pred == pan_id

            if semantic_class < len(self.colors):
                base_color = self.colors[semantic_class].copy()

                # Special handling for road (class 0)
                if semantic_class == 0:
                    road_overlay_color = np.array([128, 64, 128], dtype=np.uint8)
                    overlay[mask] = road_overlay_color
                    continue

                # For thing instances, add color variation
                if instance_id > 0 and semantic_class in self.thing_classes:
                    rng = np.random.default_rng(int(pan_id))
                    jitter = rng.integers(-30, 31, size=3)
                    base_color = np.clip(base_color.astype(int) + jitter, 0, 255)

                overlay[mask] = base_color

        # If no road was found in panoptic, try to overlay road from semantic segmentation directly
        if not road_pixels_found:
            # Create a mask for any pixels that semantically should be road
            potential_road_mask = ((panoptic_pred // self.label_divisor) == 0) & (panoptic_pred >= 0)
            if potential_road_mask.any():
                road_overlay_color = np.array([128, 64, 128], dtype=np.uint8)
                overlay[potential_road_mask] = road_overlay_color

        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output

    # ---------------------------------------------------------------------
    # Add text labels and instance IDs to panoptic segmentation results
    # ---------------------------------------------------------------------
    def add_labels_to_panoptic(
        self,
        image: np.ndarray,
        panoptic_pred: np.ndarray,
    ) -> np.ndarray:
        """Add labels to panoptic segmentation results."""

        labeled_image = image.copy()
        h, w = labeled_image.shape[:2]

        font = ImageFont.load_default()

        pil_image = Image.fromarray(labeled_image)
        draw = ImageDraw.Draw(pil_image)

        labeled_regions = []
        labeled_classes = set()

        # Get unique panoptic IDs and group by class
        unique_ids = np.unique(panoptic_pred)
        class_segments = {}

        for pan_id in unique_ids:
            if pan_id > 0:
                semantic_class = int(pan_id // self.label_divisor)
                area = (panoptic_pred == pan_id).sum()

                if semantic_class not in class_segments:
                    class_segments[semantic_class] = []
                class_segments[semantic_class].append((pan_id, area))

        # Process cars (class 13)
        if 13 in class_segments:
            car_segments = class_segments[13]
            # Filter out very small car segments
            car_segments = [(pid, area) for pid, area in car_segments if area > 300]

            # Group nearby car segments
            consolidated_cars = self._consolidate_car_segments(panoptic_pred, car_segments)

            # Filter out very small consolidated cars
            valid_cars = [
                (merged_mask, total_area) for merged_mask, total_area in consolidated_cars if total_area >= 800
            ]

            # Label cars with simple sequential numbering based on actual count
            for car_num, (merged_mask, total_area) in enumerate(valid_cars, 1):
                coords = np.column_stack(np.where(merged_mask))
                if len(coords) == 0:
                    continue

                # Find best position for this car cluster
                centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                # Try upper area for better visibility
                upper_coords = coords[coords[:, 0] < np.percentile(coords[:, 0], 50)]
                if len(upper_coords) > 20:
                    alt_y, alt_x = upper_coords.mean(axis=0).astype(int)
                    # Choose the position that's more central
                    if upper_coords.shape[0] > coords.shape[0] * 0.3:
                        centroid_y, centroid_x = alt_y, alt_x

                # Sequential numbering: car#1, car#2, car#3, car#4 (for multiple cars)
                if len(valid_cars) > 1:
                    label = f"car#{car_num}"
                else:
                    label = "car"

                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position label
                label_x = max(5, min(centroid_x - text_width // 2, w - text_width - 5))
                label_y = max(5, min(centroid_y - text_height // 2, h - text_height - 5))

                # Check overlap with existing labels
                overlap = False
                for lx, ly, lw, lh in labeled_regions:
                    center_dist = np.sqrt(
                        (label_x + text_width / 2 - lx - lw / 2) ** 2 + (label_y + text_height / 2 - ly - lh / 2) ** 2
                    )
                    if center_dist < 100:  # overlap distance
                        overlap = True
                        break

                if not overlap:
                    # Draw car label
                    padding = 4
                    draw.rectangle(
                        [
                            label_x - padding,
                            label_y - padding,
                            label_x + text_width + padding,
                            label_y + text_height + padding,
                        ],
                        fill=(255, 255, 255, 240),
                    )

                    # Add shadow
                    draw.text((label_x + 1, label_y + 1), label, fill=(0, 0, 0, 180), font=font)
                    draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                    labeled_regions.append(
                        (label_x - padding, label_y - padding, text_width + 2 * padding, text_height + 2 * padding)
                    )

            # Mark cars as processed
            labeled_classes.add(13)

        # Process other vehicle classes
        for vehicle_class in [14, 15, 16, 17, 18]:  # truck, bus, train, motorcycle, bicycle
            if vehicle_class in class_segments:
                segments = class_segments[vehicle_class]
                # Filter by size and sort by area (largest first)
                valid_segments = [(pid, area) for pid, area in segments if area > 1000]

                if valid_segments:
                    valid_segments.sort(key=lambda x: x[1], reverse=True)

                    # Label each instance with sequential numbering
                    for instance_num, (pan_id, area) in enumerate(valid_segments, 1):
                        mask = panoptic_pred == pan_id
                        coords = np.column_stack(np.where(mask))

                        if len(coords) > 0:
                            centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                            if vehicle_class < len(self.class_names):
                                base_name = self.class_names[vehicle_class]
                            else:
                                base_name = f"vehicle_{vehicle_class}"

                            # Add sequential numbering if multiple instances
                            if len(valid_segments) > 1:
                                label = f"{base_name}#{instance_num}"
                            else:
                                label = base_name

                            bbox = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]

                            label_x = max(5, min(centroid_x - text_width // 2, w - text_width - 5))
                            label_y = max(5, min(centroid_y - text_height // 2, h - text_height - 5))

                            # Check overlap
                            overlap = False
                            for lx, ly, lw, lh in labeled_regions:
                                if not (
                                    label_x + text_width < lx
                                    or label_x > lx + lw
                                    or label_y + text_height < ly
                                    or label_y > ly + lh
                                ):
                                    overlap = True
                                    break

                            if not overlap:
                                padding = 3
                                draw.rectangle(
                                    [
                                        label_x - padding,
                                        label_y - padding,
                                        label_x + text_width + padding,
                                        label_y + text_height + padding,
                                    ],
                                    fill=(255, 255, 255, 220),
                                )

                                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                                labeled_regions.append(
                                    (
                                        label_x - padding,
                                        label_y - padding,
                                        text_width + 2 * padding,
                                        text_height + 2 * padding,
                                    )
                                )

                labeled_classes.add(vehicle_class)

        # Process people (class 11)
        if 11 in class_segments:
            person_segments = class_segments[11]
            # Filter by size and sort by area
            valid_persons = [(pid, area) for pid, area in person_segments if area > 800]

            if valid_persons:
                valid_persons.sort(key=lambda x: x[1], reverse=True)

                # Label each person with sequential numbering
                for person_num, (pan_id, area) in enumerate(valid_persons, 1):
                    mask = panoptic_pred == pan_id
                    coords = np.column_stack(np.where(mask))

                    if len(coords) > 0:
                        centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                        # Add sequential numbering if multiple persons
                        if len(valid_persons) > 1:
                            label = f"person#{person_num}"
                        else:
                            label = "person"

                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        label_x = max(5, min(centroid_x - text_width // 2, w - text_width - 5))
                        label_y = max(5, min(centroid_y - text_height // 2, h - text_height - 5))

                        # Check overlap
                        overlap = False
                        for lx, ly, lw, lh in labeled_regions:
                            center_dist = np.sqrt(
                                (label_x + text_width / 2 - lx - lw / 2) ** 2
                                + (label_y + text_height / 2 - ly - lh / 2) ** 2
                            )
                            if center_dist < 80:
                                overlap = True
                                break

                        if not overlap:
                            padding = 3
                            draw.rectangle(
                                [
                                    label_x - padding,
                                    label_y - padding,
                                    label_x + text_width + padding,
                                    label_y + text_height + padding,
                                ],
                                fill=(255, 255, 255, 220),
                            )

                            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                            labeled_regions.append(
                                (
                                    label_x - padding,
                                    label_y - padding,
                                    text_width + 2 * padding,
                                    text_height + 2 * padding,
                                )
                            )

            labeled_classes.add(11)

        # Process rider (class 12)
        if 12 in class_segments:
            rider_segments = class_segments[12]
            # Filter by size and sort by area
            valid_riders = [(pid, area) for pid, area in rider_segments if area > 600]

            if valid_riders:
                valid_riders.sort(key=lambda x: x[1], reverse=True)

                # Label each rider with sequential numbering
                for rider_num, (pan_id, area) in enumerate(valid_riders, 1):
                    mask = panoptic_pred == pan_id
                    coords = np.column_stack(np.where(mask))

                    if len(coords) > 0:
                        centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                        # Add sequential numbering if multiple riders
                        if len(valid_riders) > 1:
                            label = f"rider#{rider_num}"
                        else:
                            label = "rider"

                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        label_x = max(5, min(centroid_x - text_width // 2, w - text_width - 5))
                        label_y = max(5, min(centroid_y - text_height // 2, h - text_height - 5))

                        # Check overlap
                        overlap = False
                        for lx, ly, lw, lh in labeled_regions:
                            center_dist = np.sqrt(
                                (label_x + text_width / 2 - lx - lw / 2) ** 2
                                + (label_y + text_height / 2 - ly - lh / 2) ** 2
                            )
                            if center_dist < 80:
                                overlap = True
                                break

                        if not overlap:
                            padding = 3
                            draw.rectangle(
                                [
                                    label_x - padding,
                                    label_y - padding,
                                    label_x + text_width + padding,
                                    label_y + text_height + padding,
                                ],
                                fill=(255, 255, 255, 220),
                            )

                            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                            labeled_regions.append(
                                (
                                    label_x - padding,
                                    label_y - padding,
                                    text_width + 2 * padding,
                                    text_height + 2 * padding,
                                )
                            )

            labeled_classes.add(12)

        # Process stuff classes (road, sidewalk, building, etc.) - one label per class
        for stuff_class in self.stuff_classes:
            if stuff_class in labeled_classes:
                continue

            if stuff_class in class_segments:
                segments = class_segments[stuff_class]
                total_area = sum(area for _, area in segments)

                # Lower area threshold for fence, pole, and other small stuff classes
                min_area_threshold = 1000 if stuff_class in [4, 5] else 5000  # fence=4, pole=5

                if total_area > min_area_threshold:
                    # Create combined mask for all segments of this class
                    combined_mask = np.zeros_like(panoptic_pred, dtype=bool)
                    for pan_id, _ in segments:
                        combined_mask |= panoptic_pred == pan_id

                    coords = np.column_stack(np.where(combined_mask))
                    if len(coords) > 0:
                        # Smart positioning based on class
                        if stuff_class == 2:  # building - upper area
                            upper_coords = coords[coords[:, 0] < np.percentile(coords[:, 0], 30)]
                            if len(upper_coords) > 0:
                                centroid_y, centroid_x = upper_coords.mean(axis=0).astype(int)
                            else:
                                centroid_y, centroid_x = coords.mean(axis=0).astype(int)
                        elif stuff_class == 0:  # road - lower area
                            lower_coords = coords[coords[:, 0] > np.percentile(coords[:, 0], 70)]
                            if len(lower_coords) > 0:
                                centroid_y, centroid_x = lower_coords.mean(axis=0).astype(int)
                            else:
                                centroid_y, centroid_x = coords.mean(axis=0).astype(int)
                        elif stuff_class == 4:  # fence - try right side area
                            right_coords = coords[coords[:, 1] > w * 0.7]
                            if len(right_coords) > 10:
                                centroid_y, centroid_x = right_coords.mean(axis=0).astype(int)
                            else:
                                centroid_y, centroid_x = coords.mean(axis=0).astype(int)
                        else:
                            centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                        if stuff_class < len(self.class_names):
                            label = self.class_names[stuff_class]
                        else:
                            label = f"class_{stuff_class}"

                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        # Try multiple positions for better placement
                        positions_to_try = [
                            (centroid_y, centroid_x),  # Primary position
                        ]

                        # Add alternative positions for fence and pole
                        if stuff_class in [4, 5]:  # fence, pole
                            positions_to_try.extend(
                                [
                                    (centroid_y - 30, centroid_x),
                                    (centroid_y + 30, centroid_x),
                                    (centroid_y, centroid_x + 40),
                                    (centroid_y, centroid_x - 40),
                                ]
                            )

                        label_placed = False

                        for try_y, try_x in positions_to_try:
                            label_x = max(5, min(try_x - text_width // 2, w - text_width - 5))
                            label_y = max(5, min(try_y - text_height // 2, h - text_height - 5))

                            # Check overlap - be more lenient for small classes
                            overlap = False
                            min_distance = 40 if stuff_class in [4, 5] else 60

                            for lx, ly, lw, lh in labeled_regions:
                                center_dist = np.sqrt(
                                    (label_x + text_width / 2 - lx - lw / 2) ** 2
                                    + (label_y + text_height / 2 - ly - lh / 2) ** 2
                                )
                                if center_dist < min_distance:
                                    overlap = True
                                    break

                            if not overlap:
                                padding = 2
                                draw.rectangle(
                                    [
                                        label_x - padding,
                                        label_y - padding,
                                        label_x + text_width + padding,
                                        label_y + text_height + padding,
                                    ],
                                    fill=(255, 255, 255, 200),
                                )

                                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                                labeled_regions.append(
                                    (
                                        label_x - padding,
                                        label_y - padding,
                                        text_width + 2 * padding,
                                        text_height + 2 * padding,
                                    )
                                )

                                labeled_classes.add(stuff_class)
                                label_placed = True
                                break

                    else:
                        logger.debug(f"No coords found for stuff class {stuff_class}")
                else:
                    logger.debug(
                        f"Insufficient area for stuff class {stuff_class}: {total_area} < {min_area_threshold}"
                    )
            else:
                # For road (class 0), try semantic detection
                if stuff_class == 0:
                    logger.debug(f"Road not in class_segments, trying semantic detection...")
                    road_semantic_mask = ((panoptic_pred // self.label_divisor) == 0) & (panoptic_pred >= 0)

                    if road_semantic_mask.sum() > 10000:
                        coords = np.column_stack(np.where(road_semantic_mask))

                        if len(coords) > 0:
                            lower_coords = coords[coords[:, 0] > h * 0.6]
                            if len(lower_coords) > 100:
                                centroid_y = int(lower_coords[:, 0].mean())
                                centroid_x = int(lower_coords[:, 1].mean())
                            else:
                                centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                            label = "road"
                            bbox = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]

                            label_x = max(5, min(centroid_x - text_width // 2, w - text_width - 5))
                            label_y = max(5, min(centroid_y - text_height // 2, h - text_height - 5))

                            # Check overlap
                            overlap = False
                            for lx, ly, lw, lh in labeled_regions:
                                if not (
                                    label_x + text_width < lx
                                    or label_x > lx + lw
                                    or label_y + text_height < ly
                                    or label_y > ly + lh
                                ):
                                    overlap = True
                                    break

                            if not overlap:
                                padding = 3
                                draw.rectangle(
                                    [
                                        label_x - padding,
                                        label_y - padding,
                                        label_x + text_width + padding,
                                        label_y + text_height + padding,
                                    ],
                                    fill=(255, 255, 255, 220),
                                )

                                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                                labeled_regions.append(
                                    (
                                        label_x - padding,
                                        label_y - padding,
                                        text_width + 2 * padding,
                                        text_height + 2 * padding,
                                    )
                                )

                                labeled_classes.add(0)
                                logger.debug(f"Successfully labeled road using semantic detection")
                            else:
                                # fallback position for road
                                alt_x, alt_y = w // 2, int(h * 0.7)
                                label_x = max(5, min(alt_x - text_width // 2, w - text_width - 5))
                                label_y = max(5, min(alt_y - text_height // 2, h - text_height - 5))

                                padding = 3
                                draw.rectangle(
                                    [
                                        label_x - padding,
                                        label_y - padding,
                                        label_x + text_width + padding,
                                        label_y + text_height + padding,
                                    ],
                                    fill=(255, 255, 255, 240),
                                )

                                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)
                                labeled_classes.add(0)
                                logger.debug(f"Road label placed at fallback position")

                # Only mark non-road classes as processed if they're not found
                if stuff_class != 0:
                    logger.debug(f"Stuff class {stuff_class} not found in segments")

        return np.array(pil_image)

    # ---------------------------------------------------------------------
    # Consolidate nearby car segments into single labels
    # ---------------------------------------------------------------------
    def _consolidate_car_segments(self, panoptic_pred, car_segments):
        """Consolidate nearby car segments into single labels."""
        if not car_segments:
            return []

        h, w = panoptic_pred.shape
        consolidated = []
        used_segments = set()

        # Sort by area (largest first)
        car_segments.sort(key=lambda x: x[1], reverse=True)

        for pan_id, area in car_segments:
            if pan_id in used_segments:
                continue

            # Get mask for this car segment
            main_mask = panoptic_pred == pan_id
            main_coords = np.column_stack(np.where(main_mask))

            if len(main_coords) == 0:
                continue

            # Find nearby car segments to merge
            merged_mask = main_mask.copy()
            total_area = area
            used_segments.add(pan_id)

            # Calculate main segment centroid
            main_centroid = main_coords.mean(axis=0)

            # Look for other car segments within reasonable distance
            for other_id, other_area in car_segments:
                if other_id in used_segments or other_id == pan_id:
                    continue

                other_mask = panoptic_pred == other_id
                other_coords = np.column_stack(np.where(other_mask))

                if len(other_coords) == 0:
                    continue

                other_centroid = other_coords.mean(axis=0)

                # Calculate distance between segments
                distance = np.linalg.norm(main_centroid - other_centroid)

                # Merge if segments are close enough (within ~80 pixels)
                if distance < 80:
                    merged_mask |= other_mask
                    total_area += other_area
                    used_segments.add(other_id)

            consolidated.append((merged_mask, total_area))

        return consolidated
