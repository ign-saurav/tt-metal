# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from collections import Counter
import ttnn
from typing import Dict
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class PostProcessing:
    def __init__(
        self,
        center_threshold: float = 0.1,
        nms_kernel: int = 7,
        top_k_instance: int = 200,
        thing_classes: Optional[list] = None,
        stuff_classes: Optional[list] = None,
        stuff_area_threshold: int = 4096,
        label_divisor: int = 256,
    ):
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance
        self.thing_classes = thing_classes or [11, 12, 13, 14, 15, 16, 17, 18]
        self.stuff_classes = stuff_classes or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.stuff_area_threshold = stuff_area_threshold
        self.label_divisor = label_divisor

    def find_instance_center(
        self,
        center_heatmap: torch.Tensor,
        threshold: Optional[float] = None,
        nms_kernel: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Find center points from center heatmap.
        """
        threshold = threshold if threshold is not None else self.center_threshold
        nms_kernel = nms_kernel if nms_kernel is not None else self.nms_kernel
        top_k = top_k if top_k is not None else self.top_k_instance

        # Ensure 4D tensor [B, C, H, W]
        if center_heatmap.dim() == 3:
            center_heatmap = center_heatmap.unsqueeze(0)

        # Apply threshold
        center_heatmap = F.threshold(center_heatmap, threshold, -1)

        # NMS via max pooling
        nms_padding = (nms_kernel - 1) // 2
        max_pooled = F.max_pool2d(center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding)
        center_heatmap[center_heatmap != max_pooled] = -1

        # Squeeze to 2D
        center_heatmap = center_heatmap.squeeze()
        if center_heatmap.dim() == 3:
            center_heatmap = center_heatmap[0]

        # Find non-zero elements
        if top_k is None:
            return torch.nonzero(center_heatmap > 0)
        else:
            flat = center_heatmap.flatten()
            top_k_scores, _ = torch.topk(flat, min(top_k, flat.numel()))
            return torch.nonzero(center_heatmap > top_k_scores[-1].clamp_(min=0))

    def group_pixels(self, center_points: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Assign each pixel to nearest center based on offset predictions.
        """
        _, height, width = offsets.shape
        device = offsets.device

        # Generate coordinate map
        y_coord, x_coord = torch.meshgrid(
            torch.arange(height, dtype=offsets.dtype, device=device),
            torch.arange(width, dtype=offsets.dtype, device=device),
            indexing="ij",
        )
        coord = torch.stack((y_coord, x_coord), dim=0)

        # Add offsets to get predicted centers
        center_loc = coord + offsets
        center_loc = center_loc.flatten(1).T.unsqueeze(0)  # [1, H*W, 2]
        center_points = center_points.float().unsqueeze(1)  # [K, 1, 2]

        # Compute distances
        distance = torch.norm(center_points - center_loc, dim=-1)

        # Find nearest center for each pixel
        instance_id = torch.argmin(distance, dim=0).reshape(1, height, width) + 1
        return instance_id

    def get_instance_segmentation(
        self, sem_seg: torch.Tensor, center_heatmap: torch.Tensor, offsets: torch.Tensor, thing_seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate instance segmentation from centers and offsets.
        """
        center_points = self.find_instance_center(center_heatmap)
        if center_points.size(0) == 0:
            return torch.zeros_like(sem_seg), center_points.unsqueeze(0)

        ins_seg = self.group_pixels(center_points, offsets)
        return thing_seg * ins_seg, center_points.unsqueeze(0)

    def merge_semantic_and_instance(
        self, sem_seg: torch.Tensor, ins_seg: torch.Tensor, thing_seg: torch.Tensor, void_label: int = 0
    ) -> torch.Tensor:
        """
        Merge semantic and instance predictions for panoptic output.
        """
        pan_seg = torch.full_like(sem_seg, fill_value=void_label, dtype=torch.int32)
        is_thing = (ins_seg > 0) & (thing_seg > 0)

        class_id_tracker = Counter()

        # Process thing instances
        for ins_id in torch.unique(ins_seg):
            if ins_id == 0:
                continue
            thing_mask = (ins_seg == ins_id) & is_thing
            if thing_mask.sum() == 0:
                continue
            class_id = torch.mode(sem_seg[thing_mask].view(-1))[0].item()
            if class_id not in self.thing_classes:
                continue
            class_id_tracker[class_id] += 1
            new_ins_id = class_id_tracker[class_id]
            pan_seg[thing_mask] = class_id * self.label_divisor + new_ins_id

        # Process stuff classes
        for stuff_class in self.stuff_classes:
            stuff_mask = (sem_seg == stuff_class) & (ins_seg == 0)
            if stuff_mask.sum().item() >= self.stuff_area_threshold:
                pan_seg[stuff_mask] = stuff_class * self.label_divisor

        return pan_seg

    def panoptic_fusion(
        self, semantic_logits: torch.Tensor, center_heatmap: torch.Tensor, offset_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Main panoptic segmentation fusion.
        """
        batch_size = semantic_logits.shape[0]
        semantic_pred = torch.argmax(semantic_logits, dim=1)  # [B, H, W]
        panoptic_pred = torch.zeros_like(semantic_pred, dtype=torch.int32)

        for b in range(batch_size):
            sem_seg = semantic_pred[b : b + 1]
            center_heat = center_heatmap[b : b + 1]
            offsets = offset_map[b]

            # Create thing mask
            thing_seg = torch.zeros_like(sem_seg)
            for thing_class in self.thing_classes:
                thing_seg[sem_seg == thing_class] = 1

            # Get instance segmentation
            ins_seg, _ = self.get_instance_segmentation(sem_seg, center_heat, offsets, thing_seg)

            # Merge to create panoptic
            panoptic_img = self.merge_semantic_and_instance(sem_seg[0], ins_seg[0], thing_seg[0])

            panoptic_pred[b] = panoptic_img

        return panoptic_pred

    def postprocess_outputs(
        self,
        torch_outputs: torch.Tensor,
        torch_outputs_2: torch.Tensor,
        torch_outputs_3: torch.Tensor,
        ttnn_outputs: ttnn.Tensor,
        ttnn_outputs_2: ttnn.Tensor,
        ttnn_outputs_3: ttnn.Tensor,
        original_size: Tuple[int, int],
        ttnn_device: ttnn.Device,
        output_mesh_composer: Optional[Any],
    ) -> Dict[str, Dict]:
        """Postprocess outputs from both pipelines"""
        results = {"torch": {}, "ttnn": {}}

        # Process PyTorch outputs
        if torch_outputs is not None and torch_outputs_2 is not None and torch_outputs_3 is not None:
            logger.info("Processing PyTorch outputs...")
            try:
                semantic_logits = torch_outputs
                offset_map = torch_outputs_2
                center_heatmap = torch_outputs_3

                # semantic_logits
                if semantic_logits is not None and isinstance(semantic_logits, torch.Tensor):
                    np_array = semantic_logits.squeeze(0).cpu().numpy()
                    semantic_pred = np.argmax(np_array, axis=0)
                    results["torch"]["semantic_pred"] = cv2.resize(
                        semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                    )
                    logger.debug(f"PyTorch semantic_pred shape: {results['torch']['semantic_pred'].shape}")

                # center_heatmap
                if center_heatmap is not None and isinstance(center_heatmap, torch.Tensor):
                    np_array = center_heatmap.squeeze(0).cpu().numpy()
                    if len(np_array.shape) == 3 and np_array.shape[0] == 1:
                        center_data = np_array[0]  # Remove channel dim
                    elif len(np_array.shape) == 2:
                        center_data = np_array
                    else:
                        center_data = np_array  # Use as is

                    results["torch"]["center_heatmap"] = cv2.resize(
                        center_data, original_size, interpolation=cv2.INTER_LINEAR
                    )
                    logger.debug(f"PyTorch center_heatmap shape: {results['torch']['center_heatmap'].shape}")

                # offset_map
                if offset_map is not None and isinstance(offset_map, torch.Tensor):
                    np_array = offset_map.squeeze(0).cpu().numpy()
                    if len(np_array.shape) == 3 and np_array.shape[0] == 2:
                        results["torch"]["offset_map"] = np.stack(
                            [
                                cv2.resize(np_array[0], original_size, interpolation=cv2.INTER_LINEAR),
                                cv2.resize(np_array[1], original_size, interpolation=cv2.INTER_LINEAR),
                            ]
                        )
                    else:
                        logger.warning(f"Unexpected offset_map shape: {np_array.shape}")
                # panoptic_pred
                # For PyTorch outputs:
                panoptic_pred = self.panoptic_fusion(
                    semantic_logits=semantic_logits, center_heatmap=center_heatmap, offset_map=offset_map
                )
                if panoptic_pred is not None and isinstance(panoptic_pred, torch.Tensor):
                    np_array = panoptic_pred.squeeze(0).cpu().numpy()
                    results["torch"]["panoptic_pred"] = cv2.resize(
                        np_array.astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                    )
                    logger.debug(f"PyTorch panoptic_pred shape: {results['torch']['panoptic_pred'].shape}")

            except Exception as e:
                logger.error(f"Error processing PyTorch outputs: {e}")

        # Process TTNN outputs
        if ttnn_outputs is not None and ttnn_outputs_2 is not None and ttnn_outputs_3 is not None:
            logger.info("Processing TTNN outputs...")
            try:
                # Convert TTNN to torch tensor
                torch_tensor = ttnn.to_torch(ttnn_outputs, device=ttnn_device, mesh_composer=output_mesh_composer)
                torch_tensor_2 = ttnn.to_torch(ttnn_outputs_2, device=ttnn_device, mesh_composer=output_mesh_composer)
                torch_tensor_3 = ttnn.to_torch(ttnn_outputs_3, device=ttnn_device, mesh_composer=output_mesh_composer)

                # Debug: Log raw TTNN output stats
                logger.debug(
                    f"TTNN raw output 1 (semantic): shape={torch_tensor.shape}, "
                    f"min={torch_tensor.min():.4f}, max={torch_tensor.max():.4f}, "
                    f"mean={torch_tensor.mean():.4f}"
                )
                logger.debug(
                    f"TTNN raw output 2 (offset): shape={torch_tensor_2.shape}, "
                    f"min={torch_tensor_2.min():.4f}, max={torch_tensor_2.max():.4f}, "
                    f"mean={torch_tensor_2.mean():.4f}"
                )
                logger.debug(
                    f"TTNN raw output 3 (center): shape={torch_tensor_3.shape}, "
                    f"min={torch_tensor_3.min():.4f}, max={torch_tensor_3.max():.4f}, "
                    f"mean={torch_tensor_3.mean():.4f}"
                )

                # Reshape to proper format
                reshaped_tensor = self._reshape_ttnn_output(torch_tensor, "semantic_logits")
                reshaped_tensor_2 = self._reshape_ttnn_output(torch_tensor_2, "offset_map")
                reshaped_tensor_3 = self._reshape_ttnn_output(torch_tensor_3, "center_heatmap")

                # Process semantic segmentation
                np_array = reshaped_tensor.squeeze(0).cpu().float().numpy()
                semantic_pred = np.argmax(np_array, axis=0)
                results["ttnn"]["semantic_pred"] = cv2.resize(
                    semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                )
                logger.debug(f"TTNN semantic_pred shape: {results['ttnn']['semantic_pred'].shape}")

                # Process center heatmap - FIXED: This was in elif block before
                np_array_3 = reshaped_tensor_3.squeeze(0).cpu().float().numpy()

                # Debug center heatmap values
                logger.debug(
                    f"TTNN center heatmap stats: shape={np_array_3.shape}, "
                    f"min={np_array_3.min():.4f}, max={np_array_3.max():.4f}, "
                    f"mean={np_array_3.mean():.4f}, std={np_array_3.std():.4f}"
                )

                if len(np_array_3.shape) == 3 and np_array_3.shape[0] == 1:
                    center_data = np_array_3[0]  # Remove channel dim
                elif len(np_array_3.shape) == 2:
                    center_data = np_array_3
                else:
                    center_data = np_array_3
                    logger.warning(f"Unexpected center heatmap shape: {np_array_3.shape}, using as is")

                # Check if center data has very small values and scale if needed
                if center_data.max() < 0.01:
                    logger.warning(f"Center heatmap has very small values (max={center_data.max():.6f}), scaling up")
                    center_data = center_data * 100  # Scale up for visualization

                results["ttnn"]["center_heatmap"] = cv2.resize(
                    center_data, original_size, interpolation=cv2.INTER_LINEAR
                )
                logger.debug(f"TTNN center_heatmap shape: {results['ttnn']['center_heatmap'].shape}")

                # Process offset map - FIXED: This was in elif block before
                np_array_2 = reshaped_tensor_2.squeeze(0).cpu().float().numpy()
                if len(np_array_2.shape) == 3 and np_array_2.shape[0] == 2:
                    results["ttnn"]["offset_map"] = np.stack(
                        [
                            cv2.resize(np_array_2[0], original_size, interpolation=cv2.INTER_LINEAR),
                            cv2.resize(np_array_2[1], original_size, interpolation=cv2.INTER_LINEAR),
                        ]
                    )
                    logger.debug(f"TTNN offset_map shape: {results['ttnn']['offset_map'].shape}")
                else:
                    logger.warning(f"Unexpected TTNN offset_map shape: {np_array_2.shape}")

                semantic_logits = reshaped_tensor
                center_heatmap = reshaped_tensor_3
                offset_map = reshaped_tensor_2

                panoptic_pred_ttnn = self.panoptic_fusion(
                    semantic_logits=semantic_logits, center_heatmap=center_heatmap, offset_map=offset_map
                )
                panoptic_pred = ttnn.from_torch(panoptic_pred_ttnn, dtype=ttnn.int32)

                if panoptic_pred is not None and isinstance(panoptic_pred, ttnn.Tensor):
                    # Convert TTNN tensor to numpy properly
                    torch_tensor = ttnn.to_torch(panoptic_pred, device=ttnn_device, mesh_composer=output_mesh_composer)
                    np_array = torch_tensor.squeeze().cpu().numpy()

                    # Debug information
                    logger.debug(f"Panoptic array shape: {np_array.shape}")
                    logger.debug(f"Original size for resize: {original_size}")

                    # Validate original_size before resize
                    if not original_size or len(original_size) != 2 or original_size[0] <= 0 or original_size[1] <= 0:
                        logger.error(f"Invalid original_size: {original_size}")
                        original_size = (1024, 512)
                        logger.warning(f"Using fallback size: {original_size}")

                    # Ensure valid array dimensions
                    if np_array.size > 0 and len(np_array.shape) >= 2:
                        if len(np_array.shape) > 2:
                            np_array = np_array.squeeze()

                        results["ttnn"]["panoptic_pred"] = cv2.resize(
                            np_array.astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                        )
                        logger.debug(f"TTNN panoptic_pred shape: {results['ttnn']['panoptic_pred'].shape}")
                    else:
                        logger.error("Invalid array for resize - skipping panoptic processing")

            except Exception as e:
                logger.error(f"Error processing TTNN output: {e}")
                import traceback

                traceback.print_exc()

        return results

    def _reshape_ttnn_output(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Reshape TTNN output tensor to proper format - IMPROVED VERSION"""

        logger.debug(f"Reshaping TTNN output for {key}: input shape = {tensor.shape}")

        if len(tensor.shape) == 4:  # BHWC format from TTNN
            B, H, W, C = tensor.shape

            if key == "semantic_logits":
                # Should have num_classes channels
                expected_c = 19
                if C == expected_c:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Try to reshape if flattened
                    total_elements = B * H * W * C
                    expected_h = 512 // 4  # Typical output stride
                    expected_w = 1024 // 4
                    if total_elements == B * expected_h * expected_w * expected_c:
                        tensor = tensor.reshape(B, expected_h, expected_w, expected_c)
                        result = tensor.permute(0, 3, 1, 2)
                    else:
                        logger.warning(f"Unexpected semantic_logits shape: {tensor.shape}")
                        result = tensor

            elif key == "center_heatmap":
                # Should have 1 channel
                if C == 1:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Take first channel or reshape
                    if C > 1:
                        tensor = tensor[:, :, :, :1]  # Take first channel
                    result = tensor.permute(0, 3, 1, 2)

            elif key == "offset_map":
                # Should have 2 channels (x, y offsets)
                if C == 2:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    logger.warning(f"Unexpected offset_map channels: {C}, expected 2")
                    result = tensor
            else:
                result = tensor

        elif len(tensor.shape) == 3:  # BHW format
            result = tensor

        else:
            logger.warning(f"Unexpected tensor shape for {key}: {tensor.shape}")
            result = tensor

        logger.debug(f"Reshaped TTNN output for {key}: output shape = {result.shape}")
        return result
