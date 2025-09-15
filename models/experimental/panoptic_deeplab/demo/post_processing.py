# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from collections import Counter


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
