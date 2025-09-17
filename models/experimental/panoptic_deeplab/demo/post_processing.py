# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class PostProcessing:
    """
    Post-processing for Panoptic-DeepLab, aligned with Detectron2 implementation.
    Reference: https://github.com/facebookresearch/detectron2/blob/main/projects/Panoptic-DeepLab/
    """

    def __init__(
        self,
        center_threshold: float = 0.1,
        nms_kernel: int = 7,
        top_k_instance: int = 200,
        thing_classes: Optional[list] = None,
        stuff_classes: Optional[list] = None,
        stuff_area_threshold: int = 2048,
        label_divisor: int = 256,
        instance_score_threshold: float = 0.5,
    ):
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance
        self.thing_classes = thing_classes or [11, 12, 13, 14, 15, 16, 17, 18]
        self.stuff_classes = stuff_classes or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.stuff_area_threshold = stuff_area_threshold
        self.label_divisor = label_divisor
        self.instance_score_threshold = instance_score_threshold

        # Create lookup sets for faster checking
        self.thing_set = set(self.thing_classes)
        self.stuff_set = set(self.stuff_classes)

    def find_instance_center(self, center_heatmap, threshold=0.1, nms_kernel=3, top_k=None):
        """
        Find the center points from the center heatmap.
        """
        # Thresholding, setting values below threshold to -1.
        center_heatmap = F.threshold(center_heatmap, threshold, -1)

        # NMS
        nms_padding = (nms_kernel - 1) // 2
        center_heatmap_max_pooled = F.max_pool2d(center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding)
        center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1

        # Squeeze first two dimensions.
        center_heatmap = center_heatmap.squeeze()
        assert len(center_heatmap.size()) == 2, "Something is wrong with center heatmap dimension."

        # Find non-zero elements.
        if top_k is None:
            return torch.nonzero(center_heatmap > 0)
        else:
            # find top k centers.
            top_k_scores, _ = torch.topk(torch.flatten(center_heatmap), top_k)
            return torch.nonzero(center_heatmap > top_k_scores[-1].clamp_(min=0))

    def group_pixels(self, center_points, offsets):
        """
        Gives each pixel in the image an instance id.
        """
        height, width = offsets.size()[1:]

        # Generates a coordinate map, where each location is the coordinate of
        # that location.
        y_coord, x_coord = torch.meshgrid(
            torch.arange(height, dtype=offsets.dtype, device=offsets.device),
            torch.arange(width, dtype=offsets.dtype, device=offsets.device),
            indexing="ij",
        )
        coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

        center_loc = coord + offsets
        center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
        center_points = center_points.unsqueeze(1)  # [K, 1, 2]

        # Distance: [K, H*W].
        distance = torch.norm(center_points - center_loc, dim=-1)

        # Finds center with minimum distance at each location, offset by 1, to
        # reserve id=0 for stuff.
        instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
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

        # Paste stuff to unoccupied area.
        class_ids = torch.unique(sem_seg)
        for class_id in class_ids:
            if class_id.item() in thing_ids:
                # thing class
                continue
            # Calculate stuff area.
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
            if stuff_mask.sum().item() >= stuff_area:
                pan_seg[stuff_mask] = class_id * label_divisor
            pan_seg = torch.clamp(pan_seg, min=0)

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
        threshold=0.1,
        nms_kernel=7,
        top_k=200,
        foreground_mask=None,
    ):
        """
        Post-processing for panoptic segmentation.
        """
        if sem_seg.dim() == 4 and sem_seg.shape[1] > 1:
            sem_seg = torch.argmax(sem_seg, dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            sem_seg = sem_seg

        if sem_seg.dim() == 4 and sem_seg.shape[0] == 1:
            sem_seg = sem_seg.squeeze(0)  # [1, H, W]
        if center_heatmap.dim() == 2:  # [H, W]
            center_heatmap = center_heatmap.unsqueeze(0)
        elif center_heatmap.dim() == 4:  # [B, 1, H, W]
            center_heatmap = center_heatmap.squeeze(0)
        if offsets.dim() == 2:  # [H, W]
            offsets = offsets.unsqueeze(0)
        elif offsets.dim() == 4:  # [B, 2, H, W]
            offsets = offsets.squeeze(0)
        if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
            raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))

        if center_heatmap.dim() != 3:
            raise ValueError("Center prediction with un-supported dimension: {}.".format(center_heatmap.dim()))
        if offsets.dim() != 3:
            raise ValueError("Offset prediction with un-supported dimension: {}.".format(offsets.dim()))
        if foreground_mask is not None:
            if foreground_mask.dim() != 3 and foreground_mask.size(0) != 1:
                raise ValueError("Foreground prediction with un-supported shape: {}.".format(sem_seg.size()))
            thing_seg = foreground_mask
        else:
            # inference from semantic segmentation
            thing_seg = torch.zeros_like(sem_seg)
            for thing_class in list(thing_ids):
                thing_seg[sem_seg == thing_class] = 1

        if sem_seg.shape[1] > 1:  # Multi-class
            semantic_pred = torch.argmax(sem_seg, dim=1)
        else:
            semantic_pred = sem_seg.squeeze(1)

        instance, center = self.get_instance_segmentation(
            semantic_pred,
            center_heatmap,
            offsets,
            thing_seg,
            thing_ids,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )
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

                # Optional: save center heatmap
                center_np = torch_outputs_3[0, 0].cpu().numpy()
                results["torch"]["center_heatmap"] = cv2.resize(
                    center_np, original_size, interpolation=cv2.INTER_LINEAR
                )

                # Optional: save offset map
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
                    panoptic_pred[0].cpu().numpy().astype(np.int32), original_size, interpolation=cv2.INTER_NEAREST
                )

                # Optional outputs
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
        if len(tensor.shape) == 4:
            # Assume NHWC format from TTNN
            N, H, W, C = tensor.shape

            if key == "semantic_logits":
                expected_c = 19  # Cityscapes classes
                if C == expected_c:
                    return tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW

            elif key == "center_heatmap":
                if C == 1 or C == 19:  # Could be 1 channel or per-class
                    return tensor.permute(0, 3, 1, 2)

            elif key == "offset_map":
                if C == 2:  # x, y offsets
                    return tensor.permute(0, 3, 1, 2)

        return tensor
