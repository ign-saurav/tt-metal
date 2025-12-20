# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import List, Dict, Optional


# Configuration parameters for SSD512 detection post-processing
class DetectionConfig:
    def __init__(
        self,
        num_classes: int = 21,
        top_k: int = 200,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        variance: Optional[List[float]] = None,
    ):
        self.num_classes = num_classes
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance or [0.1, 0.2]


# Decodes predicted box offsets relative to prior boxes into absolute coordinates
class BoxDecoder:
    def __init__(self, variance: List[float], device):
        self.variance = variance
        self.device = device
        self._init_variance_tensors()

    def _init_variance_tensors(self):
        self.var_center = ttnn.from_torch(
            torch.tensor([self.variance[0]], dtype=torch.float32),
            device=self.device,
            dtype=ttnn.bfloat16,
        )
        self.var_size = ttnn.from_torch(
            torch.tensor([self.variance[1]], dtype=torch.float32),
            device=self.device,
            dtype=ttnn.bfloat16,
        )

    # Decodes location predictions
    def decode(self, loc_data: ttnn.Tensor, priors: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = loc_data.shape[0]
        num_priors = loc_data.shape[1]

        # Expand priors to match batch size if needed
        if len(priors.shape) == 2:
            priors = ttnn.reshape(priors, (1, num_priors, 4))
            priors = ttnn.repeat_interleave(priors, batch_size, dim=0)

        prior_centers = ttnn.slice(priors, [0, 0, 0], [batch_size, num_priors, 2])
        prior_sizes = ttnn.slice(priors, [0, 0, 2], [batch_size, num_priors, 4])

        loc_centers = ttnn.slice(loc_data, [0, 0, 0], [batch_size, num_priors, 2])
        loc_sizes = ttnn.slice(loc_data, [0, 0, 2], [batch_size, num_priors, 4])

        # Apply variance scaling and compute predicted centers
        center_offset = ttnn.multiply(loc_centers, self.var_center)
        center_offset = ttnn.multiply(center_offset, prior_sizes)
        pred_centers = ttnn.add(prior_centers, center_offset)

        # Apply variance scaling and exponential for size prediction
        size_factor = ttnn.multiply(loc_sizes, self.var_size)
        size_factor = ttnn.to_layout(size_factor, ttnn.TILE_LAYOUT)
        size_factor = ttnn.exp(size_factor)
        size_factor = ttnn.to_layout(size_factor, ttnn.ROW_MAJOR_LAYOUT)
        pred_sizes = ttnn.multiply(prior_sizes, size_factor)

        half_sizes = ttnn.multiply(pred_sizes, 0.5)
        x1y1 = ttnn.subtract(pred_centers, half_sizes)
        x2y2 = ttnn.add(pred_centers, half_sizes)

        boxes = ttnn.concat([x1y1, x2y2], dim=-1)

        return boxes


# Processes confidence predictions
class ConfidenceProcessor:
    def __init__(self, device):
        self.device = device

    # Applies softmax to convert logits to class probabilities
    def process(self, conf_data: ttnn.Tensor) -> ttnn.Tensor:
        conf_data = ttnn.to_layout(conf_data, ttnn.TILE_LAYOUT)
        conf_probs = ttnn.softmax(conf_data, dim=-1)
        conf_probs = ttnn.to_layout(conf_probs, ttnn.ROW_MAJOR_LAYOUT)
        return conf_probs


# Non-maximum suppression to remove overlapping detections
class NMSProcessor:
    def __init__(self, nms_thresh: float, top_k: int):
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    # Applies NMS: keeps highest scoring boxes, suppresses overlapping boxes above threshold
    def apply(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.long)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1) * (y2 - y1)
        _, idx = scores.sort(0, descending=True)

        keep = []
        while idx.numel() > 0:
            i = idx[0]
            keep.append(i)

            if idx.numel() == 1:
                break

            # Compute intersection over union (IoU) with remaining boxes
            xx1 = x1[idx[1:]].clamp(min=x1[i])
            yy1 = y1[idx[1:]].clamp(min=y1[i])
            xx2 = x2[idx[1:]].clamp(max=x2[i])
            yy2 = y2[idx[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)

            inter = w * h
            ovr = inter / (area[i] + area[idx[1:]] - inter)

            # Keep only boxes with IoU below threshold
            idx = idx[1:][ovr <= self.nms_thresh]

        keep = torch.tensor(keep, dtype=torch.long)
        if keep.numel() > self.top_k:
            keep = keep[: self.top_k]

        return keep


# Post-processes detections: filters by confidence threshold, applies NMS per class
class DetectionPostProcessor:
    def __init__(self, config: DetectionConfig, device):
        self.config = config
        self.device = device
        self.nms_processor = NMSProcessor(config.nms_thresh, config.top_k)

    # Processes batch: filters detections by confidence, applies NMS, returns top-k per class
    def process_batch(self, boxes: torch.Tensor, conf_probs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        batch_size = boxes.size(0)
        num_priors = boxes.size(1)

        output = []
        for batch_idx in range(batch_size):
            batch_boxes = boxes[batch_idx]
            batch_conf = conf_probs[batch_idx]

            box_list = []
            score_list = []
            label_list = []

            # Process each class
            for class_idx in range(1, self.config.num_classes):
                scores = batch_conf[:, class_idx]

                # Filter by confidence threshold
                mask = scores > self.config.conf_thresh
                if not mask.any():
                    continue

                class_scores = scores[mask]
                class_boxes = batch_boxes[mask]

                # Apply NMS to remove overlapping detections
                keep = self.nms_processor.apply(class_boxes, class_scores)

                if keep.numel() > 0:
                    box_list.append(class_boxes[keep])
                    score_list.append(class_scores[keep])
                    label_list.extend([class_idx] * keep.numel())

            if len(box_list) > 0:
                output.append(
                    {
                        "boxes": torch.cat(box_list, 0),
                        "scores": torch.cat(score_list, 0),
                        "labels": torch.tensor(label_list, dtype=torch.long),
                    }
                )
            else:
                output.append(
                    {
                        "boxes": torch.zeros((0, 4)),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )

        return output


# Complete detection pipeline: decodes boxes, processes confidence, applies NMS
class TtDetect:
    def __init__(
        self,
        num_classes: int,
        top_k: int,
        conf_thresh: float,
        nms_thresh: float,
        device=None,
    ):
        self.device = device
        self.config = DetectionConfig(num_classes, top_k, conf_thresh, nms_thresh)
        self.box_decoder = BoxDecoder(self.config.variance, device)
        self.conf_processor = ConfidenceProcessor(device)
        self.post_processor = DetectionPostProcessor(self.config, device)

    # Runs full detection pipeline: decode boxes, compute class probabilities, filter and NMS
    def __call__(
        self, loc_data: ttnn.Tensor, conf_data: ttnn.Tensor, prior_data: ttnn.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        decoded_boxes = self.box_decoder.decode(loc_data, prior_data)
        conf_probs = self.conf_processor.process(conf_data)

        # Convert to torch for post-processing (NMS uses torch operations)
        boxes_torch = ttnn.to_torch(decoded_boxes)
        conf_torch = ttnn.to_torch(conf_probs)

        return self.post_processor.process_batch(boxes_torch, conf_torch)
