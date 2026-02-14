# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

"""
Adapted and modified from CenterNet Pytorch (https://github.com/xingyizhou/CenterNet)
Original repository: https://github.com/xingyizhou/CenterNet
Original author: Xingyi Zhou
"""

import os
import cv2
import torch
import ttnn
import numpy as np
from PIL import Image
from loguru import logger
from typing import Dict, Tuple, Optional, Any

from models.experimental.centernet.reference.image import get_affine_transform, transform_preds
from models.experimental.centernet.reference.decode import ctdet_decode
from models.experimental.centernet.reference.debugger import Debugger


def preprocess_image(
    image_path: str,
    input_size: int = 512,
    down_ratio: int = 4,
    device: Optional[Any] = None,
) -> Tuple[torch.Tensor, Optional[ttnn.Tensor], Dict]:
    """
    Preprocess image for CenterNet.

    Args:
        image_path: Path to input image
        input_size: Model input size (default: 512)
        down_ratio: Output downsampling ratio (default: 4)
        device: TTNN device for creating device tensors (optional)

    Returns:
        Tuple of (torch_input, ttnn_input, meta_dict)
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    height, width = img.shape[0], img.shape[1]

    c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    s = max(height, width) * 1.0
    input_h, input_w = input_size, input_size

    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # Normalize
    inp = inp.astype(np.float32) / 255.0
    inp = (inp - np.array([0.408, 0.447, 0.470])) / np.array([0.289, 0.274, 0.278])
    inp = inp.transpose(2, 0, 1)

    torch_input = torch.from_numpy(inp).unsqueeze(0).float()

    ttnn_input = None
    if device is not None:
        ttnn_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)

    meta = {
        "c": c,
        "s": s,
        "out_height": input_size // down_ratio,
        "out_width": input_size // down_ratio,
    }
    return torch_input, ttnn_input, meta


def postprocess_output(output_dict: Dict[str, torch.Tensor], K: int = 100) -> torch.Tensor:
    """
    Post-process model outputs to get detections.

    Args:
        output_dict: Dictionary with 'hm', 'wh', 'reg' outputs
        K: Maximum number of detections to return

    Returns:
        Detections tensor of shape [batch, K, 6] (x1, y1, x2, y2, score, class)
    """
    hm = output_dict["hm"]
    wh = output_dict["wh"]
    reg = output_dict["reg"]
    detections = ctdet_decode(hm, wh, reg, K=K)
    return detections


def draw_detections(
    image_path: str,
    output_path: str,
    detections: torch.Tensor,
    model_name: str,
    meta: Optional[Dict] = None,
    score_threshold: float = 0.3,
) -> str:
    """
    Draw bounding boxes on image.

    Args:
        image_path: Path to input image
        output_path: Directory to save output image
        detections: Detection tensor [batch, K, 6] (x1, y1, x2, y2, score, class)
        model_name: Name for output file (e.g., 'pytorch' or 'ttnn')
        meta: Metadata dict with 'c', 's', 'out_height', 'out_width' for coordinate transform
        score_threshold: Minimum confidence score to display (default: 0.3)

    Returns:
        Path to saved output image
    """
    debugger = Debugger(dataset="coco", theme="black")
    img = cv2.imread(image_path)

    detections = detections[0].cpu().numpy()
    valid_mask = detections[:, 4] > score_threshold
    detections = detections[valid_mask]

    logger.info(f"Total detections: {len(detections)}")

    # Transform detections back to original image coordinates
    if meta is not None:
        c = meta["c"]
        s = meta["s"]
        out_h = meta["out_height"]
        out_w = meta["out_width"]

        for i in range(len(detections)):
            detections[i, :2] = transform_preds(detections[i, :2].reshape(1, 2), c, s, (out_w, out_h)).reshape(-1)
            detections[i, 2:4] = transform_preds(detections[i, 2:4].reshape(1, 2), c, s, (out_w, out_h)).reshape(-1)

    # Draw detections
    debugger.add_img(img, img_id=model_name)
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)
        logger.info(
            f"Detection: class={cls_id} ({debugger.names[cls_id]}), score={score:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )
        debugger.add_coco_bbox([int(x1), int(y1), int(x2), int(y2)], cls_id, score, img_id=model_name)

    # Save output
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{model_name}.png")
    debugger.save_all_imgs(output_path, prefix="")  # img_id already contains model_name
    logger.info(f"Saved output to {output_file} with {len(detections)} detections")

    return output_file
