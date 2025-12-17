# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from models.experimental.SSD512.common import (
    SSD512_L1_SMALL_SIZE,
    SSD512_NUM_CLASSES,
    load_torch_model,
    generate_prior_boxes,
)
from models.experimental.SSD512.reference.data.voc0712 import VOC_CLASSES
from models.experimental.SSD512.tt.tt_ssd import build_ssd512
from models.experimental.SSD512.tt.layers.detect import TtDetect


def load_image(image_path, size=512):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    original_height, original_width = image_bgr.shape[:2]
    original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(original_img_rgb)

    x = cv2.resize(image_bgr, (size, size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    img_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    original_img.original_size = (original_width, original_height)
    return img_tensor, original_img


def filter_top_detections(detections, max_detections=2, min_score=0.1):
    if len(detections) == 0:
        return detections

    det = detections[0]
    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    if len(boxes) == 0:
        return detections

    score_mask = scores >= min_score
    if not score_mask.any():
        score_mask = torch.ones_like(scores, dtype=torch.bool)

    filtered_boxes = boxes[score_mask]
    filtered_scores = scores[score_mask]
    filtered_labels = labels[score_mask]

    if len(filtered_boxes) == 0:
        return detections

    sorted_indices = torch.argsort(filtered_scores, descending=True)
    keep_indices = sorted_indices[:max_detections]

    return [
        {
            "boxes": filtered_boxes[keep_indices],
            "scores": filtered_scores[keep_indices],
            "labels": filtered_labels[keep_indices],
        }
    ]


def draw_detections(image, detections, output_path, model_name):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    if len(detections) > 0:
        det = detections[0]
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if hasattr(image, "original_size"):
            img_width, img_height = image.original_size
        else:
            img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            box_scaled = (box * scale).cpu()
            x1, y1, x2, y2 = box_scaled[0].item(), box_scaled[1].item(), box_scaled[2].item(), box_scaled[3].item()

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            if x1 >= x2 or y1 >= y2:
                continue

            class_idx = label.item()
            class_name = VOC_CLASSES[class_idx] if class_idx < len(VOC_CLASSES) else f"Class {class_idx}"
            color = colors[class_idx % len(colors)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_text = f"{class_name}: {score.item():.2f}"
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)

    title = f"{model_name} Detections"
    bbox = draw.textbbox((10, 10), title, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    image.save(output_path)


def run_ttnn_detection(model, image_tensor, priors, device, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
    ttnn.synchronize_device(device)

    loc, conf = model.forward(image_tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    loc_torch = ttnn.to_torch(loc).float()
    conf_torch = ttnn.to_torch(conf).float()

    batch_size = 1
    loc_torch = loc_torch.reshape(batch_size, -1)
    conf_torch = conf_torch.reshape(batch_size, -1)

    num_priors = loc_torch.shape[1] // 4
    loc_torch = loc_torch.view(batch_size, num_priors, 4)
    conf_torch = conf_torch.view(batch_size, num_priors, model.num_classes)

    conf_torch = F.softmax(conf_torch, dim=-1)

    priors_ttnn = ttnn.from_torch(priors, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    loc_ttnn = ttnn.from_torch(loc_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conf_ttnn = ttnn.from_torch(conf_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    detect = TtDetect(
        num_classes=model.num_classes, top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh, device=device
    )
    detections = detect(loc_ttnn, conf_ttnn, priors_ttnn)

    if device is not None:
        ttnn.deallocate(loc_ttnn)
        ttnn.deallocate(conf_ttnn)
        ttnn.deallocate(priors_ttnn)
        ttnn.synchronize_device(device)

    return detections


def main():
    parser = argparse.ArgumentParser(description="SSD512 Demo")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/experimental/SSD512/resources/sample_output")
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_detections", type=int, default=2)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--restart_device", action="store_true")
    parser.add_argument("--l1_small_size", type=int, default=SSD512_L1_SMALL_SIZE)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch_model = load_torch_model(phase="test", size=512, num_classes=SSD512_NUM_CLASSES)

    device = None
    ttnn_model = None

    def init_device_and_model():
        nonlocal device, ttnn_model

        if device is not None:
            ttnn.close_device(device)
            gc.collect()

        if args.l1_small_size == 0:
            device = ttnn.open_device(device_id=args.device_id)
        else:
            device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)

        ttnn_model = build_ssd512(num_classes=SSD512_NUM_CLASSES, device=device)
        ttnn_model.load_weights_from_torch(torch_model)

        return device, ttnn_model

    device, ttnn_model = init_device_and_model()

    try:
        priors_torch = generate_prior_boxes()

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
            image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            raise ValueError(f"No images found in {args.input_dir}")

        for img_idx, img_path in enumerate(image_files):
            if args.restart_device and img_idx > 0:
                device, ttnn_model = init_device_and_model()

            image_tensor, original_img = load_image(str(img_path))

            ttnn.synchronize_device(device)
            gc.collect()

            try:
                ttnn_detections = run_ttnn_detection(
                    ttnn_model,
                    image_tensor,
                    priors_torch,
                    device,
                    conf_thresh=args.conf_thresh,
                    nms_thresh=args.nms_thresh,
                    top_k=args.top_k,
                )
            except RuntimeError as e:
                if "Out of Memory" in str(e) or "L1" in str(e):
                    continue
                else:
                    raise

            ttnn.synchronize_device(device)

            ttnn_detections = filter_top_detections(ttnn_detections, max_detections=args.max_detections, min_score=0.1)

            base_name = img_path.stem
            ttnn_output_path = os.path.join(args.output_dir, f"{base_name}_ttnn.jpg")
            draw_detections(original_img.copy(), ttnn_detections, ttnn_output_path, "SSD512")
            logger.info(f"Demo completed! Results saved to: {ttnn_output_path}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
