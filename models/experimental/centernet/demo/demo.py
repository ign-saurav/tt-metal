# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict

import torch
import ttnn
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import tt2torch_tensor, comp_pcc
from models.experimental.centernet.reference.network.dlav0 import DLASeg
from models.experimental.centernet.tt.dla_seg import TtDLASeg
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.centernet.reference.model import load_model
from models.experimental.centernet.reference.utils.decode import ctdet_decode
from models.experimental.centernet.reference.utils.debugger import Debugger
from models.experimental.centernet.reference.utils.image import get_affine_transform, transform_preds


class Demo:
    """CenterNet demo supporting both PyTorch and TTNN pipelines."""

    def __init__(self) -> None:
        self.torch_model: Optional[Any] = None
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

        self.heads = {"hm": 80, "wh": 2, "reg": 2}
        self.down_ratio = 4
        self.head_conv = 256
        self.input_size = 512
        self.K = 100

    def initialize_torch_model(self, weights_path: str) -> None:
        """Initialize PyTorch model and load weights."""
        logger.info("Initializing PyTorch CenterNet model...")
        pytorch_dla_seg = DLASeg(
            base_name="dla34",
            heads=self.heads,
            pretrained=False,
            down_ratio=self.down_ratio,
            head_conv=self.head_conv,
        )
        self.torch_model = load_model(pytorch_dla_seg, weights_path)
        self.torch_model.eval()
        logger.info("PyTorch model ready.")

    def initialize_ttnn_model(self, weights_path: str) -> None:
        """Initialize TTNN model, preprocess parameters, and build runtime graph."""
        logger.info("Initializing TTNN CenterNet model...")

        self.ttnn_device = ttnn.open_device(
            device_id=0, l1_small_size=32768, trace_region_size=1702912, num_command_queues=2
        )

        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(
            self.ttnn_device
        )

        reference_model = DLASeg(
            base_name="dla34",
            heads=self.heads,
            pretrained=False,
            down_ratio=self.down_ratio,
            head_conv=self.head_conv,
        )
        reference_model = load_model(reference_model, weights_path)
        reference_model.eval()

        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)

        logger.info("Preprocessing model parameters for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=self.ttnn_device,
        )

        parameters.layer_args = infer_ttnn_module_args(
            model=reference_model, run_model=lambda model: reference_model(dummy_input), device=self.ttnn_device
        )

        self.ttnn_model = TtDLASeg(
            heads=self.heads,
            down_ratio=self.down_ratio,
            head_conv=self.head_conv,
            parameters=parameters.dla_seg,
            device=self.ttnn_device,
            layer_args=parameters.layer_args,
        )
        logger.info("TTNN model ready.")

    def run_torch_inference(self, input_tensor: torch.Tensor):
        """Run PyTorch inference."""
        if self.torch_model is None:
            raise RuntimeError("Torch model not initialized.")
        logger.info("Running PyTorch inference...")
        start = time.time()
        with torch.no_grad():
            output = self.torch_model(input_tensor)
        logger.info("PyTorch inference completed in {:.4f}s", time.time() - start)
        return output

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor):
        """Run TTNN inference."""
        if self.ttnn_model is None or self.ttnn_device is None:
            raise RuntimeError("TTNN model/device not initialized.")
        logger.info("Running TTNN inference...")
        start = time.time()
        output = self.ttnn_model.forward(input_tensor)
        logger.info("TTNN inference completed in {:.4f}s", time.time() - start)
        return output

    def preprocess_image(self, image_path: str):
        """Preprocess image for CenterNet using original CenterNet preprocessing."""
        img = np.array(Image.open(image_path).convert("RGB"))
        height, width = img.shape[0], img.shape[1]

        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0
        input_h, input_w = self.input_size, self.input_size

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = inp.astype(np.float32) / 255.0
        inp = (inp - np.array([0.408, 0.447, 0.470])) / np.array([0.289, 0.274, 0.278])
        inp = inp.transpose(2, 0, 1)

        torch_input = torch.from_numpy(inp).unsqueeze(0).float()

        ttnn_input = None
        if self.ttnn_device is not None:
            ttnn_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
            ttnn_input = ttnn.to_device(ttnn_input, self.ttnn_device)

        meta = {
            "c": c,
            "s": s,
            "out_height": self.input_size // self.down_ratio,
            "out_width": self.input_size // self.down_ratio,
        }
        return torch_input, ttnn_input, meta

    def postprocess_output(self, output_dict: Dict[str, torch.Tensor], K=100):
        """Post-process model outputs to get detections using CenterNet decode."""
        hm = output_dict["hm"]
        wh = output_dict["wh"]
        reg = output_dict["reg"]
        detections = ctdet_decode(hm, wh, reg, K=K)
        return detections

    def draw_detections(
        self,
        image_path: str,
        output_path: str,
        detections: torch.Tensor,
        model_name: str,
        meta: dict = None,
        score_threshold: float = 0.3,
    ):
        """Draw bounding boxes on image using CenterNet's Debugger."""
        debugger = Debugger(dataset="coco", theme="black")
        img = cv2.imread(image_path)

        detections = detections[0].cpu().numpy()
        valid_mask = detections[:, 4] > score_threshold
        detections = detections[valid_mask]

        logger.info(f"Total detections: {len(detections)}")

        if meta is not None:
            c = meta["c"]
            s = meta["s"]
            out_h = meta["out_height"]
            out_w = meta["out_width"]

            for i in range(len(detections)):
                detections[i, :2] = transform_preds(detections[i, :2].reshape(1, 2), c, s, (out_w, out_h)).reshape(-1)
                detections[i, 2:4] = transform_preds(detections[i, 2:4].reshape(1, 2), c, s, (out_w, out_h)).reshape(-1)

        debugger.add_img(img, img_id=model_name)
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            cls_id = int(cls_id)
            logger.info(
                f"Detection: class={cls_id} ({debugger.names[cls_id]}), score={score:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )
            debugger.add_coco_bbox([int(x1), int(y1), int(x2), int(y2)], cls_id, score, img_id=model_name)

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{model_name}.png")
        cv2.imwrite(output_file, debugger.imgs[model_name])
        logger.info(f"Saved output to {output_file} with {len(detections)} detections")

    def run_demo(self, image_path: str, weights_path: str, output_dir: str) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        self.initialize_torch_model(weights_path)
        self.initialize_ttnn_model(weights_path)

        torch_input, ttnn_input, meta = self.preprocess_image(image_path)

        torch_output = self.run_torch_inference(torch_input)
        torch_detections = self.postprocess_output(torch_output[0], K=self.K)
        self.draw_detections(image_path, output_dir, torch_detections, "pytorch", meta=meta, score_threshold=0.3)

        if ttnn_input is not None:
            tt_output = self.run_ttnn_inference(ttnn_input)

            tt_output_torch = {}
            output_h = self.input_size // self.down_ratio
            output_w = self.input_size // self.down_ratio

            for head_name in tt_output[0]:
                head_output = tt2torch_tensor(tt_output[0][head_name])

                if len(head_output.shape) == 4:
                    if head_output.shape[1] == 1 and head_output.shape[2] == output_h * output_w:
                        num_channels = head_output.shape[3]
                        head_output = head_output.reshape(1, output_h, output_w, num_channels)
                    head_output = head_output.permute(0, 3, 1, 2)

                tt_output_torch[head_name] = head_output

            tt_output_torch["hm"] = torch.sigmoid(tt_output_torch["hm"])

            logger.info("Comparing TTNN vs PyTorch outputs:")
            for head_name in ["hm", "wh", "reg"]:
                if head_name in torch_output[0]:
                    pt_out = torch_output[0][head_name]
                    tt_out = tt_output_torch[head_name]

                    if head_name == "hm":
                        pt_out = torch.sigmoid(pt_out)

                    passing, pcc_value = comp_pcc(pt_out, tt_out, pcc=0.90)
                    logger.info(f"  {head_name}: PCC = {pcc_value:.4f}, passing = {passing}")

            tt_detections = self.postprocess_output(tt_output_torch, K=self.K)
            self.draw_detections(image_path, output_dir, tt_detections, "ttnn", meta=meta, score_threshold=0.5)
        else:
            logger.warning("TTNN input not available, skipping TTNN inference")

        logger.info("Demo completed. Output dir: {}", output_dir)

    def cleanup(self) -> None:
        """Release device resources."""
        if self.ttnn_device is not None:
            try:
                ttnn.close_device(self.ttnn_device)
                logger.info("TTNN device closed.")
            finally:
                self.ttnn_device = None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT CenterNet Demo")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        default="models/experimental/centernet/reference/resources/16004479832_a748d55f21_k.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="models/experimental/centernet/ctdet_coco_dlav0_1x.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/experimental/centernet/demo/outputs",
        help="Output directory for results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if not args.input or not os.path.exists(args.input):
        logger.error("Input image not found: {}", args.input)
        return 1

    if not os.path.exists(args.weights):
        logger.error("Weights file not found: {}", args.weights)
        return 1

    out_dir = args.output or "models/experimental/centernet/demo/outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    demo: Optional[Demo] = None

    logger.info("=== CenterNet Demo ===")
    try:
        demo = Demo()
        demo.run_demo(args.input, args.weights, out_dir)
        return 0
    except Exception as e:
        logger.exception("Demo failed: {}", e)
        return 1
    finally:
        if demo is not None:
            try:
                demo.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
