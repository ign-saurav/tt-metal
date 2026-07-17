# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.common.utility_functions import tt2torch_tensor, comp_pcc
from models.experimental.centernet.reference.dlav0 import DLASeg
from models.experimental.centernet.tt.dla_seg import TtDLASeg
from models.experimental.centernet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.centernet.tt.utils import download_weights
from models.experimental.centernet.reference.model import load_model
from models.experimental.centernet.tests.perf.performant_infra import CenterNetPerformantTestInfra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Import preprocessing and visualization utilities
from models.experimental.centernet.demo.preprocess import preprocess_image, postprocess_output, draw_detections


class Demo:
    """CenterNet demo supporting both PyTorch and TTNN pipelines."""

    def __init__(self) -> None:
        self.torch_model: Optional[Any] = None
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None
        self.infra: Optional[CenterNetPerformantTestInfra] = None
        self.pipeline: Optional[Any] = None

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
        """Initialize TTNN model with performant infrastructure."""
        logger.info("Initializing TTNN CenterNet model with performant infra...")

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

        # Initialize performant infrastructure
        self.infra = CenterNetPerformantTestInfra(self.ttnn_device, self.ttnn_model, dtype=ttnn.bfloat16)
        logger.info("TTNN model with performant infra ready.")

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

    def run_ttnn_inference(self, torch_input: torch.Tensor):
        """Run TTNN inference using pipeline."""
        if self.infra is None or self.ttnn_device is None:
            raise RuntimeError("TTNN infra/device not initialized.")

        logger.info("Running TTNN inference with pipeline...")
        start = time.time()

        # Pass the original PyTorch tensor to create_pipeline_memory_configs
        ttnn_input_tensor, l1_input_memory_config, dram_input_memory_config = self.infra.create_pipeline_memory_configs(
            torch_input  # Use torch_input instead of ttnn_input
        )

        assert ttnn_input_tensor.storage_type() == ttnn.StorageType.HOST, "Input tensor must be on host"

        # Create pipeline if not already created
        if self.pipeline is None:
            self.pipeline = create_pipeline_from_config(
                config=PipelineConfig(
                    use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False
                ),
                model=self.infra,
                device=self.ttnn_device,
                dram_input_memory_config=dram_input_memory_config,
                l1_input_memory_config=l1_input_memory_config,
            )

            ttnn.synchronize_device(self.ttnn_device)
            self.pipeline.compile(ttnn_input_tensor)

        # Run inference
        outputs = self.pipeline.enqueue([ttnn_input_tensor]).pop_all()

        logger.info("TTNN inference completed in {:.4f}s", time.time() - start)
        return outputs

    def run_demo(self, image_path: str, weights_path: str, output_dir: str) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        self.initialize_torch_model(weights_path)
        self.initialize_ttnn_model(weights_path)

        torch_input, ttnn_input, meta = preprocess_image(image_path, self.input_size, self.down_ratio, self.ttnn_device)

        torch_output = self.run_torch_inference(torch_input)
        torch_detections = postprocess_output(torch_output[0], K=self.K)
        draw_detections(image_path, output_dir, torch_detections, "pytorch", meta=meta, score_threshold=0.3)

        if ttnn_input is not None:
            tt_output = self.run_ttnn_inference(torch_input)

            # Convert list output to dictionary format
            tt_output_torch = {}
            head_names = ["hm", "wh", "reg"]  # Order matches performant_infra output

            for i, head_name in enumerate(head_names):
                if i < len(tt_output[0]):
                    head_output = tt2torch_tensor(tt_output[0][i])

                    # Reshape if needed
                    if len(head_output.shape) == 4:
                        output_h = self.input_size // self.down_ratio
                        output_w = self.input_size // self.down_ratio
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

            tt_detections = postprocess_output(tt_output_torch, K=self.K)
            draw_detections(image_path, output_dir, tt_detections, "ttnn", meta=meta, score_threshold=0.5)
        else:
            logger.warning("TTNN input not available, skipping TTNN inference")

        logger.info("Demo completed. Output dir: {}", output_dir)

    def cleanup(self) -> None:
        """Release device resources."""
        if self.pipeline is not None:
            try:
                self.pipeline.cleanup()
                logger.info("Pipeline cleaned up.")
            except Exception:
                pass

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
        default=None,
        help="Path to model weights (auto-detected if not provided)",
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

    # Use download_weights utility to locate or download weights
    try:
        weights_path = download_weights(args.weights)
        logger.info(f"Using weights: {weights_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    out_dir = args.output or "models/experimental/centernet/demo/outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    demo: Optional[Demo] = None

    logger.info("=== CenterNet Demo ===")
    try:
        demo = Demo()
        demo.run_demo(args.input, weights_path, out_dir)
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
