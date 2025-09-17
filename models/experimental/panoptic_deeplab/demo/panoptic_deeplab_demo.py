# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import ttnn
from PIL import Image
from loguru import logger

from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from post_processing import PostProcessing
from models.experimental.panoptic_deeplab.demo.config import DemoConfig
from models.experimental.panoptic_deeplab.common import (
    _populate_all_decoders,
    preprocess_image,
    save_preprocessed_inputs,
    load_torch_model_state,
)


class Demo:
    """Panoptic-DeepLab demo supporting both PyTorch and TTNN pipelines."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.torch_model: Optional[TorchPanopticDeepLab] = None
        self.ttnn_model: Optional[TTPanopticDeepLab] = None
        self.ttnn_device: Optional[Any] = None

        # Visualization palette (Cityscapes)
        self.colors = self.config._get_cityscapes_colors()

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def initialize_torch_model(self) -> None:
        """Initialize PyTorch model and load weights."""
        logger.info("Initializing PyTorch Panoptic-DeepLab model…")
        model = TorchPanopticDeepLab().eval()
        self.torch_model = load_torch_model_state(model, "panoptic_deeplab")
        logger.info("PyTorch model ready.")

    def initialize_ttnn_model(self) -> None:
        """Initialize TTNN model, preprocess parameters, and build runtime graph."""
        logger.info("Initializing TTNN Panoptic-DeepLab model…")

        # Initialize TT device (L1 size tuned for demo)
        self.ttnn_device = ttnn.open_device(device_id=self.config.device_id, l1_small_size=16384)

        # Setup mesh mappers
        self._setup_mesh_mappers()

        # Create reference torch model to extract parameters
        reference_model = self.torch_model or load_torch_model_state(TorchPanopticDeepLab().eval(), "panoptic_deeplab")

        # Preprocess model parameters
        from ttnn.model_preprocessing import preprocess_model_parameters

        logger.info("Preprocessing model parameters for TTNN…")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        _populate_all_decoders(reference_model, parameters)

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
        }

        # Create TTNN model
        self.ttnn_model = TTPanopticDeepLab(parameters=parameters, model_config=model_config)
        logger.info("TTNN model ready.")

    def _setup_mesh_mappers(self) -> None:
        """Setup mesh mappers for multi-device support."""
        if self.ttnn_device.get_num_devices() != 1:
            self.inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.ttnn_device, dim=0)
            self.weights_mesh_mapper = None
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)
        else:
            self.inputs_mesh_mapper = None
            self.weights_mesh_mapper = None
            self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def run_torch_inference(self, input_tensor: torch.Tensor):
        """Run PyTorch inference."""
        if self.torch_model is None:
            raise RuntimeError("Torch model not initialized.")
        logger.info("Running PyTorch inference…")
        start = time.time()
        with torch.no_grad():
            outputs = self.torch_model(input_tensor)  # expected tuple of three heads
        logger.info("PyTorch inference completed in {:.4f}s", time.time() - start)
        return outputs  # (outputs, outputs_2, outputs_3)

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor):
        """Run TTNN inference."""
        if self.ttnn_model is None or self.ttnn_device is None:
            raise RuntimeError("TTNN model/device not initialized.")
        logger.info("Running TTNN inference…")
        start = time.time()
        outputs = self.ttnn_model(input_tensor, self.ttnn_device)  # expected tuple of three heads
        logger.info("TTNN inference completed in {:.4f}s", time.time() - start)
        return outputs  # (ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3)

    # ---------------------------------------------------------------------
    # Comparison / Visualization / I/O
    # ---------------------------------------------------------------------

    def compare_outputs(self, results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """Compare PyTorch and TTNN outputs via Pearson correlation (PCC).

        Returns a dict of PCC scores per key, logging stats and handling shape mismatches robustly.
        """
        if not (self.config.compare_outputs and "torch" in results and "ttnn" in results):
            return {}

        logger.info("Comparing PyTorch and TTNN outputs…")
        pcc_scores: Dict[str, float] = {}
        keys = ["semantic_pred", "center_heatmap", "offset_map", "panoptic_pred"]

        for key in keys:
            torch_arr = results.get("torch", {}).get(key)
            ttnn_arr = results.get("ttnn", {}).get(key)
            if torch_arr is None or ttnn_arr is None:
                logger.debug("Skipping {}: missing in one of the pipelines.", key)
                continue

            # Flatten and align lengths (robust to layout differences)
            t_flat = np.asarray(torch_arr).ravel()
            n_flat = np.asarray(ttnn_arr).ravel()
            if t_flat.size == 0 or n_flat.size == 0:
                logger.debug("Skipping {}: empty output.", key)
                continue

            if t_flat.size != n_flat.size:
                min_sz = min(t_flat.size, n_flat.size)
                logger.warning(
                    "Shape mismatch for {} ({} vs {}), truncating to {}.", key, t_flat.size, n_flat.size, min_sz
                )
                t_flat = t_flat[:min_sz]
                n_flat = n_flat[:min_sz]

            # Compute PCC
            if np.std(t_flat) == 0 or np.std(n_flat) == 0:
                logger.warning("Zero-variance arrays for {}, PCC undefined. Skipping.", key)
                continue

            pcc = float(np.corrcoef(t_flat, n_flat)[0, 1])
            pcc_scores[key] = pcc
            logger.info("PCC[{}] = {:.4f}", key, pcc)

        return pcc_scores

    def visualize_results(
        self, original_image: np.ndarray, results: Dict[str, Dict[str, np.ndarray]], save_path: str
    ) -> None:
        """Create a comprehensive visualization with panoptic results optionally side-by-side."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # local import to avoid heavy import on startup

        logger.info("Creating visualization…")
        has_torch = "torch" in results and results["torch"]
        has_ttnn = "ttnn" in results and results["ttnn"]

        if has_torch and has_ttnn:
            # 4 rows: original, torch outputs, ttnn outputs, panoptic comparison
            fig = plt.figure(figsize=(16, 20))
            gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 1, 1, 1], hspace=0.3, wspace=0.2)
            pipelines = ["torch", "ttnn"]
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1, 1], hspace=0.3, wspace=0.2)
            pipelines = ["torch"] if has_torch else ["ttnn"]

        # Row 0: Original image (centered)
        ax_orig = fig.add_subplot(gs[0, 1:3])
        ax_orig.imshow(original_image)
        ax_orig.set_title("Original Image", fontsize=14, fontweight="bold")
        ax_orig.axis("off")

        # Hide unused cells in first row
        for i in [0, 3]:
            ax = fig.add_subplot(gs[0, i])
            ax.axis("off")

        # Rows 1-2: Pipeline outputs (semantic, centers, offset)
        for i, pipeline in enumerate(pipelines):
            if pipeline not in results:
                continue
            pipeline_results = results[pipeline]
            row = i + 1

            if self.config.save_heads:
                # Semantic segmentation
                ax_sem = fig.add_subplot(gs[row, 0])
                if "semantic_pred" in pipeline_results:
                    semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                    ax_sem.imshow(semantic_colored)
                    ax_sem.set_title(f"{pipeline.upper()} Semantic", fontsize=11)
                ax_sem.axis("off")

                # Centers
                ax_center = fig.add_subplot(gs[row, 1])
                if "center_heatmap" in pipeline_results:
                    center = pipeline_results["center_heatmap"]
                    cmin, cmax = np.min(center), np.max(center)
                    norm = (center - cmin) / (cmax - cmin + 1e-8)
                    ax_center.imshow(original_image, alpha=0.5)
                    ax_center.imshow(norm, cmap="hot", alpha=0.5, vmin=0, vmax=1)
                    ax_center.set_title(f"{pipeline.upper()} Centers", fontsize=11)
                ax_center.axis("off")

                # Offset magnitude
                ax_offset = fig.add_subplot(gs[row, 2])
                if "offset_map" in pipeline_results:
                    offset = pipeline_results["offset_map"]
                    if offset.ndim == 3 and offset.shape[0] == 2:
                        mag = np.sqrt(offset[0] ** 2 + offset[1] ** 2)
                    else:
                        mag = np.asarray(offset)
                    vmax = float(np.max(mag)) if np.max(mag) > 0 else 1.0
                    im = ax_offset.imshow(mag, cmap="viridis", vmin=0, vmax=vmax)
                    ax_offset.set_title(f"{pipeline.upper()} Offset", fontsize=11)

                    # Small colorbar
                    divider = make_axes_locatable(ax_offset)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                ax_offset.axis("off")

                # Hide the 4th column in these rows
                fig.add_subplot(gs[row, 3]).axis("off")

        # Row 3 (or 2 for single pipeline): Panoptic comparison
        if self.config.save_panoptic:
            if has_torch and has_ttnn:
                for i, pipeline in enumerate(pipelines):
                    ax_pan = fig.add_subplot(gs[3, i * 2 : (i * 2) + 2])
                    if "panoptic_pred" in results[pipeline]:
                        panoptic_colored = self._colorize_panoptic(results[pipeline]["panoptic_pred"])
                        ax_pan.imshow(panoptic_colored)
                        ax_pan.set_title(f"{pipeline.upper()} Panoptic", fontsize=10)
                        ax_pan.axis("off")
            else:
                # Single pipeline panoptic in last row (row=2)
                ax_pan = fig.add_subplot(gs[2, 1:3])
                single = pipelines[0]
                if "panoptic_pred" in results[single]:
                    panoptic_colored = self._colorize_panoptic(results[single]["panoptic_pred"])
                    ax_pan.imshow(panoptic_colored)
                    ax_pan.set_title(f"{single.upper()} Panoptic", fontsize=10)
                ax_pan.axis("off")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Visualization saved: {}", save_path)

    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to color image using the predefined palette."""
        h, w = segmentation.shape[-2], segmentation.shape[-1]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(min(self.config.num_classes, len(self.colors))):
            mask = segmentation == class_id
            colored[mask] = self.colors[class_id]
        return colored

    def _colorize_panoptic(self, panoptic: np.ndarray) -> np.ndarray:
        """Convert panoptic prediction to colored image with per-instance color jitter."""
        colored = np.zeros((*panoptic.shape, 3), dtype=np.uint8)
        taken = set()

        def pick_color(base: np.ndarray, instance_id: int, max_dist: int = 40) -> np.ndarray:
            rng = np.random.default_rng(instance_id)
            for attempt in range(8):
                if attempt == 0:
                    rgb = tuple(int(x) for x in base)
                else:
                    jitter = rng.integers(-max_dist, max_dist + 1, size=3)
                    rgb = tuple(int(np.clip(base.astype(float) + jitter, 0, 255)[i]) for i in range(3))
                if rgb not in taken:
                    taken.add(rgb)
                    return np.array(rgb, dtype=np.uint8)
            return base

        unique_ids = np.unique(panoptic)
        for pan_id in unique_ids:
            mask = panoptic == pan_id
            semantic_class = int(pan_id // self.config.label_divisor)
            instance_id = int(pan_id % self.config.label_divisor)

            if pan_id == 0:
                colored[mask] = self.colors[0]
                continue

            if semantic_class < len(self.colors):
                base = self.colors[semantic_class]
                if instance_id == 0:  # stuff
                    colored[mask] = base
                else:  # thing instance
                    colored[mask] = pick_color(base, int(pan_id))
            else:
                colored[mask] = np.array([128, 128, 128], dtype=np.uint8)  # unknown

        return colored

    def save_results(self, results: Dict[str, Dict[str, np.ndarray]], output_dir: str, filename: str) -> None:
        """Save panoptic predictions (and optionally heads) to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for pipeline, pipeline_results in results.items():
            # choose destination dir
            if self.config.compare_outputs:
                pipeline_dir = os.path.join(output_dir, pipeline)
            else:
                pipeline_dir = "models/experimental/panoptic_deeplab/resources"
            Path(pipeline_dir).mkdir(parents=True, exist_ok=True)

            # Required: panoptic
            if "panoptic_pred" in pipeline_results:
                panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic.png")
                Image.fromarray(panoptic_colored).save(panoptic_path)

            # Optional heads (if requested)
            if self.config.save_heads:
                if "semantic_pred" in pipeline_results:
                    semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                    Image.fromarray(semantic_colored).save(os.path.join(pipeline_dir, f"{filename}_semantic.png"))
                if "center_heatmap" in pipeline_results:
                    center = pipeline_results["center_heatmap"]
                    cmin, cmax = float(np.min(center)), float(np.max(center))
                    norm = (center - cmin) / (cmax - cmin + 1e-8)
                    plt.figure()
                    plt.imshow(norm, cmap="hot", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.savefig(os.path.join(pipeline_dir, f"{filename}_centers.png"), dpi=150, bbox_inches="tight")
                    plt.close()

        logger.info("Results saved under {}", output_dir)

    def _save_metadata(
        self, image_path: str, results: Dict[str, Dict[str, Any]], output_dir: str, filename: str
    ) -> None:
        """Save metadata and comparison manifest for downstream analysis."""
        meta = {
            "image_path": image_path,
            "config": asdict(self.config),
            "results": {"pipelines_run": list(results.keys())},
            "output_files": {
                "visualization": f"{filename}_comparison.png",
                "original": f"{filename}_original.png",
            },
        }

        for pipeline in results.keys():
            meta["output_files"][pipeline] = {
                "semantic": f"{pipeline}/{filename}_semantic.png",
                "centers": f"{pipeline}/{filename}_centers.png",
                "panoptic": f"{pipeline}/{filename}_panoptic.png",
            }

        path = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Metadata saved: {}", path)

    def run_demo(self, image_path: str, output_dir: str) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        # Initialize models (Torch + TTNN)
        self.initialize_torch_model()
        self.initialize_ttnn_model()

        # Preprocess image
        torch_input, ttnn_input, original_image, original_size = preprocess_image(
            image_path, self.config.input_width, self.config.input_height, self.ttnn_device, self.inputs_mesh_mapper
        )

        base_name = Path(image_path).stem
        # Save preprocessed inputs (torch tensor + stats) for reproducibility
        _ = save_preprocessed_inputs(torch_input, output_dir, base_name)

        # Run inference
        torch_outputs = self.run_torch_inference(torch_input)  # (o1, o2, o3)
        ttnn_outputs = self.run_ttnn_inference(ttnn_input)  # (o1, o2, o3)

        # Postprocess to comparable outputs
        results = PostProcessing().postprocess_outputs(
            *torch_outputs,
            *ttnn_outputs,
            original_size,
            self.ttnn_device,
            self.output_mesh_composer,
        )

        # Compare outputs (optional)
        if self.config.compare_outputs:
            _ = self.compare_outputs(results)
        else:
            logger.info("Skipping output comparison per config.")

        # Persist artifacts
        if self.config.save_results:
            self.save_results(results, output_dir, base_name)

        if self.config.save_visualization:
            viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            self.visualize_results(original_image, results, viz_path)

        if self.config.save_comparison:
            self._save_metadata(image_path, results, output_dir, base_name)

        # Save original image for reference
        Image.fromarray(original_image).save(os.path.join(output_dir, f"{base_name}_original.png"))

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
    parser = argparse.ArgumentParser(description="TT Panoptic-DeepLab Demo")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default="outputs/demo", help="Output directory for results")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Validate input file
    if not args.input or not os.path.exists(args.input):
        logger.error("Input image not found: {}", args.input)
        return 1

    # Prepare output directory
    out_dir = args.output or "outputs/demo"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    config = DemoConfig()
    demo: Optional[Demo] = None

    logger.info("=== Panoptic-DeepLab Demo ===")
    try:
        demo = Demo(config)
        demo.run_demo(args.input, out_dir)
        logger.info("Demo completed successfully.")
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
