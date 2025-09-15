# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import time
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger
import json
from dataclasses import asdict
import torch
import ttnn
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from post_processing import PostProcessing
from models.experimental.panoptic_deeplab.common import load_torch_model_state
from models.experimental.panoptic_deeplab.demo.config import DemoConfig
from models.experimental.panoptic_deeplab.common import parameter_conv_args, preprocess_image, save_preprocessed_inputs


class Demo:
    """Demo supporting both PyTorch and TTNN pipelines"""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.torch_model = None
        self.ttnn_model = None
        self.ttnn_device = None

        # Color palette for visualization
        self.colors = self.config._get_cityscapes_colors()

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    def initialize_torch_model(self):
        """Initialize PyTorch model"""
        logger.info("Initializing PyTorch Panoptic DeepLab model...")

        self.torch_model = TorchPanopticDeepLab().eval()
        self.torch_model = load_torch_model_state(self.torch_model, "panoptic_deeplab")

        logger.info("PyTorch model initialization completed")
        logger.info("PyTorch model initialized")

    def initialize_ttnn_model(self):
        """Initialize TTNN model"""
        logger.info("Initializing TTNN Panoptic DeepLab model...")

        # Initialize TT device
        self.ttnn_device = ttnn.open_device(device_id=self.config.device_id, l1_small_size=16384)

        # Setup mesh mappers
        self._setup_mesh_mappers()

        # Create reference model for parameter extraction
        if self.torch_model is not None:
            reference_model = self.torch_model
        else:
            reference_model = TorchPanopticDeepLab().eval()
            reference_model = load_torch_model_state(reference_model, "panoptic_deeplab")

        # Preprocess model parameters
        from ttnn.model_preprocessing import preprocess_model_parameters

        logger.info("Preprocessing model parameters for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters = parameter_conv_args(reference_model, parameters)

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
        }

        # Create TTNN model
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        logger.info("TTNN model initialized")

    def _setup_mesh_mappers(self):
        """Setup mesh mappers for multi-device support"""
        if self.ttnn_device.get_num_devices() != 1:
            self.inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.ttnn_device, dim=0)
            self.weights_mesh_mapper = None
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)
        else:
            self.inputs_mesh_mapper = None
            self.weights_mesh_mapper = None
            self.output_mesh_composer = None

    def run_torch_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run PyTorch inference"""
        logger.info("Running PyTorch inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs, outputs_2, outputs_3 = self.torch_model(input_tensor)

        inference_time = time.time() - start_time
        logger.info(f"PyTorch inference completed in {inference_time:.4f}s")

        return outputs, outputs_2, outputs_3

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        """Run TTNN inference"""
        logger.info("Running TTNN inference...")
        start_time = time.time()

        ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3 = self.ttnn_model(input_tensor, self.ttnn_device)

        inference_time = time.time() - start_time
        logger.info(f"TTNN inference completed in {inference_time:.4f}s")

        return ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3

    def compare_outputs(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Compare PyTorch and TTNN outputs"""
        if not (self.config.compare_outputs and "torch" in results and "ttnn" in results):
            return {}

        logger.info("Comparing PyTorch and TTNN outputs...")
        pcc_scores = {}

        for key in ["semantic_pred", "center_heatmap", "offset_map", "panoptic_pred"]:
            if key in results["torch"] and key in results["ttnn"]:
                torch_output = results["torch"][key]
                ttnn_output = results["ttnn"][key]

                # Debug shape information
                logger.debug(f"Comparing {key}:")
                logger.debug(f"  PyTorch shape: {torch_output.shape}")
                logger.debug(f"  TTNN shape: {ttnn_output.shape}")

                # Handle different shaped arrays
                if torch_output.shape != ttnn_output.shape:
                    logger.warning(f"  Shape mismatch for {key}: {torch_output.shape} vs {ttnn_output.shape}")
                    # Try to make compatible
                    if key == "offset_map":
                        if len(torch_output.shape) == 3 and len(ttnn_output.shape) == 2:
                            # Flatten both to same shape
                            torch_flat = torch_output.flatten()
                            ttnn_flat = ttnn_output.flatten()
                        else:
                            continue
                    else:
                        # Flatten both arrays
                        torch_flat = torch_output.flatten()
                        ttnn_flat = ttnn_output.flatten()

                        # Truncate to same length if needed
                        min_len = min(len(torch_flat), len(ttnn_flat))
                        torch_flat = torch_flat[:min_len]
                        ttnn_flat = ttnn_flat[:min_len]
                else:
                    torch_flat = torch_output.flatten()
                    ttnn_flat = ttnn_output.flatten()

                # Calculate statistics
                logger.debug(f"  PyTorch stats: mean={torch_flat.mean():.4f}, std={torch_flat.std():.4f}")
                logger.debug(f"  TTNN stats: mean={ttnn_flat.mean():.4f}, std={ttnn_flat.std():.4f}")

    #     logger.info(f"Visualization saved to: {save_path}")
    def visualize_results(self, original_image: np.ndarray, results: Dict, save_path: str):
        """Create comprehensive visualization with panoptic in separate row"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        logger.info("Creating visualization...")
        has_torch = "torch" in results and results["torch"]
        has_ttnn = "ttnn" in results and results["ttnn"]

        if has_torch and has_ttnn:
            # 4 rows for dual pipeline: original, torch outputs, ttnn outputs, panoptic comparison
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
                center_data = pipeline_results["center_heatmap"]
                if center_data.max() > center_data.min():
                    center_normalized = (center_data - center_data.min()) / (center_data.max() - center_data.min())
                else:
                    center_normalized = center_data
                ax_center.imshow(original_image, alpha=0.5)
                ax_center.imshow(center_normalized, cmap="hot", alpha=0.5, vmin=0, vmax=1)
                ax_center.set_title(f"{pipeline.upper()} Centers", fontsize=11)
            ax_center.axis("off")

            # Offset
            ax_offset = fig.add_subplot(gs[row, 2])
            if "offset_map" in pipeline_results:
                offset_data = pipeline_results["offset_map"]
                if len(offset_data.shape) == 3 and offset_data.shape[0] == 2:
                    offset_magnitude = np.sqrt(offset_data[0] ** 2 + offset_data[1] ** 2)
                else:
                    offset_magnitude = offset_data
                vmax = offset_magnitude.max() if offset_magnitude.max() > 0 else 1
                im = ax_offset.imshow(offset_magnitude, cmap="viridis", vmin=0, vmax=vmax)
                ax_offset.set_title(f"{pipeline.upper()} Offset", fontsize=11)

                # Add small colorbar
                divider = make_axes_locatable(ax_offset)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            ax_offset.axis("off")

            # Hide the 4th column in these rows
            ax_empty = fig.add_subplot(gs[row, 3])
            ax_empty.axis("off")

        # Row 3 (or 2 for single pipeline): Panoptic comparison
        if has_torch and has_ttnn:
            # Both panoptic side by side
            for i, pipeline in enumerate(pipelines):
                ax_pan = fig.add_subplot(gs[3, i * 2 : (i * 2) + 2])
                # Column 3: Panoptic segmentation
                if "panoptic_pred" in pipeline_results:
                    panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                    ax_pan.imshow(panoptic_colored)
                    ax_pan.set_title(f"{pipeline.upper()} Panoptic", fontsize=10)
                    ax_pan.axis("off")
        else:
            # Single panoptic centered
            pipeline = pipelines[0]
            ax_pan = fig.add_subplot(gs[2, :])
            if "panoptic_pred" in results[pipeline]:
                panoptic_colored = self._colorize_panoptic(results[pipeline]["panoptic_pred"])
                ax_pan.imshow(panoptic_colored)
                ax_pan.set_title(f"{pipeline.upper()} Panoptic Segmentation", fontsize=12, fontweight="bold")
            ax_pan.axis("off")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Visualization saved to: {save_path}")

    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored image"""
        colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for class_id in range(self.config.num_classes):
            mask = segmentation == class_id
            if class_id < len(self.colors):
                colored[mask] = self.colors[class_id]
        return colored

    def _colorize_panoptic(self, panoptic: np.ndarray) -> np.ndarray:
        """Convert panoptic prediction to colored image"""
        colored = np.zeros((*panoptic.shape, 3), dtype=np.uint8)
        label_divisor = 256

        unique_ids = np.unique(panoptic)

        for pan_id in unique_ids:
            # Don't skip 0 - it's a valid panoptic ID for road
            mask = panoptic == pan_id
            semantic_class = pan_id // label_divisor
            instance_id = pan_id % label_divisor

            if semantic_class < len(self.colors):
                if instance_id == 0:
                    # Stuff class - use base color
                    colored[mask] = self.colors[semantic_class]
                else:
                    # Thing instance - create variation
                    np.random.seed(int(pan_id))
                    base_color = self.colors[semantic_class].astype(float)
                    # Create variation for different instances
                    variation = np.random.randint(-40, 40, 3)
                    instance_color = np.clip(base_color + variation, 0, 255)
                    colored[mask] = instance_color.astype(np.uint8)
            else:
                # Unknown class
                colored[mask] = [128, 128, 128]

        return colored

    def save_results(self, results: Dict, original_image: np.ndarray, output_dir: str, filename: str):
        """Save all results to output directory"""
        os.makedirs(output_dir, exist_ok=True)

        # Save original image
        original_path = os.path.join(output_dir, f"{filename}_original.png")
        Image.fromarray(original_image).save(original_path)

        # Save results for each pipeline
        for pipeline, pipeline_results in results.items():
            pipeline_dir = os.path.join(output_dir, pipeline)
            os.makedirs(pipeline_dir, exist_ok=True)
            # Save panoptic segmentation
            if "panoptic_pred" in pipeline_results:
                panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic.png")
                Image.fromarray(panoptic_colored).save(panoptic_path)

                # Save raw panoptic prediction
                raw_panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic_raw.npy")
                np.save(raw_panoptic_path, pipeline_results["panoptic_pred"])

        logger.info(f"Results saved to: {output_dir}")

    def run_demo(self, image_path: str, output_dir: str):
        """Run complete demo pipeline"""
        logger.info(f"Starting demo for image: {image_path}")

        self.current_image_path = image_path
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize models
        logger.info("Initializing models...")
        self.initialize_torch_model()
        self.initialize_ttnn_model()

        # Preprocess image
        torch_input, ttnn_input, original_image, original_size = preprocess_image(
            image_path, self.config.input_width, self.config.input_height, self.ttnn_device, self.inputs_mesh_mapper
        )

        base_name = Path(image_path).stem
        torch_input_path = save_preprocessed_inputs(torch_input, output_dir, base_name)
        logger.info(f"Preprocessed inputs saved for testing: {torch_input_path}")

        # Run inference
        torch_outputs, torch_outputs_2, torch_outputs_3 = self.run_torch_inference(torch_input)
        ttnn_outputs, ttnn_outputs_2, ttnn_outputs_3 = self.run_ttnn_inference(ttnn_input)

        # Postprocess results
        results = PostProcessing().postprocess_outputs(
            torch_outputs,
            torch_outputs_2,
            torch_outputs_3,
            ttnn_outputs,
            ttnn_outputs_2,
            ttnn_outputs_3,
            original_size,
            self.ttnn_device,
            self.output_mesh_composer,
        )
        # Compare outputs if both pipelines ran
        self.compare_outputs(results)

        # Generate filename
        base_name = Path(image_path).stem

        # Save individual results
        self.save_results(results, original_image, output_dir, base_name)

        # Create visualization
        viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        self.visualize_results(original_image, results, viz_path)

        # Save metadata and results summary
        self._save_metadata(image_path, results, output_dir, base_name)

        logger.info(f"Demo completed! Results saved to: {output_dir}")

        # Cleanup
        if ttnn_input is not None:
            ttnn.deallocate(ttnn_input)

    def _save_metadata(self, image_path: str, results: Dict, output_dir: str, filename: str):
        # def _save_metadata(self, image_path: str, results: Dict, pcc_scores: Dict, output_dir: str, filename: str):
        """Save metadata and comparison results"""
        metadata = {
            "image_path": image_path,
            "config": asdict(self.config),
            "results": {
                "pipelines_run": list(results.keys()),
                # "pcc_scores": pcc_scores,
            },
            "output_files": {
                "visualization": f"{filename}_comparison.png",
                "original": f"{filename}_original.png",
            },
        }

        # Add pipeline-specific metadata
        for pipeline in results.keys():
            metadata["output_files"][pipeline] = {
                "semantic": f"{pipeline}/{filename}_semantic.png",
                "centers": f"{pipeline}/{filename}_centers.png",
                "panoptic": f"{pipeline}/{filename}_panoptic.png",
            }

        metadata_path = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def cleanup(self):
        """Cleanup resources"""
        if self.ttnn_device is not None:
            ttnn.close_device(self.ttnn_device)
            logger.info("TTNN device closed")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="TT Panoptic DeepLab Demo with Default Config")
    parser.add_argument("--input", "-i", type=str, required=False, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=False, help="Output directory for results")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return 1

    # Initialize configuration
    config = DemoConfig()

    # Initialize demo
    logger.info("=== Panoptic DeepLab Demo ===")

    try:
        demo = Demo(config)
        demo.run_demo(args.input, args.output)
        logger.info("Demo completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        try:
            demo.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit(main())
