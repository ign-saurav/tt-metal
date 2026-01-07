# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
from loguru import logger
import ttnn
from models.experimental.BevDepth.demo.processing import (
    RESOURCES_DIR,
    generate_infos,
    load_images_and_mats,
    load_lidar_points,
    decode_predictions,
    boxes_to_corners,
    get_gt_corners,
    visualize_results,
)
from models.experimental.BevDepth.common import run_torch_inference, run_ttnn_inference
from models.experimental.BevDepth.tt.custom_preprocessing import prepare_all_parameters_from_reference

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser(description="BEVDepth Demo - TTNN visualization")
    parser.add_argument(
        "--mode",
        choices=["ttnn", "both"],
        default="ttnn",
        help="Inference mode: 'ttnn' for TTNN only, 'both' for Torch and TTNN",
    )
    parser.add_argument("--output", default="bevdepth_demo_output.png", help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--show-range", type=float, default=60.0, help="Show range in meters")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BEVDepth Demo")
    logger.info("=" * 60)

    # Load sample data
    infos = generate_infos()
    info = infos[0]

    imgs, mats_dict, ego2global_rotation, ego2global_translation = load_images_and_mats(info)

    # Load lidar points for BEV visualization
    pts = load_lidar_points(info)

    # Get ground truth corners
    gt_corners = get_gt_corners(info, ego2global_rotation, ego2global_translation, args.show_range)

    # Class names for each task head
    class_names = [
        ["car"],
        ["truck", "construction_vehicle"],
        ["bus", "trailer"],
        ["barrier"],
        ["motorcycle", "bicycle"],
        ["pedestrian", "traffic_cone"],
    ]

    pred_corners_torch = []
    pred_classes_torch = []
    pred_corners_ttnn = None
    pred_classes_ttnn = None

    # Run Torch inference if mode is "both"
    if args.mode == "both":
        torch_preds = run_torch_inference(model=None, imgs=imgs, mats_dict=mats_dict)
        if torch_preds is not None:
            boxes_torch, classes_torch, scores_torch = decode_predictions(torch_preds, class_names, args.threshold)
            pred_corners_torch, pred_classes_torch = boxes_to_corners(boxes_torch, classes_torch, args.show_range)

    # Run TTNN inference
    if args.mode in ["ttnn", "both"]:
        device = ttnn.open_device(device_id=0, l1_small_size=32768)
        try:
            params, _ = prepare_all_parameters_from_reference(device)
            if params is not None:
                ttnn_preds = run_ttnn_inference(device, params, imgs, mats_dict)
                boxes_ttnn, classes_ttnn, scores_ttnn = decode_predictions(ttnn_preds, class_names, args.threshold)
                logger.info(f"TTNN detections: {len(boxes_ttnn)} boxes found")
                if len(boxes_ttnn) > 0:
                    pred_corners_ttnn, pred_classes_ttnn = boxes_to_corners(boxes_ttnn, classes_ttnn, args.show_range)
                else:
                    logger.warning("No TTNN detections found, check threshold or model output")
                    pred_corners_ttnn, pred_classes_ttnn = [], []
        finally:
            ttnn.close_device(device)

    # Visualize results - save to resources directory
    resources_base_dir = os.path.join(os.path.dirname(RESOURCES_DIR))
    output_path = os.path.join(resources_base_dir, args.output)
    visualize_results(
        info,
        pts,
        pred_corners_torch,
        pred_classes_torch,
        pred_corners_ttnn,
        pred_classes_ttnn,
        gt_corners,
        output_path,
        args.show_range,
    )

    logger.info("=" * 60)
    logger.info("Demo complete. Results saved to: " + output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
