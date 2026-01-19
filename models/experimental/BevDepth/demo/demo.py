# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
from loguru import logger
import torch
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
from models.experimental.BevDepth.common import run_torch_inference
from models.experimental.BevDepth.tt.custom_preprocessing import prepare_all_parameters_from_reference
from models.experimental.BevDepth.tt.ttnn_bevdepth import TtBEVDepth
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)

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

    # Run TTNN inference with pipeline
    if args.mode in ["ttnn", "both"]:
        device = ttnn.open_device(device_id=0, l1_small_size=32768)
        try:
            params, _ = prepare_all_parameters_from_reference(device)
            if params is not None:
                lss_conf = {
                    "x_bound": [-51.2, 51.2, 0.8],
                    "y_bound": [-51.2, 51.2, 0.8],
                    "z_bound": [-5.0, 3.0, 0.2],
                    "d_bound": [2.0, 58.0, 0.5],
                    "final_dim": [256, 704],
                    "downsample_factor": 16,
                    "output_channels": 80,
                }

                model_config = {
                    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
                    "WEIGHTS_DTYPE": ttnn.bfloat16,
                    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
                    "batch_size": 1,
                    "neck_in_channels": [256, 512, 1024, 2048],
                    "neck_out_channels": [128, 128, 128, 128],
                    "neck_upsample_strides": [0.25, 0.5, 1, 2],
                    "depthnet_in_channels": 512,
                    "depthnet_mid_channels": 512,
                    "depthnet_context_channels": 80,
                    "depthnet_depth_channels": 112,
                }

                ttnn_model = TtBEVDepth(
                    device=device,
                    backbone_parameters=params["backbone"],
                    neck_parameters=params["neck"],
                    depthnet_parameters=params["depthnet"],
                    head_parameters=params["head"],
                    lss_conf=lss_conf,
                    model_config=model_config,
                )

                def create_pipeline_model(model, mats, input_imgs):
                    """
                    Create a pipeline model function for BevDepth demo.

                    Args:
                        model: The BevDepth TTNN model instance
                        mats: Dictionary containing camera matrices and transformations
                        input_imgs: Input image tensor

                    Returns:
                        Callable pipeline model function
                    """

                    def run(dummy_input):
                        ttnn_output = model(input_imgs, mats)

                        output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
                        output_tensors = []

                        for task_idx in range(len(ttnn_output)):
                            for key in output_keys:
                                ttnn_tensor, _ = ttnn_output[task_idx][key]
                                if ttnn_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                                    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
                                ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, ttnn.DRAM_MEMORY_CONFIG)
                                output_tensors.append(ttnn_tensor)

                        return tuple(output_tensors)

                    return run

                pipeline_model = create_pipeline_model(ttnn_model, mats_dict, imgs)

                ttnn_input_tensor = ttnn.from_torch(
                    torch.zeros(1, 1, 1, 32),
                    device=None,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                pipeline = create_pipeline_from_config(
                    config=PipelineConfig(
                        use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False
                    ),
                    model=pipeline_model,
                    device=device,
                    dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
                )

                logger.info("Compiling and running inference...")
                pipeline.compile(ttnn_input_tensor)
                pipeline.preallocate_output_tensors_on_host(1)

                all_outputs = pipeline.enqueue([ttnn_input_tensor]).pop_all()
                pipeline.cleanup()

                last_output = all_outputs[0]

                output_keys = ["heatmap", "reg", "height", "dim", "rot", "vel"]
                ttnn_output_reconstructed = []

                output_idx = 0
                num_tasks = 6
                for task_idx in range(num_tasks):
                    task_dict = {}
                    for key in output_keys:
                        task_dict[key] = (last_output[output_idx], None)
                        output_idx += 1
                    ttnn_output_reconstructed.append(task_dict)

                torch_preds = []
                for task_idx in range(len(ttnn_output_reconstructed)):
                    task_dict = {}
                    for key in output_keys:
                        ttnn_tensor, _ = ttnn_output_reconstructed[task_idx][key]
                        tensor_torch = ttnn.to_torch(ttnn_tensor)

                        if len(tensor_torch.shape) == 4:
                            tensor_torch = tensor_torch.permute(0, 3, 1, 2).contiguous()

                        task_dict[key] = tensor_torch
                    torch_preds.append([task_dict])

                ttnn_preds = torch_preds
                boxes_ttnn, classes_ttnn, scores_ttnn = decode_predictions(ttnn_preds, class_names, args.threshold)
                pred_corners_ttnn, pred_classes_ttnn = boxes_to_corners(boxes_ttnn, classes_ttnn, args.show_range)

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
