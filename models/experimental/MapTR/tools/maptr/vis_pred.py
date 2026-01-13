import argparse
import os
import shutil
import torch
from models.experimental.MapTR.dependency import Config
from models.experimental.MapTR.dependency import MMDataParallel
from models.experimental.MapTR.dependency import load_checkpoint, wrap_fp16_model
from models.experimental.MapTR.dependency import get_logger, ProgressBar
from models.experimental.MapTR.dependency import build_dataset
from models.experimental.MapTR.projects.mmdet3d_plugin.datasets.builder import build_dataloader
from models.experimental.MapTR.dependency import build_model

# from models.experimental.MapTR.projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from models.experimental.MapTR.dependency import replace_ImageToTensor
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]
CANDIDATE = [
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
]


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def parse_args():
    parser = argparse.ArgumentParser(description="vis hdmaptr map gt label")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--score-thresh", default=0.4, type=float, help="samples to visualize")
    parser.add_argument("--show-dir", help="directory where visualizations will be saved")
    parser.add_argument("--show-cam", action="store_true", help="show camera pic")
    parser.add_argument(
        "--gt-format",
        type=str,
        nargs="+",
        default=[
            "fixed_num_pts",
        ],
        help='vis format, default should be "points",' 'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            try:
                if hasattr(cfg, "plugin_dir"):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = plugin_dir.rstrip("/").replace("/", ".")
                    if not _module_dir.startswith("models.experimental.MapTR."):
                        _module_path = "models.experimental.MapTR." + _module_dir
                    else:
                        _module_path = _module_dir
                else:
                    _module_path = "models.experimental.MapTR.projects.mmdet3d_plugin"
                print(f"Importing plugin from: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            except Exception as e:
                print(f"Warning: Failed to import plugin module {_module_path}: {e}")
                print("Trying default plugin path...")
                _module_path = "models.experimental.MapTR.projects.mmdet3d_plugin"
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0], "vis_pred")
    # create vis_label dir
    os.makedirs(osp.abspath(args.show_dir), exist_ok=True)
    import json

    with open(osp.join(args.show_dir, osp.basename(args.config)), "w") as f:
        json.dump(dict(cfg), f, indent=2)
    logger = get_logger("vis_pred")
    logger.info(f"DONE create vis_pred dir: {args.show_dir}")

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True  # TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info("Done build test data set")

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info("loading check point")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info("DONE load check point")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(img_norm_cfg["std"], dtype=np.float32)
    to_bgr = img_norm_cfg["to_rgb"]

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open("models/experimental/MapTR/figs/lidar_car.png")

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ["orange", "b", "g"]

    logger.info("BEGIN vis test dataset samples gt label & pred")

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False

    # Populate CANDIDATE from available samples in dataset
    if len(CANDIDATE) == 0:
        # Get available samples from dataset
        for i in range(min(len(dataset), 10)):  # Limit to first 10 samples
            try:
                data_info = dataset.get_data_info(i)
                if "lidar_path" in data_info:
                    lidar_filename = osp.basename(data_info["lidar_path"])
                    sample_name = lidar_filename.replace("__LIDAR_TOP__", "_").split(".")[0]
                    CANDIDATE.append(sample_name)
            except:
                continue

    prog_bar = ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        has_gt = False
        if "gt_labels_3d" in data and hasattr(data["gt_labels_3d"], "data") and len(data["gt_labels_3d"].data[0]) > 0:
            has_gt = (data["gt_labels_3d"].data[0][0] != -1).any()
        if not has_gt:
            logger.warning(f"\n empty gt for index {i}, will visualize predictions only")

        if "img" not in data:
            logger.warning(
                f"\n no img in data for index {i}, available keys: {list(data.keys())}, will continue with map visualization only"
            )
            img = None
        else:
            img = data["img"][0].data[0] if len(data["img"][0].data) > 0 else None

        img_metas_extracted = None
        if "img_metas" in data and data.get("img_metas") is not None:
            if hasattr(data["img_metas"], "data"):
                dc_data = data["img_metas"].data
                if len(dc_data) > 0:
                    img_metas_extracted = dc_data[0]
            elif isinstance(data["img_metas"], list) and len(data["img_metas"]) > 0:
                if hasattr(data["img_metas"][0], "data"):
                    img_metas_extracted = data["img_metas"][0].data[0] if len(data["img_metas"][0].data) > 0 else None
                else:
                    img_metas_extracted = data["img_metas"][0]

        img_metas = img_metas_extracted

        if "gt_bboxes_3d" in data and hasattr(data["gt_bboxes_3d"], "data"):
            gt_bboxes_3d = data["gt_bboxes_3d"].data[0] if len(data["gt_bboxes_3d"].data) > 0 else None
        else:
            gt_bboxes_3d = None

        if "gt_labels_3d" in data and hasattr(data["gt_labels_3d"], "data"):
            gt_labels_3d = data["gt_labels_3d"].data[0] if len(data["gt_labels_3d"].data) > 0 else None
        else:
            gt_labels_3d = None

        pts_filename = None
        if img_metas is not None and len(img_metas) > 0 and isinstance(img_metas[0], dict):
            pts_filename = img_metas[0].get("pts_filename", img_metas[0].get("lidar_path", ""))

        if not pts_filename:
            if "pts_filename" in data:
                if hasattr(data["pts_filename"], "data"):
                    pts_filename_raw = data["pts_filename"].data[0] if len(data["pts_filename"].data) > 0 else None
                else:
                    pts_filename_raw = data["pts_filename"]
                if isinstance(pts_filename_raw, list) and len(pts_filename_raw) > 0:
                    pts_filename = pts_filename_raw[0]
                elif isinstance(pts_filename_raw, str):
                    pts_filename = pts_filename_raw

        if not pts_filename:
            if "lidar_path" in data:
                if hasattr(data["lidar_path"], "data"):
                    lidar_path_raw = data["lidar_path"].data[0] if len(data["lidar_path"].data) > 0 else None
                else:
                    lidar_path_raw = data["lidar_path"]
                if isinstance(lidar_path_raw, list) and len(lidar_path_raw) > 0:
                    pts_filename = lidar_path_raw[0]
                elif isinstance(lidar_path_raw, str):
                    pts_filename = lidar_path_raw

        if not pts_filename or not isinstance(pts_filename, str):
            logger.error(f"\n Cannot determine pts_filename for index {i}, got: {type(pts_filename)}, skipping")
            continue

        pts_filename = osp.basename(pts_filename)
        pts_filename_processed = pts_filename.replace("__LIDAR_TOP__", "_").split(".")[0]

        if len(CANDIDATE) > 0:
            normalized_candidates = []
            for candidate in CANDIDATE:
                if "__CAM_" in candidate:
                    sample_token = candidate.split("__")[0]
                    normalized_candidates.append(sample_token)
                else:
                    normalized_candidate = candidate.replace("__LIDAR_TOP__", "_").split(".")[0]
                    normalized_candidates.append(normalized_candidate)

            sample_token = (
                pts_filename_processed.split("_")[0] if "_" in pts_filename_processed else pts_filename_processed
            )

            if sample_token not in [c.split("_")[0] if "_" in c else c for c in normalized_candidates]:
                if pts_filename_processed not in normalized_candidates:
                    logger.debug(f"Skipping sample {pts_filename_processed} - not in CANDIDATE list")
                    continue

        pts_filename = pts_filename_processed

        if img_metas is None or (isinstance(img_metas, list) and len(img_metas) == 0):
            if "img_metas" in data:
                del data["img_metas"]
            logger.info(f"img_metas missing - model will reconstruct from kwargs")

        logger.info(f"=== DEBUG: Model Input Analysis ===")
        logger.info(f"Data keys: {list(data.keys())}")
        if "img" in data:
            img_data = data["img"]
            if hasattr(img_data, "data"):
                img_tensor = img_data.data[0] if len(img_data.data) > 0 else None
                if img_tensor is not None:
                    logger.info(
                        f"Image shape: {img_tensor.shape if hasattr(img_tensor, 'shape') else type(img_tensor)}"
                    )
        if "lidar2img" in data:
            lidar2img_data = data["lidar2img"]
            if hasattr(lidar2img_data, "data"):
                lidar2img = lidar2img_data.data[0] if len(lidar2img_data.data) > 0 else None
            else:
                lidar2img = lidar2img_data
            if lidar2img is not None:
                if isinstance(lidar2img, (list, np.ndarray)) and len(lidar2img) > 0:
                    logger.info(f"lidar2img shape: {np.array(lidar2img).shape}, num_cams: {len(lidar2img)}")
                    if len(lidar2img) > 0:
                        logger.info(f"First lidar2img matrix:\n{np.array(lidar2img[0])}")
        if "can_bus" in data:
            can_bus_data = data["can_bus"]
            if hasattr(can_bus_data, "data"):
                can_bus = can_bus_data.data[0] if len(can_bus_data.data) > 0 else None
            else:
                can_bus = can_bus_data
            if can_bus is not None:
                logger.info(f"can_bus shape/len: {len(can_bus) if hasattr(can_bus, '__len__') else type(can_bus)}")
        logger.info(f"PC range: {pc_range}")

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        logger.info(f"=== DEBUG: Model Output Analysis ===")
        logger.info(
            f"Result type: {type(result)}, length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}"
        )
        if isinstance(result, (list, tuple)) and len(result) > 0:
            logger.info(f"Result[0] keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}")
        sample_dir = osp.join(args.show_dir, pts_filename)
        os.makedirs(osp.abspath(sample_dir), exist_ok=True)

        if img_metas is not None and len(img_metas) > 0 and isinstance(img_metas[0], dict):
            filename_list = img_metas[0].get("filename", [])
        else:
            if "filename" in data:
                if hasattr(data["filename"], "data"):
                    filename_list = data["filename"].data[0] if len(data["filename"].data) > 0 else []
                else:
                    filename_list = data["filename"] if isinstance(data["filename"], list) else []
            else:
                filename_list = []
        img_path_dict = {}
        for filepath in filename_list:
            if isinstance(filepath, (tuple, list)):
                filepath = filepath[0] if len(filepath) > 0 else None
            if not filepath or not isinstance(filepath, str):
                continue
            if not osp.exists(filepath):
                logger.warning(f"Image file not found: {filepath}, skipping")
                continue
            try:
                filename = osp.basename(filepath)
                filename_splits = filename.split("__")
                if len(filename_splits) < 2:
                    if "__" in filename:
                        parts = filename.split("__")
                        cam_name = parts[-1].split(".")[0] if "." in parts[-1] else parts[-1]
                    else:
                        cam_name = osp.basename(osp.dirname(filepath))
                    img_name = cam_name + ".jpg"
                else:
                    img_name = filename_splits[1] + ".jpg"
                img_path = osp.join(sample_dir, img_name)
                shutil.copyfile(filepath, img_path)
                cam_key = filename_splits[1] if len(filename_splits) >= 2 else cam_name
                img_path_dict[cam_key] = img_path
            except Exception as e:
                logger.warning(f"Failed to process image {filepath}: {e}, skipping")
                continue

        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + ".jpg"
            cam_img_path = osp.join(sample_dir, cam_img_name)
            if osp.exists(cam_img_path):
                cam_img = cv2.imread(cam_img_path)
                if cam_img is not None:
                    row_1_list.append(cam_img)

        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + ".jpg"
            cam_img_path = osp.join(sample_dir, cam_img_name)
            if osp.exists(cam_img_path):
                cam_img = cv2.imread(cam_img_path)
                if cam_img is not None:
                    row_2_list.append(cam_img)

        if len(row_1_list) > 0 or len(row_2_list) > 0:
            if len(row_1_list) > 0:
                target_height = row_1_list[0].shape[0]
                row_1_resized = []
                for img in row_1_list:
                    if img.shape[0] != target_height:
                        img = cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
                    row_1_resized.append(img)
                row_1_img = cv2.hconcat(row_1_resized) if len(row_1_resized) > 0 else None
            else:
                row_1_img = None

            if len(row_2_list) > 0:
                target_height = row_2_list[0].shape[0]
                row_2_resized = []
                for img in row_2_list:
                    if img.shape[0] != target_height:
                        img = cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
                    row_2_resized.append(img)
                row_2_img = cv2.hconcat(row_2_resized) if len(row_2_resized) > 0 else None
            else:
                row_2_img = None

            if row_1_img is not None and row_2_img is not None:
                target_width = max(row_1_img.shape[1], row_2_img.shape[1])
                if row_1_img.shape[1] != target_width:
                    row_1_img = cv2.resize(row_1_img, (target_width, row_1_img.shape[0]))
                if row_2_img.shape[1] != target_width:
                    row_2_img = cv2.resize(row_2_img, (target_width, row_2_img.shape[0]))
                cams_img = cv2.vconcat([row_1_img, row_2_img])
            elif row_1_img is not None:
                cams_img = row_1_img
            elif row_2_img is not None:
                cams_img = row_2_img
            else:
                logger.warning(f"No valid images found for surrounding view for sample {pts_filename}")
                cams_img = None

            if cams_img is not None:
                cams_img_path = osp.join(sample_dir, "surroud_view.jpg")
                cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

        for vis_format in args.gt_format:
            if not has_gt or gt_bboxes_3d is None or gt_labels_3d is None:
                logger.warning(f"Skipping GT visualization for format {vis_format} - no GT data")
                continue
            if vis_format == "se_pts":
                gt_line_points = gt_bboxes_3d[0].start_end_points
                for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1, 2).numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(
                        x[:-1],
                        y[:-1],
                        x[1:] - x[:-1],
                        y[1:] - y[:-1],
                        scale_units="xy",
                        angles="xy",
                        scale=1,
                        color=colors_plt[gt_label_3d],
                    )
            elif vis_format == "bbox":
                gt_lines_bbox = gt_bboxes_3d[0].bbox
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                    gt_bbox_3d = gt_bbox_3d.numpy()
                    xy = (gt_bbox_3d[0], gt_bbox_3d[1])
                    width = gt_bbox_3d[2] - gt_bbox_3d[0]
                    height = gt_bbox_3d[3] - gt_bbox_3d[1]
                    # import pdb;pdb.set_trace()
                    plt.gca().add_patch(
                        Rectangle(xy, width, height, linewidth=0.4, edgecolor=colors_plt[gt_label_3d], facecolor="none")
                    )
            elif vis_format == "fixed_num_pts":
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis("off")
                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d], s=2, alpha=0.8, zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_fixedpts_map_path = osp.join(sample_dir, "GT_fixednum_pts_MAP.png")
                plt.savefig(gt_fixedpts_map_path, bbox_inches="tight", format="png", dpi=1200)
                plt.close()
            elif vis_format == "polyline_pts":
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis("off")
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d], s=1, alpha=0.8, zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_polyline_map_path = osp.join(sample_dir, "GT_polyline_pts_MAP.png")
                plt.savefig(gt_polyline_map_path, bbox_inches="tight", format="png", dpi=1200)
                plt.close()

            else:
                logger.error(f"WRONG visformat for GT: {vis_format}")
                raise ValueError(f"WRONG visformat for GT: {vis_format}")

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis("off")

        result_dic = result[0]["pts_bbox"]
        boxes_3d = result_dic["boxes_3d"]  # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic["scores_3d"]
        labels_3d = result_dic["labels_3d"]
        pts_3d = result_dic["pts_3d"]

        # DEBUG: Log raw prediction info
        logger.info(f"=== DEBUG: Raw Model Output ===")
        logger.info(
            f"boxes_3d type: {type(boxes_3d)}, shape: {boxes_3d.shape if hasattr(boxes_3d, 'shape') else 'N/A'}"
        )
        logger.info(
            f"scores_3d type: {type(scores_3d)}, shape: {scores_3d.shape if hasattr(scores_3d, 'shape') else 'N/A'}"
        )
        logger.info(
            f"labels_3d type: {type(labels_3d)}, shape: {labels_3d.shape if hasattr(labels_3d, 'shape') else 'N/A'}"
        )
        logger.info(f"pts_3d type: {type(pts_3d)}, shape: {pts_3d.shape if hasattr(pts_3d, 'shape') else 'N/A'}")

        # Convert to numpy if needed (original version assumed numpy, but handle tensors)
        if hasattr(scores_3d, "cpu"):
            scores_3d = scores_3d.cpu().numpy()
        elif hasattr(scores_3d, "numpy"):
            scores_3d = scores_3d.numpy()
        elif not isinstance(scores_3d, np.ndarray):
            scores_3d = np.array(scores_3d)

        if hasattr(labels_3d, "cpu"):
            labels_3d = labels_3d.cpu().numpy()
        elif hasattr(labels_3d, "numpy"):
            labels_3d = labels_3d.numpy()
        elif not isinstance(labels_3d, np.ndarray):
            labels_3d = np.array(labels_3d)

        # Use original simple approach - no complex deduplication/filtering
        # Just apply score threshold and plot directly (like original version)
        keep = scores_3d > args.score_thresh
        num_predictions = keep.sum()
        logger.info(
            f"Found {num_predictions} predictions above threshold {args.score_thresh} (out of {len(scores_3d)} total)"
        )
        if len(scores_3d) > 0:
            logger.info(f"All scores range: [{scores_3d.min():.4f}, {scores_3d.max():.4f}]")
            logger.info(f"Unique labels: {np.unique(labels_3d)}")

        if num_predictions == 0:
            logger.warning(f"No predictions above threshold {args.score_thresh} for sample {pts_filename}")
            if len(scores_3d) > 0:
                max_score = float(scores_3d.max())
                logger.info(f"Max prediction score: {max_score:.4f}")

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis("off")

        pred_count = 0
        all_pred_x = []
        all_pred_y = []

        # Original simple approach: directly plot all predictions above threshold
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(
            scores_3d[keep], boxes_3d[keep], labels_3d[keep], pts_3d[keep]
        ):
            # Convert to numpy (original assumed numpy, but handle tensors)
            if hasattr(pred_pts_3d, "cpu"):
                pred_pts_3d = pred_pts_3d.cpu().numpy()
            elif hasattr(pred_pts_3d, "numpy"):
                pred_pts_3d = pred_pts_3d.numpy()
            elif not isinstance(pred_pts_3d, np.ndarray):
                pred_pts_3d = np.array(pred_pts_3d)

            # DEBUG: Log first few predictions
            if pred_count < 3:
                logger.info(f"=== DEBUG: Prediction {pred_count} ===")
                logger.info(f"pred_pts_3d shape: {pred_pts_3d.shape}")
                logger.info(f"pred_pts_3d sample (first 5):\n{pred_pts_3d[:5]}")
                logger.info(f"pred_label_3d: {pred_label_3d}, pred_score_3d: {pred_score_3d:.4f}")

            # Extract coordinates (original simple approach - no filtering)
            pts_x = pred_pts_3d[:, 0]
            pts_y = pred_pts_3d[:, 1]

            all_pred_x.extend(pts_x)
            all_pred_y.extend(pts_y)

            # DEBUG: Log coordinate ranges for first prediction
            if pred_count == 0:
                logger.info(
                    f"First prediction coordinate ranges - X: [{pts_x.min():.2f}, {pts_x.max():.2f}], Y: [{pts_y.min():.2f}, {pts_y.max():.2f}]"
                )
                logger.info(
                    f"PC range: X: [{pc_range[0]:.2f}, {pc_range[3]:.2f}], Y: [{pc_range[1]:.2f}, {pc_range[4]:.2f}]"
                )

            # Original simple plotting - plot all points directly
            pred_label_idx = int(pred_label_3d) if hasattr(pred_label_3d, "__int__") else pred_label_3d
            if pred_label_idx < 0 or pred_label_idx >= len(colors_plt):
                pred_label_idx = pred_label_idx % len(colors_plt) if pred_label_idx >= 0 else 0

            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_idx], linewidth=1, alpha=0.8, zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_idx], s=2, alpha=0.8, zorder=-1)
            pred_count += 1

        logger.info(f"Visualized {pred_count} predictions for sample {pts_filename}")

        # DEBUG: Compare with GT if available
        if has_gt and gt_bboxes_3d is not None and gt_labels_3d is not None and len(all_pred_x) > 0:
            try:
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                gt_all_x = []
                gt_all_y = []
                for gt_bbox_3d in gt_lines_fixed_num_pts:
                    pts = gt_bbox_3d.numpy()
                    gt_all_x.extend([pt[0] for pt in pts])
                    gt_all_y.extend([pt[1] for pt in pts])
                if len(gt_all_x) > 0:
                    logger.info(f"=== DEBUG: GT vs Pred Comparison ===")
                    logger.info(
                        f"GT X: [{min(gt_all_x):.2f}, {max(gt_all_x):.2f}], Y: [{min(gt_all_y):.2f}, {max(gt_all_y):.2f}]"
                    )
                    logger.info(
                        f"Pred X: [{min(all_pred_x):.2f}, {max(all_pred_x):.2f}], Y: [{min(all_pred_y):.2f}, {max(all_pred_y):.2f}]"
                    )
                    logger.info(f"GT center: ({np.mean(gt_all_x):.2f}, {np.mean(gt_all_y):.2f})")
                    logger.info(f"Pred center: ({np.mean(all_pred_x):.2f}, {np.mean(all_pred_y):.2f})")
                    logger.info(f"GT span: X={max(gt_all_x)-min(gt_all_x):.2f}, Y={max(gt_all_y)-min(gt_all_y):.2f}")
                    logger.info(
                        f"Pred span: X={max(all_pred_x)-min(all_pred_x):.2f}, Y={max(all_pred_y)-min(all_pred_y):.2f}"
                    )
            except Exception as e:
                logger.debug(f"Could not compare with GT: {e}")
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, "PRED_MAP_plot.png")
        plt.savefig(map_path, bbox_inches="tight", format="png", dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info("\n DONE vis test dataset samples gt label & pred")


if __name__ == "__main__":
    main()
