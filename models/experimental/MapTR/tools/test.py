# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import os
import warnings
import numpy as np
import random
import torch
import re

# from mmcv import Config, DictAction
from models.experimental.MapTR.dependency import Config, DictAction

# from mmcv.cnn import fuse_conv_bn
from models.experimental.MapTR.dependency import fuse_conv_bn
from models.experimental.MapTR.dependency import MMDataParallel
from models.experimental.MapTR.dependency import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from models.experimental.MapTR.dependency import (
    single_gpu_test,
    build_dataset,
    build_dataloader,
    build_model,
    set_random_seed,
)
from models.experimental.MapTR.dependency import replace_ImageToTensor
import time
import os.path as osp
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase" "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument("--gpu-collect", action="store_true", help="whether to use gpu to collect results.")
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = plugin_dir.rstrip("/").replace("/", ".")
                _module_path = "models.experimental.MapTR." + _module_dir
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed} (matches training)")

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    if "meta" in checkpoint:
        print("Checkpoint meta:", checkpoint["meta"])
    if "config" in checkpoint.get("meta", {}):
        config_str = checkpoint["meta"]["config"]
        num_vec_match = re.search(r"num_vec=(\d+)", config_str)
        num_pts_match = re.search(r"num_pts_per_vec=(\d+)", config_str)
        num_query_match = re.search(r"num_query=(\d+)", config_str)
        ckpt_num_vec = int(num_vec_match.group(1)) if num_vec_match else None
        ckpt_num_pts = int(num_pts_match.group(1)) if num_pts_match else None
        ckpt_num_query = int(num_query_match.group(1)) if num_query_match else None
        if num_vec_match:
            print(f"Checkpoint num_vec: {ckpt_num_vec}")
        if num_pts_match:
            print(f"Checkpoint num_pts_per_vec: {ckpt_num_pts}")
        if num_query_match:
            print(f"Checkpoint num_query: {ckpt_num_query}")
        if hasattr(model, "pts_bbox_head"):
            model_num_query = model.pts_bbox_head.num_query if hasattr(model.pts_bbox_head, "num_query") else None
            model_num_vec = model.pts_bbox_head.num_vec if hasattr(model.pts_bbox_head, "num_vec") else None
            model_num_pts = (
                model.pts_bbox_head.num_pts_per_vec if hasattr(model.pts_bbox_head, "num_pts_per_vec") else None
            )
            print(f"Model num_query: {model_num_query}, num_vec: {model_num_vec}, num_pts_per_vec: {model_num_pts}")
            if ckpt_num_query and model_num_query and ckpt_num_query != model_num_query:
                print(f"WARNING: Query count mismatch! Checkpoint: {ckpt_num_query}, Model: {model_num_query}")
            if ckpt_num_vec and model_num_vec and ckpt_num_vec != model_num_vec:
                print(f"WARNING: num_vec mismatch! Checkpoint: {ckpt_num_vec}, Model: {model_num_vec}")
            if ckpt_num_pts and model_num_pts and ckpt_num_pts != model_num_pts:
                print(f"WARNING: num_pts_per_vec mismatch! Checkpoint: {ckpt_num_pts}, Model: {model_num_pts}")

    # Debug: Check reference points and query embeddings
    import os

    debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
    if debug_enabled:
        print("\n=== Checking Reference Points and Query Embeddings ===")
        # Check if reference_points Linear layer exists
        if hasattr(model, "pts_bbox_head") and hasattr(model.pts_bbox_head, "transformer"):
            transformer = model.pts_bbox_head.transformer
            if hasattr(transformer, "reference_points"):
                ref_pts_layer = transformer.reference_points
                print(
                    f"Reference points Linear layer: weight shape={ref_pts_layer.weight.shape}, bias shape={ref_pts_layer.bias.shape}"
                )
                print(
                    f"Reference points weight: mean={ref_pts_layer.weight.mean():.4f}, std={ref_pts_layer.weight.std():.4f}"
                )
                print(
                    f"Reference points bias: mean={ref_pts_layer.bias.mean():.4f}, std={ref_pts_layer.bias.std():.4f}"
                )

                # Check if weights look randomly initialized (low std suggests random init)
                if ref_pts_layer.weight.std() < 0.1:
                    print("WARNING: Reference points Linear layer weights look randomly initialized!")
                    print("They may not have been loaded from checkpoint properly.")

            # Check query embedding
            if hasattr(model.pts_bbox_head, "query_embedding"):
                query_embed = model.pts_bbox_head.query_embedding
                if query_embed is not None:
                    print(f"Query embedding: shape={query_embed.weight.shape}")
                    print(
                        f"Query embedding weight: mean={query_embed.weight.mean():.4f}, std={query_embed.weight.std():.4f}"
                    )
                else:
                    print("Query embedding is None (using instance_pts mode)")
                    # Check for instance/pts embeddings
                    if hasattr(model.pts_bbox_head, "instance_embedding"):
                        inst_embed = model.pts_bbox_head.instance_embedding
                        if inst_embed is not None:
                            print(f"Instance embedding: shape={inst_embed.weight.shape}")
                    if hasattr(model.pts_bbox_head, "pts_embedding"):
                        pts_embed = model.pts_bbox_head.pts_embedding
                        if pts_embed is not None:
                            print(f"Pts embedding: shape={pts_embed.weight.shape}")

                # Check checkpoint keys
                checkpoint_keys = list(checkpoint.get("state_dict", checkpoint).keys())
                ref_point_keys = [k for k in checkpoint_keys if "reference_points" in k]
                query_embed_keys = [k for k in checkpoint_keys if "query_embedding" in k]
                print(
                    f"Reference point keys in checkpoint: {ref_point_keys[:5]}..."
                    if len(ref_point_keys) > 5
                    else f"Reference point keys in checkpoint: {ref_point_keys}"
                )
                print(
                    f"Query embedding keys in checkpoint: {query_embed_keys[:5]}..."
                    if len(query_embed_keys) > 5
                    else f"Query embedding keys in checkpoint: {query_embed_keys}"
                )

            fix_ref_points = os.environ.get("MAPTR_FIX_REF_POINTS", "0") == "1"
            if fix_ref_points and hasattr(transformer, "reference_points"):
                print("\n=== Fixing Reference Points Initialization ===")
                ref_pts_layer = transformer.reference_points
                state_dict = checkpoint.get("state_dict", checkpoint)
                ckpt_w_key = "pts_bbox_head.transformer.reference_points.weight"
                ckpt_b_key = "pts_bbox_head.transformer.reference_points.bias"
                if ckpt_w_key in state_dict and ckpt_b_key in state_dict:
                    ckpt_weight = state_dict[ckpt_w_key]
                    ckpt_bias = state_dict[ckpt_b_key]
                    if not torch.allclose(ref_pts_layer.weight, ckpt_weight, atol=1e-3):
                        print(f"Restoring reference points from checkpoint...")
                        with torch.no_grad():
                            ref_pts_layer.weight.copy_(ckpt_weight)
                            ref_pts_layer.bias.copy_(ckpt_bias)
                        print(
                            f"Restored: weight_std={ref_pts_layer.weight.std():.4f}, bias={ref_pts_layer.bias.tolist()}"
                        )
                    else:
                        print(f"Reference points already match checkpoint")
                else:
                    num_queries = model.pts_bbox_head.num_query if hasattr(model.pts_bbox_head, "num_query") else 1000
                    h, w = 50, 20
                    if h * w != num_queries:
                        h = int(np.sqrt(num_queries))
                        w = num_queries // h
                    y_coords = torch.linspace(0.05, 0.95, h)
                    x_coords = torch.linspace(0.05, 0.95, w)
                    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
                    ref_pts_uniform = torch.stack(
                        [grid_x.flatten()[:num_queries], grid_y.flatten()[:num_queries]], dim=-1
                    )
                    ref_pts_logit = torch.log(ref_pts_uniform / (1 - ref_pts_uniform + 1e-6))
                    with torch.no_grad():
                        ref_pts_layer.bias[0].fill_(ref_pts_logit[:, 0].mean().item())
                        ref_pts_layer.bias[1].fill_(ref_pts_logit[:, 1].mean().item())
                        torch.nn.init.normal_(ref_pts_layer.weight, mean=0.0, std=0.5)
                    print(
                        f"Fixed {num_queries} queries: range=[{ref_pts_uniform.min():.4f}, {ref_pts_uniform.max():.4f}]"
                    )

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
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

    # if not distributed:
    #     assert False
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
    # args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            assert False
            # mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs["jsonfile_prefix"] = osp.join(
            "test", args.config.split("/")[-1].split(".")[-2], time.ctime().replace(" ", "_").replace(":", "_")
        )
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            # Remove internal config keys (starting with _)
            eval_kwargs = {k: v for k, v in eval_kwargs.items() if not k.startswith("_")}
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
