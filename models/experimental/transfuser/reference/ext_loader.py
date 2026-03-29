# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch
import torchvision.ops as ops


class CPUNMSWrapper:
    """Wrapper providing NMS functions using torchvision (CPU compatible)."""

    @staticmethod
    def nms(boxes, scores, iou_threshold, offset=0):
        """Non-maximum suppression using torchvision."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        return ops.nms(boxes, scores, iou_threshold)

    @staticmethod
    def softnms(boxes, scores, iou_threshold, sigma=0.5, min_score=0.001, method=1, offset=0):
        """Soft-NMS fallback - uses regular NMS as approximation."""
        return CPUNMSWrapper.nms(boxes, scores, iou_threshold, offset)

    @staticmethod
    def nms_match(dets, iou_threshold):
        """NMS match - fallback to regular NMS."""
        if dets.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=dets.device)
        boxes = dets[:, :4]
        scores = dets[:, 4]
        return ops.nms(boxes, scores, iou_threshold)

    @staticmethod
    def nms_rotated(dets, scores, iou_threshold, labels=None):
        """Rotated NMS fallback - uses regular NMS as approximation."""
        if dets.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=dets.device)
        # Convert rotated boxes to axis-aligned for approximation
        if dets.shape[1] == 5:
            x_c, y_c, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            half_w, half_h = w / 2, h / 2
            boxes = torch.stack([x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h], dim=1)
        else:
            boxes = dets[:, :4]
        return ops.nms(boxes, scores, iou_threshold)


class ExtWrapper:
    """Wrapper to provide attribute access to extension functions."""

    def __init__(self, ext):
        self._ext = ext

    def __getattr__(self, name):
        return getattr(self._ext, name)


if torch.__version__ != "parrots":

    def load_ext(name, funcs):
        """Load extension module with CPU-compatible implementations."""
        if name == "_ext":
            # Return CPU-compatible NMS wrapper for _ext
            return ExtWrapper(CPUNMSWrapper)
        else:
            # For other extensions, try to import from mmcv
            try:
                ext = importlib.import_module("mmcv." + name)
                for fun in funcs:
                    assert hasattr(ext, fun), f"{fun} miss in module {name}"
                return ext
            except (ImportError, ModuleNotFoundError) as e:
                raise ImportError(
                    f"Cannot load extension '{name}'. " f"MMCV CUDA extensions are not available. " f"Error: {e}"
                )

else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        "nms",
        "softnms",
        "nms_match",
        "nms_rotated",
        "top_pool_forward",
        "top_pool_backward",
        "bottom_pool_forward",
        "bottom_pool_backward",
        "left_pool_forward",
        "left_pool_backward",
        "right_pool_forward",
        "right_pool_backward",
        "fused_bias_leakyrelu",
        "upfirdn2d",
        "ms_deform_attn_forward",
        "pixel_group",
        "contour_expand",
        "diff_iou_rotated_sort_vertices_forward",
    ]

    def get_fake_func(name, e):
        def fake_func(*args, **kwargs):
            warnings.warn(f"{name} is not supported in parrots now")
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple("ExtModule", funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if "No element registered" not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist() -> bool:
    ext_loader = pkgutil.find_loader("mmcv._ext")
    return ext_loader is not None
