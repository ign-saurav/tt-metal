# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Compare PyTorch and TTNN intermediate tensors saved as *.npy.

Usage:
  python -m models.demos.yolov5x.tests.pcc.compare_intermediates [--dir INTERMEDIATES_DIR]
  # or from repo root after pytest has run:
  python models/demos/yolov5x/tests/pcc/compare_intermediates.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two flattened arrays."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    if a_flat.size != b_flat.size:
        return float("nan")
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def try_align_ttnn_to_pt(ttnn_arr: np.ndarray, pt_shape: tuple) -> np.ndarray | None:
    """
    Align TTNN tensor to PyTorch shape by transpose/reshape (no flattening).
    - 4D PT (N, C, H, W): TTNN (1, 1, H*W, C) -> transpose to (N, C, H, W).
    - 3D PT (N, C, L): TTNN (1, 1, L, C) -> transpose to (N, C, L).
    Returns aligned array or None if not possible.
    """
    if ttnn_arr.shape == pt_shape:
        return ttnn_arr.copy()
    if ttnn_arr.size != np.prod(pt_shape):
        return None
    # TTNN convention: (1, 1, spatial, C) or (1, 1, L, C)
    if ttnn_arr.ndim != 4 or ttnn_arr.shape[0] != 1 or ttnn_arr.shape[1] != 1:
        return None
    _, _, dim1, C = ttnn_arr.shape  # dim1 is spatial (H*W) or L

    if len(pt_shape) == 4:
        # PT NCHW (N, C, H, W)
        N, c_pt, H, W = pt_shape
        if c_pt != C or N * H * W != dim1:
            return None
        # (1, 1, N*H*W, C) -> (N, H, W, C) then transpose -> (N, C, H, W)
        ttnn_reshaped = ttnn_arr.reshape(1, 1, N, H, W, C).squeeze(0).squeeze(0)  # (N, H, W, C)
        return np.transpose(ttnn_reshaped, (0, 3, 1, 2))  # (N, C, H, W)

    if len(pt_shape) == 3:
        # PT (N, C, L) e.g. Detect x_cat, box, cls, out
        N, c_pt, L = pt_shape
        if c_pt != C or N * L != dim1:
            return None
        # (1, 1, L, C) -> transpose to (1, 1, C, L) -> reshape (N, C, L)
        ttnn_transposed = np.transpose(ttnn_arr, (0, 1, 3, 2))  # (1, 1, C, L)
        return ttnn_transposed.reshape(N, C, L)

    return None


def compare_pair(pt_path: Path, ttnn_path: Path) -> dict:
    """Load pt and ttnn .npy files, align shapes if needed, and compute metrics."""
    pt = np.load(pt_path)
    ttnn = np.load(ttnn_path)

    pt_shape_orig = pt.shape
    ttnn_shape_orig = ttnn.shape

    # Align TTNN to PT shape by transpose/reshape (no flattening to force layout)
    if ttnn.shape != pt.shape:
        ttnn_aligned = try_align_ttnn_to_pt(ttnn, pt.shape)
        if ttnn_aligned is not None:
            ttnn = ttnn_aligned
        else:
            return {
                "pt_path": str(pt_path.name),
                "ttnn_path": str(ttnn_path.name),
                "pt_shape": pt_shape_orig,
                "ttnn_shape": ttnn_shape_orig,
                "pcc": float("nan"),
                "max_abs_diff": float("nan"),
                "mean_abs_diff": float("nan"),
                "error": "shape/layout incompatible, could not align (no flatten)",
            }
    # pt and ttnn now have same shape; compute PCC on aligned elements
    pt_flat = pt.astype(np.float64).ravel()
    ttnn_flat = ttnn.astype(np.float64).ravel()
    pcc = pearson_correlation(pt, ttnn)
    max_abs_diff = float(np.max(np.abs(pt_flat - ttnn_flat)))
    mean_abs_diff = float(np.mean(np.abs(pt_flat - ttnn_flat)))

    return {
        "pt_path": pt_path.name,
        "ttnn_path": ttnn_path.name,
        "pt_shape": pt_shape_orig,
        "ttnn_shape": ttnn_shape_orig,
        "pcc": pcc,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PyTorch vs TTNN intermediate .npy tensors")
    default_dir = Path(__file__).resolve().parent / "intermediates"
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help=f"Directory containing pt_*.npy and ttnn_*.npy files (default: {default_dir})",
    )
    parser.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.99,
        help="PCC threshold for pass/fail (default: 0.99)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip printing header line",
    )
    args = parser.parse_args()
    dir_path = args.dir

    if not dir_path.is_dir():
        print(f"Error: directory not found: {dir_path}", file=sys.stderr)
        return 1

    pt_files = sorted(dir_path.glob("pt_*.npy"))
    if not pt_files:
        print(f"No pt_*.npy files in {dir_path}", file=sys.stderr)
        return 1

    results = []
    for pt_path in pt_files:
        # pt_00_conv_in.npy -> ttnn_00_conv_in.npy
        # pt_12_concat_in_0.npy -> ttnn_12_concat_in_0.npy
        suffix = pt_path.name.replace("pt_", "ttnn_", 1)
        ttnn_path = dir_path / suffix
        if not ttnn_path.exists():
            results.append(
                {
                    "pt_path": pt_path.name,
                    "ttnn_path": suffix,
                    "pt_shape": None,
                    "ttnn_shape": None,
                    "pcc": float("nan"),
                    "max_abs_diff": float("nan"),
                    "mean_abs_diff": float("nan"),
                    "error": "ttnn file not found",
                }
            )
            continue
        results.append(compare_pair(pt_path, ttnn_path))

    # Report
    if not args.no_header:
        print(f"Comparing intermediates in: {dir_path}")
        print(f"PCC threshold: {args.pcc_threshold}")
        print()
        print(
            f"{'Name':<30} {'PT shape':<24} {'TTNN shape':<24} {'PCC':>10} {'Max|diff|':>12} {'Mean|diff|':>12} {'Status':<8}"
        )
        print("-" * 120)

    failed = 0
    for r in results:
        name = r["pt_path"].replace(".npy", "")
        pt_s = str(r["pt_shape"]) if r["pt_shape"] is not None else "?"
        tt_s = str(r["ttnn_shape"]) if r["ttnn_shape"] is not None else "?"
        pcc = r["pcc"]
        max_d = r["max_abs_diff"]
        mean_d = r["mean_abs_diff"]
        if r.get("error"):
            status = "SKIP" if "not found" in r["error"] else "ERR"
            if "ERR" in status:
                failed += 1
            print(f"{name:<30} {pt_s:<24} {tt_s:<24} {'N/A':>10} {'N/A':>12} {'N/A':>12} {status:<8}  # {r['error']}")
        else:
            ok = not np.isnan(pcc) and pcc >= args.pcc_threshold
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
            print(f"{name:<30} {pt_s:<24} {tt_s:<24} {pcc:>10.6f} {max_d:>12.6g} {mean_d:>12.6g} {status:<8}")

    if not args.no_header:
        print("-" * 120)
        print(f"Total: {len(results)} pairs, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
